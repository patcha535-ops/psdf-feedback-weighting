"""
psdf_core.py -- simulation engine for the Punch Score Decision Framework (PSDF) study.

Implements the protocol of the manuscript equation by equation (Eq. N = manuscript equation number).
All randomness is derived from one master seed (SeedSequence), with separate streams per replication for
environment, outcome/label noise, AHP judgment noise, feedback-mask keys and preference-gap draw, so that
within a replication every method (and every experimental cell) sees the same generated environment.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import stats

RI = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}          # Saaty random indices; RI_6 = 1.24 is used in the paper


def pairs(nA: int):
    return [(j, l) for j in range(nA) for l in range(j + 1, nA)]


# ----------------------------------------------------------------------------- configuration (Table 1)
@dataclass(frozen=True)
class Config:
    m: int = 30
    n: int = 5
    T: int = 60
    R: int = 100
    reversal: int = 30
    w_pre: tuple = (0.70, 0.10, 0.08, 0.07, 0.05)        # Eq. 18
    w_post: tuple = (0.05, 0.07, 0.08, 0.10, 0.70)
    rho: float = 0.5
    lam: float = 1.0
    alpha: float = 0.1
    L: int = 15
    nmin: int = 12                                        # minimum observed records to fit the ridge
    gamma: float = 0.9                                    # discount factor of the exponentially weighted ridge
    sd_outcome: float = 0.05
    sd_label: float = 0.05
    sd_ahp: float = 0.05
    gap: float = 0.0                                      # stated-vs-outcome preference gap g (Sec. 7.5)
    k_primary: int = 10
    k_sens: tuple = (5, 20)
    q: float = 1.0
    seed: int = 6621
    beta_a: float = 2.0
    beta_b: float = 2.0
    n_boot: int = 10000
    rank_metrics: bool = True                             # Kendall / top-5 (skipped in the factorial for speed)
    recov_factor: float = 1.25
    recov_run: int = 3

    def ks(self):
        return sorted(set(self.k_sens) | {self.k_primary})


def semi_config(cfg: Config) -> Config:
    """Semi-synthetic variant (Sec. 7.6): n = 3 benefit criteria, weights reverse (0.60,0.25,0.15)<->(0.15,0.25,0.60)."""
    return replace(cfg, n=3, w_pre=(0.60, 0.25, 0.15), w_post=(0.15, 0.25, 0.60))


def true_weights(cfg: Config) -> np.ndarray:
    W = np.empty((cfg.T, cfg.n))
    W[:cfg.reversal] = cfg.w_pre
    W[cfg.reversal:] = cfg.w_post
    return W


def stated_vstar(cfg: Config, w_true_t: np.ndarray, gz: np.ndarray) -> np.ndarray:
    """Stated AHP target (Eq. 10 with the Sec. 7.5 distortion): benefit weights w* (.) exp(g z), renormalised;
    the safety weight rho is not distorted."""
    w = w_true_t * np.exp(cfg.gap * gz) if cfg.gap > 0 else w_true_t
    w = w / w.sum()
    return np.append(w, cfg.rho) / (1.0 + cfg.rho)


# ----------------------------------------------------------------------------- random environment
@dataclass
class RepData:
    Z: np.ndarray      # (T,m,n)
    R: np.ndarray      # (T,m)
    eps: np.ndarray    # (T,m) standard normal, scaled by sd_outcome at use (Eq. 2)
    eta: np.ndarray    # (T,m) standard normal, scaled by sd_label at use (Eq. 3)
    ahp: np.ndarray    # (T,npairs) standard normal AHP log-noise; row = elicitation time
    keys: np.ndarray   # (T,m) uniform keys -> which labels are observed (nested across q)
    gz: np.ndarray     # (n,) preference-gap direction z ~ N(0,I), fixed within a replication


def rep_seeds(cfg: Config):
    return np.random.SeedSequence(cfg.seed).spawn(cfg.R)


def make_rep_data(cfg: Config, ss, feature_table: np.ndarray | None = None) -> RepData:
    s_feat, s_out, s_ahp, s_mask, s_gap = ss.spawn(5)
    rf = np.random.default_rng(s_feat)
    if feature_table is None:                                        # synthetic: independent Beta(2,2)
        Z = rf.beta(cfg.beta_a, cfg.beta_b, size=(cfg.T, cfg.m, cfg.n))
    else:                                                            # semi-synthetic: m of P real units per period
        P = feature_table.shape[0]
        assert feature_table.shape[1] == cfg.n and P >= cfg.m
        idx = np.argsort(rf.random((cfg.T, P)), axis=1)[:, :cfg.m]   # sampling without replacement
        Z = feature_table[idx]
    R = rf.beta(cfg.beta_a, cfg.beta_b, size=(cfg.T, cfg.m))
    ro = np.random.default_rng(s_out)
    eps, eta = ro.standard_normal((cfg.T, cfg.m)), ro.standard_normal((cfg.T, cfg.m))
    nA = cfg.n + 1
    ahp = np.random.default_rng(s_ahp).standard_normal((cfg.T, nA * (nA - 1) // 2))
    keys = np.random.default_rng(s_mask).random((cfg.T, cfg.m))
    gz = np.random.default_rng(s_gap).standard_normal(cfg.n)
    return RepData(Z, R, eps, eta, ahp, keys, gz)


def obs_mask(keys: np.ndarray, q: float) -> np.ndarray:
    """round(q*m) labels observed per period, uniform without replacement; Python round (half to even):
    q = .25,.50,.75,1 with m = 30 gives 8, 15, 22, 30 observed (Sec. 7.4)."""
    m = keys.shape[1]
    k = max(1, min(m, int(round(q * m))))
    idx = np.argsort(keys, axis=1)[:, :k]
    mask = np.zeros(keys.shape, bool)
    np.put_along_axis(mask, idx, True, axis=1)
    return mask


# ----------------------------------------------------------------------------- outcome-based weight estimators (Eq. 7-9)
def weighted_ridge_slopes(X, y, w, alpha):
    """argmin sum w (y - b0 - X b)^2 + alpha ||b||^2 with an unpenalised intercept (profiled out by centering)."""
    sw = w.sum()
    xm, ym = (w[:, None] * X).sum(0) / sw, (w * y).sum() / sw
    Xc, yc = X - xm, y - ym
    return np.linalg.solve(Xc.T @ (w[:, None] * Xc) + alpha * np.eye(X.shape[1]), Xc.T @ (w * yc))


class OutcomeWeights:
    """Feedback-driven benefit weights from COMPLETED periods only. mode in {rolling, expanding, ewma, static}.

    rolling   : calendar window W_t = {s : max(0,t-L) <= s < t}                     (primary PSDF)
    expanding : all completed periods
    ewma      : all completed periods, record weight gamma^(t-1-s)
    static    : first estimate used (t = 1, after period-zero feedback) frozen thereafter (fixed-weight PSDF, Sec. 4.4)
    The ridge is fitted only if >= nmin observed records are in the window; otherwise uniform weights (Eq. 9).
    """

    def __init__(self, mode, n, L, alpha, nmin, gamma):
        self.mode, self.n, self.L, self.alpha, self.nmin, self.gamma = mode, n, L, alpha, nmin, gamma
        self._p = []                 # (t, X, y) for periods with >= 1 observed label
        self._last = -1
        self._frozen = None

    def n_completed(self):
        return self._last + 1

    def update(self, t, X, y):
        if t != self._last + 1:
            raise ValueError(f"out-of-order update: got t={t}, expected {self._last + 1}")
        self._last = t
        if len(y) > 0:
            self._p.append((t, np.array(X, float), np.array(y, float)))

    def _fit(self, t, mode):
        uni = np.full(self.n, 1.0 / self.n)
        if mode == "rolling" or mode == "static":
            win = [p for p in self._p if p[0] >= max(0, t - self.L)] if mode == "rolling" else list(self._p)
        else:
            win = list(self._p)
        n_obs = sum(len(p[2]) for p in win)
        info = dict(n_obs=n_obs, n_periods=len(win), fallback=False)
        if n_obs < self.nmin:
            info["fallback"] = True
            return uni, info
        X, y = np.vstack([p[1] for p in win]), np.concatenate([p[2] for p in win])
        if mode == "ewma":
            wts = np.concatenate([np.full(len(p[2]), self.gamma ** (t - 1 - p[0])) for p in win])
        else:
            wts = np.ones(len(y))
        beta = weighted_ridge_slopes(X, y, wts, self.alpha)
        pos = np.maximum(0.0, beta[:self.n])                        # Eq. 8: risk slope is not turned into a weight
        s = pos.sum()
        if not np.isfinite(s) or s <= 1e-12:                        # Eq. 9
            info["fallback"] = True
            return uni, info
        return pos / s, info

    def weights(self, t):
        if self._last != t - 1:
            raise ValueError(f"decision at t={t} requires history through t-1, have {self._last}")
        uni = np.full(self.n, 1.0 / self.n)
        if t == 0:                                                  # Sec. 4.3: no labels yet
            return uni, dict(n_obs=0, n_periods=0, fallback=False)
        if self.mode == "static":
            if self._frozen is None:                                # first estimate, after period-zero feedback
                self._frozen = self._fit(t, "static")
            return self._frozen
        return self._fit(t, self.mode)


def critic_weights(Zt: np.ndarray) -> np.ndarray:
    """CRITIC objective weights from the current feature matrix (no outcome feedback)."""
    rng_ = Zt.max(0) - Zt.min(0)
    N = np.where(rng_ > 0, (Zt - Zt.min(0)) / np.where(rng_ > 0, rng_, 1.0), 0.0)
    sd = N.std(0, ddof=1)
    r = np.nan_to_num(np.corrcoef(N, rowvar=False), nan=0.0)
    c = sd * (1.0 - r).sum(1)
    s = c.sum()
    return c / s if s > 1e-12 else np.full(Zt.shape[1], 1.0 / Zt.shape[1])


# ----------------------------------------------------------------------------- AHP (Eq. 12-15)
@dataclass
class AHPResult:
    v: np.ndarray
    ci: float
    cr: float
    clipped: int
    npairs: int


def ahp_elicit(v_stated: np.ndarray, z: np.ndarray, sd: float) -> AHPResult:
    nA = len(v_stated)
    A = np.ones((nA, nA))
    clipped = 0
    P = pairs(nA)
    for p, (j, l) in enumerate(P):                                  # one perturbation per unordered pair
        raw = v_stated[j] / v_stated[l] * np.exp(sd * z[p])
        a = min(max(raw, 1.0 / 9.0), 9.0)                           # Eq. 12
        clipped += int(raw < 1.0 / 9.0 or raw > 9.0)
        A[j, l], A[l, j] = a, 1.0 / a
    vals, vecs = np.linalg.eig(A)                                   # Eq. 13
    i = int(np.argmax(vals.real))
    vec = vecs[:, i].real
    v = vec / vec.sum()
    ci = max(0.0, (float(vals[i].real) - nA) / (nA - 1))            # Eq. 14
    return AHPResult(v, ci, ci / RI[nA], clipped, len(P))


def benefit_part(v):
    return v[:-1] / v[:-1].sum()


def freeze(x):
    x = np.asarray(x)
    x.flags.writeable = False
    return x


# ----------------------------------------------------------------------------- one replication (Sec. 6.1 order)
def run_replication(cfg: Config, d: RepData) -> dict:
    T, m, n = cfg.T, cfg.m, cfg.n
    W = true_weights(cfg)
    mu = np.einsum("tmn,tn->tm", d.Z, W) - cfg.rho * d.R                       # Eq. 1
    Y = mu + cfg.sd_outcome * d.eps + cfg.sd_label * d.eta                     # Eq. 2-3
    mask = obs_mask(d.keys, cfg.q)
    ks = cfg.ks()
    sched = ["static"] + ks
    kp = cfg.k_primary

    ests = {md: OutcomeWeights(md, n, cfg.L, cfg.alpha, cfg.nmin, cfg.gamma) for md in ("rolling", "expanding", "ewma", "static")}
    cur = {}
    names = (["psdf_adaptive", "psdf_additive", "psdf_fixed", "psdf_ahpw", "psdf_expanding", "psdf_ewma",
              "psdf_critic", "psdf_uniform", "ahp_static"] + [f"ahp_k{k}" for k in ks]
             + ["diag_psdf_oracle_mult", "diag_oracle_additive"])
    sel = {k: np.zeros(T, int) for k in names}
    reg = {k: np.zeros(T) for k in names}
    tau = {k: np.zeros(T) for k in names}
    top5 = {k: np.zeros(T) for k in names}
    w_hat_hist = np.zeros((T, n))
    v_hist = {s: np.zeros((T, n + 1)) for s in sched}
    rounds = []
    n_hist = np.zeros(T, int)
    win_obs, win_per, fb = np.zeros(T), np.zeros(T), np.zeros(T, bool)
    uni = np.full(n, 1.0 / n)

    def mult(Zt, w, Rt):                                                       # Eq. 4
        return (Zt @ w) / (1.0 + cfg.lam * Rt)

    for t in range(T):
        Zt, Rt = d.Z[t], d.R[t]
        n_hist[t] = ests["rolling"].n_completed()
        wts, infos = {}, {}
        for md, e in ests.items():
            wts[md], infos[md] = e.weights(t)                                   # completed periods only
        w_hat_hist[t] = wts["rolling"]
        win_obs[t], win_per[t], fb[t] = infos["rolling"]["n_obs"], infos["rolling"]["n_periods"], infos["rolling"]["fallback"] and t > 0

        vs = stated_vstar(cfg, W[t], d.gz)                                     # DM's stated preference at t
        if t == 0:
            cur["static"] = ahp_elicit(vs, d.ahp[t], cfg.sd_ahp)
            rounds.append(("static", t, "pre", cur["static"].cr, cur["static"].clipped, cur["static"].npairs))
        for k in ks:                                                           # Eq. 15, elicited online, no look-ahead
            if t % k == 0:
                cur[k] = ahp_elicit(vs, d.ahp[t], cfg.sd_ahp)
                rounds.append((f"k{k}", t, "pre" if t < cfg.reversal else "post", cur[k].cr, cur[k].clipped, cur[k].npairs))
        for s in sched:
            v_hist[s][t] = cur[s].v

        def ahp_score(v):                                                      # Eq. 11
            return Zt @ v[:-1] + v[-1] * (1.0 - Rt)

        scores = {
            "psdf_adaptive": mult(Zt, wts["rolling"], Rt),
            "psdf_additive": Zt @ wts["rolling"] - cfg.rho * Rt,               # Eq. 6
            "psdf_fixed": mult(Zt, wts["static"], Rt),                         # Sec. 4.4
            "psdf_ahpw": mult(Zt, benefit_part(cur["static"].v), Rt),          # Eq. 16-17 (initial static AHP, never updated)
            "psdf_expanding": mult(Zt, wts["expanding"], Rt),
            "psdf_ewma": mult(Zt, wts["ewma"], Rt),
            "psdf_critic": mult(Zt, critic_weights(Zt), Rt),
            "psdf_uniform": mult(Zt, uni, Rt),
            "ahp_static": ahp_score(cur["static"].v),
            "diag_psdf_oracle_mult": mult(Zt, W[t], Rt),
            "diag_oracle_additive": Zt @ W[t] - cfg.rho * Rt,
        }
        for k in ks:
            scores[f"ahp_k{k}"] = ahp_score(cur[k].v)

        mu_t = mu[t]
        best = mu_t.max()
        if cfg.rank_metrics:
            sm = np.sign(mu_t[:, None] - mu_t[None, :])
            top_mu = set(np.argsort(mu_t)[-5:].tolist())
        for name, s in scores.items():
            i = int(np.argmax(s))                                              # Eq. 5
            sel[name][t] = i
            reg[name][t] = best - mu_t[i]                                      # Eq. 19
            if cfg.rank_metrics:
                ss_ = np.sign(s[:, None] - s[None, :])
                tau[name][t] = (ss_ * sm).sum() / (m * (m - 1))
                top5[name][t] = len(top_mu & set(np.argsort(s)[-5:].tolist())) / 5.0

        mk = mask[t]                                                           # feedback enters only after the decision
        Xo, yo = np.hstack([Zt[mk], Rt[mk][:, None]]), Y[t][mk]
        for e in ests.values():
            e.update(t, Xo, yo)

    w_hist = {"psdf_adaptive": w_hat_hist}
    for s in sched:
        key = "ahp_static" if s == "static" else f"ahp_k{s}"
        w_hist[key] = v_hist[s][:, :-1] / v_hist[s][:, :-1].sum(1, keepdims=True)
    l1 = {k: np.abs(w - W).sum(1) for k, w in w_hist.items()}                  # Eq. 20
    safety = {("ahp_static" if s == "static" else f"ahp_k{s}"): v_hist[s][:, -1] for s in sched}

    out = dict(sel=sel, reg=reg, tau=tau, top5=top5, l1=l1, safety=safety, w_hist=w_hist, v_hist=v_hist,
               rounds=rounds, n_hist=n_hist, win_obs=win_obs, win_per=win_per, fallback=fb)
    for grp in ("sel", "reg", "tau", "top5", "l1", "safety", "w_hist", "v_hist"):
        out[grp] = {k: freeze(v) for k, v in out[grp].items()}                 # snapshot immutability
    return out


# ----------------------------------------------------------------------------- one experiment = R replications
def recovery_time(reg_t, cfg):
    ref = reg_t[cfg.reversal - cfg.L: cfg.reversal].mean()
    thr = cfg.recov_factor * ref + 1e-12
    for s in range(0, cfg.T - cfg.reversal - cfg.recov_run + 1):
        if np.all(reg_t[cfg.reversal + s: cfg.reversal + s + cfg.recov_run] <= thr):
            return float(s), False
    return float(cfg.T - cfg.reversal), True


def run_experiment(cfg: Config, feature_table=None) -> dict:
    reps = [run_replication(cfg, make_rep_data(cfg, s, feature_table)) for s in rep_seeds(cfg)]
    out = {"cfg": cfg}
    out["regret_t"] = {k: np.stack([r["reg"][k] for r in reps]) for k in reps[0]["reg"]}
    out["regret"] = {k: v.mean(1) for k, v in out["regret_t"].items()}         # Eq. 19 (replication mean)
    out["tau"] = {k: np.stack([r["tau"][k] for r in reps]).mean(1) for k in reps[0]["tau"]}
    out["top5"] = {k: np.stack([r["top5"][k] for r in reps]).mean(1) for k in reps[0]["top5"]}
    out["l1_t"] = {k: np.stack([r["l1"][k] for r in reps]) for k in reps[0]["l1"]}
    out["safety_t"] = {k: np.stack([r["safety"][k] for r in reps]) for k in reps[0]["safety"]}
    out["rounds"] = [(ri,) + row for ri, r in enumerate(reps) for row in r["rounds"]]
    out["win_obs"] = np.stack([r["win_obs"] for r in reps])
    out["win_per"] = np.stack([r["win_per"] for r in reps])
    out["fallback"] = np.stack([r["fallback"] for r in reps])
    return out


# ----------------------------------------------------------------------------- statistics (Sec. 9)
def boot_mean_ci(x, rng, B):
    idx = rng.integers(0, len(x), (B, len(x)))                                 # resample complete replications
    lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
    return float(lo), float(hi)


def paired(a, b, rng, B):
    """d = a - b (Eq. 21): positive => B has lower regret."""
    d = a - b
    lo, hi = boot_mean_ci(d, rng, B)
    sd = float(d.std(ddof=1))
    p = (1.0 if d.mean() == 0 else 0.0) if sd == 0 else float(stats.ttest_rel(a, b).pvalue)
    return dict(mean=float(d.mean()), sd=sd, lo=lo, hi=hi, p=p)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj, run = np.empty_like(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj
