#!/usr/bin/env python3
"""
empirical_tourism.py -- empirical application of Section 11 (monthly provincial tourism data).

Input: a tidy panel CSV (one row per province-month) with columns
    province, month (YYYY-MM), occupancy, guests, visitors_total, visitors_foreign, receipts_total
covering a balanced 77-province x 45-month panel (2023-01 .. 2026-09).  Building that CSV from the MOTS workbooks is the
job of build_panel.py (to be added once the workbooks are available).

    python empirical_tourism.py panel.csv --outdir results_empirical

Design (Sec. 11.2, fixed in advance): decision month t = 2024-02 .. 2026-09 (32 months), 77 provinces as alternatives,
features measured in month t-1 (percentile-ranked within month), risk = percentile rank of the coefficient of variation of
monthly visitors over the 12 months ending in t-1, outcome = within-month z-score (sample SD) of the year-on-year log growth of receipts.
Training pairs for the decision at t are (X_{s-1}, Y_s), s <= t-1.  PSDF variants use the multiplicative score with
lambda = 1, alpha = 0.1, N_min = 12; rolling window L = 12 months; EW discount 0.9.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from scipy import stats

from psdf_core import OutcomeWeights, critic_weights, weighted_ridge_slopes

LAM, ALPHA, NMIN, L, GAMMA, TOP, BLOCK, BOOT, SEED = 1.0, 0.1, 12, 12, 0.9, 10, 3, 10000, 6621
FEATS = ["occupancy", "log_visitors", "foreign_share", "log_receipts_per_visitor", "guests_per_visitor"]


def load_panel(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    provs = list(dict.fromkeys(r["province"] for r in rows))
    months = sorted({r["month"] for r in rows})
    P, M = len(provs), len(months)
    pi, mi = {p: i for i, p in enumerate(provs)}, {m: j for j, m in enumerate(months)}
    A = {k: np.full((P, M), np.nan) for k in ["occupancy", "guests", "visitors_total", "visitors_foreign", "receipts_total"]}
    for r in rows:
        for k in A:
            A[k][pi[r["province"]], mi[r["month"]]] = float(r[k])
    assert not any(np.isnan(a).any() for a in A.values()), "panel must be balanced with no missing values"
    return provs, months, A


def pct_rank(x):
    return (stats.rankdata(x) - 1.0) / (len(x) - 1.0)                        # average ranks, scaled to [0,1]


def month_features(A, j):
    """Benefit features and risk measured in month index j (information available at the end of month j)."""
    vis, frn, rec = A["visitors_total"][:, j], A["visitors_foreign"][:, j], A["receipts_total"][:, j]
    raw = [A["occupancy"][:, j], np.log(vis), frn / vis, np.log(rec / vis), A["guests"][:, j] / vis]
    Z = np.column_stack([pct_rank(x) for x in raw])
    win = A["visitors_total"][:, j - 11: j + 1]                                # twelve months ending in j
    cv = win.std(axis=1, ddof=1) / win.mean(axis=1)
    return Z, pct_rank(cv)


def outcome(A, s):
    g = np.log(A["receipts_total"][:, s] / A["receipts_total"][:, s - 12])      # year-on-year log growth
    return (g - g.mean()) / g.std(ddof=1)                                       # within-month z-score (sample SD)


def mbb_ci(d, rng, b=BLOCK, B=BOOT):
    n = len(d)
    k = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=(B, k))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(B, -1)[:, :n]
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return float(lo), float(hi)


def run(A, n_months=None):
    P, M = A["visitors_total"].shape
    first_y = 12                                                               # first month with a year-on-year outcome (2024-01)
    ests = {m: OutcomeWeights(m, 5, L, ALPHA, NMIN, GAMMA) for m in ("rolling", "expanding", "ewma", "static")}
    signed_hist = []                                                           # (period, X, y) for the signed ridge forecast
    methods = ["psdf", "psdf_expanding", "psdf_ewma", "psdf_static", "uniform", "critic", "signed_ridge", "momentum"]
    reg10 = {m: [] for m in methods}; reg5 = {m: [] for m in methods}; sp = {m: [] for m in methods}
    best10, best5, learned, fallback, sel_log = [], [], [], [], []
    Z0, R0 = month_features(A, first_y - 1)                                    # pair (X_{2023-12}, Y_{2024-01}) = period d = 0
    X0, y0 = np.column_stack([Z0, R0]), outcome(A, first_y)
    for e in ests.values():
        e.update(0, X0, y0)
    signed_hist.append((0, X0, y0))
    last = M - 1 if n_months is None else n_months - 1
    for t in range(first_y + 1, last + 1):                                     # decision months 2024-02 .. 2026-09
        d = t - first_y                                                        # estimator period index of month t
        Xprev, Rprev = month_features(A, t - 1)                                # features measured in t-1
        Zt = Xprev
        y_t = outcome(A, t)                                                    # outcome realised in month t (used only for evaluation)
        wts = {}
        for name, e in ests.items():
            wts[name], info = e.weights(d)
            if name == "rolling":
                learned.append(wts[name]); fallback.append(info["fallback"])
        mult = lambda w: (Zt @ w) / (1.0 + LAM * Rprev)
        scores = {"psdf": mult(wts["rolling"]), "psdf_expanding": mult(wts["expanding"]), "psdf_ewma": mult(wts["ewma"]),
                  "psdf_static": mult(wts["static"]), "uniform": mult(np.full(5, 0.2)), "critic": mult(critic_weights(Zt))}
        win = [h for h in signed_hist if h[0] >= max(0, d - L)]                # signed rolling ridge forecast
        if sum(len(h[2]) for h in win) >= NMIN:
            Xw, yw = np.vstack([h[1] for h in win]), np.concatenate([h[2] for h in win])
            beta = weighted_ridge_slopes(Xw, yw, np.ones(len(yw)), ALPHA)
            scores["signed_ridge"] = yw.mean() + (np.column_stack([Zt, Rprev]) - Xw.mean(0)) @ beta
        else:
            scores["signed_ridge"] = np.zeros(P)
        scores["momentum"] = outcome(A, t - 1)                                  # previous month's outcome
        order_best = np.argsort(-y_t, kind="stable")
        best10.append(y_t[order_best[:TOP]].mean()); best5.append(y_t[order_best[:5]].mean())
        for m, s in scores.items():
            order = np.argsort(-s, kind="stable")
            reg10[m].append(y_t[order_best[:TOP]].mean() - y_t[order[:TOP]].mean())
            reg5[m].append(y_t[order_best[:5]].mean() - y_t[order[:5]].mean())
            sp[m].append(stats.spearmanr(s, y_t).statistic)
        sel_log.append(np.argsort(-scores["psdf"], kind="stable")[:TOP])
        Xo = np.column_stack([Zt, Rprev])                                       # pair (X_{t-1}, Y_t) enters only AFTER the decision
        for e in ests.values():
            e.update(d, Xo, y_t)
        signed_hist.append((d, Xo, y_t))
    return dict(reg10={k: np.array(v) for k, v in reg10.items()}, reg5={k: np.array(v) for k, v in reg5.items()},
                sp={k: np.array(v) for k, v in sp.items()}, best10=np.array(best10), best5=np.array(best5),
                learned=np.array(learned), fallback=np.array(fallback), sel=np.array(sel_log))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--outdir", default="results_empirical")
    ap.add_argument("--boot", type=int, default=BOOT)
    a = ap.parse_args()
    os.makedirs(f"{a.outdir}/tex", exist_ok=True)
    provs, months, A = load_panel(a.panel)
    assert len(provs) == 77 and len(months) == 45, (len(provs), len(months))
    R = run(A)
    n = len(R["best10"])
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 12]))
    names = [("psdf", "PSDF (rolling ridge, $L=12$)"), ("psdf_expanding", "PSDF, expanding window"),
             ("psdf_ewma", "PSDF, exponentially weighted"), ("psdf_static", "PSDF, static weights"),
             ("uniform", "Uniform weights"), ("critic", "CRITIC weights"), ("signed_ridge", "Signed rolling ridge forecast"),
             ("momentum", "Momentum (previous month)")]
    sgn = lambda x, d=3: (f"$+{x:.{d}f}$" if x >= 0 else f"$-{abs(x):.{d}f}$")
    rows, csvr = [], []
    for k, lab in names:
        r10, r5, rho = R["reg10"][k], R["reg5"][k], R["sp"][k].mean()
        if k == "psdf":
            diff = "--"; dd = None
        else:
            d = R["reg10"][k] - R["reg10"]["psdf"]; lo, hi = mbb_ci(d, rng, B=a.boot)
            diff = f"{sgn(d.mean())} [{sgn(lo)}, {sgn(hi)}]"; dd = (d.mean(), lo, hi)
        rows.append([lab, f"{r10.mean():.3f}", f"{r5.mean():.3f}", sgn(rho), diff])
        csvr.append([k, r10.mean(), r5.mean(), rho] + (list(dd) if dd else ["", "", ""]))
    rnd10, rnd5 = R["best10"].mean(), R["best5"].mean()                          # random selection: expected selected outcome is 0
    d = R["best10"] - R["reg10"]["psdf"]                                       # random minus PSDF regret, month by month
    lo, hi = mbb_ci(d, rng, B=a.boot)
    rows.append(["Random selection (expected)", f"{rnd10:.3f}", f"{rnd5:.3f}", "0", f"{sgn(d.mean())} [{sgn(lo)}, {sgn(hi)}]"])
    csvr.append(["random", rnd10, rnd5, 0, d.mean(), lo, hi])
    body = ("\\begin{tabular}{lrrrl}\n\\toprule\nMethod & Top-10 regret & Top-5 regret & Spearman & Diff.\\ vs PSDF (top 10) \\\\\n\\midrule\n"
            + "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(f"{a.outdir}/tex/table12.tex", "w", encoding="utf-8").write(body)
    with open(f"{a.outdir}/table12.csv", "w") as f:
        f.write("method,top10_regret,top5_regret,spearman,diff_vs_psdf,ci_lo,ci_hi\n" + "\n".join(",".join(str(x) for x in r) for r in csvr) + "\n")
    half = n // 2
    summ = dict(decision_months=n, best10_mean=rnd10, best5_mean=rnd5, psdf_top5_minus_random=float(R["reg5"]["psdf"].mean() - rnd5),
                psdf_regret_first_half=float(R["reg10"]["psdf"][:half].mean()), psdf_regret_second_half=float(R["reg10"]["psdf"][half:].mean()),
                learned_weight_mean=dict(zip(FEATS, map(float, R["learned"].mean(0)))), fallback_rate=float(R["fallback"].mean()))
    d5 = R["reg5"]["psdf"] - R["best5"]                                          # PSDF minus random, top 5
    d10 = R["reg10"]["psdf"] - R["best10"]
    lo5, hi5 = mbb_ci(d5, rng, B=a.boot); lo10, hi10 = mbb_ci(d10, rng, B=a.boot)
    summ.update(psdf_minus_random_top10=[float(d10.mean()), lo10, hi10], psdf_minus_random_top5=[float(d5.mean()), lo5, hi5])
    json.dump(summ, open(f"{a.outdir}/summary.json", "w"), indent=2)
    sg2 = lambda x: f"${'+' if x >= 0 else '-'}{abs(x):.3f}$"
    W = summ["learned_weight_mean"]
    mac = {"EmpMonths": n, "EmpBestTen": f"{rnd10:.3f}", "EmpBestFive": f"{rnd5:.3f}",
           "EmpTenDiff": sg2(d10.mean()), "EmpTenLo": sg2(lo10), "EmpTenHi": sg2(hi10),
           "EmpFiveDiff": sg2(d5.mean()), "EmpFiveLo": sg2(lo5), "EmpFiveHi": sg2(hi5),
           "EmpSpearman": sg2(R["sp"]["psdf"].mean()), "EmpFirstHalf": f"{summ['psdf_regret_first_half']:.3f}",
           "EmpSecondHalf": f"{summ['psdf_regret_second_half']:.3f}", "EmpWOcc": f"{W['occupancy']:.2f}", "EmpWForeign": f"{W['foreign_share']:.2f}",
           "EmpWGuests": f"{W['guests_per_visitor']:.2f}", "EmpWReceipts": f"{W['log_receipts_per_visitor']:.2f}", "EmpWVisitors": f"{W['log_visitors']:.2f}"}
    with open(f"{a.outdir}/macros_empirical.tex", "w") as f:
        f.write("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
