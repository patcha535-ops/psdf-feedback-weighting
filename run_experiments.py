#!/usr/bin/env python3
"""
run_experiments.py -- runs every simulation experiment of the manuscript from ONE master seed and writes
CSV files, LaTeX table bodies (tex/), LaTeX number macros (macros.tex) and a manifest, all stamped with one run_id.

    python run_experiments.py --outdir results                       # full protocol (100 replications)
    python run_experiments.py --outdir results --features prov.csv   # also the semi-synthetic regime map (Table 10)
    python run_experiments.py --outdir scratch --quick               # smoke test: NOT FOR REPORTING

prov.csv schema (one row per province, 2024 values): visitors_total, visitors_foreign, receipts_total
Features: log(visitors_total), log(receipts_total / visitors_total), visitors_foreign / visitors_total, min-max scaled.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from dataclasses import asdict, replace

import numpy as np
import scipy

from psdf_core import (Config, boot_mean_ci, holm, paired, run_experiment, semi_config)

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_FILES = ["psdf_core.py", "run_experiments.py", "seed_robustness.py", "test_temporal_integrity.py"]

FAMILY = [  # prespecified family, Sec. 9 (d = Regret_A - Regret_B)
    ("ahp_k10", "psdf_adaptive", r"Periodic AHP ($k=10$) $-$ adaptive PSDF"),
    ("ahp_static", "psdf_adaptive", r"Static AHP $-$ adaptive PSDF"),
    ("psdf_fixed", "psdf_adaptive", r"Fixed-weight PSDF $-$ adaptive PSDF"),
    ("psdf_additive", "psdf_adaptive", r"Additive PSDF $-$ multiplicative PSDF"),
    ("ahp_k5", "psdf_adaptive", r"Periodic AHP ($k=5$) $-$ adaptive PSDF"),
    ("ahp_k20", "psdf_adaptive", r"Periodic AHP ($k=20$) $-$ adaptive PSDF"),
    ("psdf_ahpw", "psdf_adaptive", r"AHP-weighted PSDF $-$ adaptive PSDF"),
]
NAMES = {
    "psdf_adaptive": "Adaptive PSDF (rolling ridge)", "psdf_additive": "Adaptive additive PSDF",
    "psdf_ewma": "Exponentially weighted ridge", "psdf_expanding": "Expanding-window ridge",
    "psdf_fixed": "Static ridge (after period 0)", "psdf_critic": "CRITIC weights", "psdf_uniform": "Uniform weights",
    "ahp_static": "Static AHP", "ahp_k10": r"Periodic AHP ($k=10$)", "ahp_k20": r"Periodic AHP ($k=20$)",
    "ahp_k5": r"Periodic AHP ($k=5$)", "psdf_ahpw": "AHP-weighted PSDF",
}


# ------------------------------------------------------------------------- formatting helpers
def sg(x, d=4):                      # signed number for LaTeX
    s = f"{abs(x):.{d}f}"
    return f"$+{s}$" if x >= 0 and round(x, d) != 0 else (f"$-{s}$" if x < 0 and round(x, d) != 0 else f"${s}$")


def nm(x, d=4):
    return f"{x:.{d}f}"


def fp(p):
    if p >= 0.001:
        return f"{p:.3f}"
    e = int(np.floor(np.log10(p))) if p > 0 else -300
    return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"


class Out:
    def __init__(self, d):
        self.d = d
        os.makedirs(f"{d}/tex", exist_ok=True)
        self.macros = {}

    def csv(self, name, headers, rows):
        with open(f"{self.d}/{name}", "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")

    def tex(self, name, spec, header, rows):
        s = f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
        s += "\n".join(" & ".join(str(c) for c in r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n"
        open(f"{self.d}/tex/{name}.tex", "w", encoding="utf-8").write(s)

    def mac(self, name, value):
        assert name.isalpha(), name
        self.macros[name] = value


def code_hash():
    h = hashlib.sha256()
    for fn in CODE_FILES:
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            h.update(open(p, "rb").read())
    return h.hexdigest()


def load_features(path):
    import csv
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    v = np.array([float(r["visitors_total"]) for r in rows])
    f = np.array([float(r["visitors_foreign"]) for r in rows])
    rc = np.array([float(r["receipts_total"]) for r in rows])
    F = np.column_stack([np.log(v), np.log(rc / v), f / v])
    return (F - F.min(0)) / (F.max(0) - F.min(0))


# ------------------------------------------------------------------------- regime map (Tables 9-10)
def regime_map(cfg, rng, B, feature_table, tag):
    cells, rows = [], []
    for sd, rev, g in itertools.product([0.05, 0.20], [30, 33, 37], [0.0, 0.5, 1.0, 1.5]):
        c = replace(cfg, sd_outcome=sd, sd_label=sd, reversal=rev, gap=g, rank_metrics=False)
        e = run_experiment(c, feature_table)
        r = paired(e["regret"]["ahp_k10"], e["regret"]["psdf_adaptive"], rng, B)
        r.update(sd=sd, rev=rev, g=g, exp=e)
        cells.append(r)
    for r, h in zip(cells, holm([c["p"] for c in cells])):          # Holm across the 24 cells of this environment
        r["holm"] = float(h)
    return cells


def regime_table(out, cells, name):
    rows = []
    for sd, rev in itertools.product([0.05, 0.20], [30, 33, 37]):
        row = [f"{sd:.2f}", f"$t={rev}$"]
        for g in [0.0, 0.5, 1.0, 1.5]:
            c = next(x for x in cells if x["sd"] == sd and x["rev"] == rev and x["g"] == g)
            row.append(sg(c["mean"]) + ("$^{*}$" if c["holm"] < 0.05 else ""))
        rows.append(row)
    out.tex(name, "llrrrr", r"Noise SD & Reversal & $g=0$ & $g=0.5$ & $g=1.0$ & $g=1.5$", rows)
    out.csv(f"{name}.csv", ["noise_sd", "reversal", "gap", "d_ahp10_minus_psdf", "ci_lo", "ci_hi", "p", "holm_p"],
            [[c["sd"], c["rev"], c["g"], c["mean"], c["lo"], c["hi"], c["p"], c["holm"]] for c in cells])


def regime_counts(out, cells, pre):
    for g, tag in [(0.0, "Zero"), (0.5, "Half"), (1.0, "One"), (1.5, "OneHalf")]:
        cs = [c for c in cells if c["g"] == g]
        out.mac(f"{pre}PosPoint{tag}", sum(c["mean"] > 0 for c in cs))
        out.mac(f"{pre}PosSig{tag}", sum(c["mean"] > 0 and c["holm"] < 0.05 for c in cs))
        out.mac(f"{pre}NegSig{tag}", sum(c["mean"] < 0 and c["holm"] < 0.05 for c in cs))
        out.mac(f"{pre}NegPoint{tag}", sum(c["mean"] < 0 for c in cs))
    for c in cells:                                                  # every cell as a macro (letters only in names)
        nz = "Low" if c["sd"] == 0.05 else "High"
        out.mac(f"{pre}Cell{nz}Rev{ {30: 'A', 33: 'B', 37: 'C'}[c['rev']] }Gap{ {0.0: 'A', 0.5: 'B', 1.0: 'C', 1.5: 'D'}[c['g']] }",
                sg(c["mean"]) + ("$^{*}$" if c["holm"] < 0.05 else ""))
    ns = [c for c in cells if c["g"] == 0.0 and c["holm"] >= 0.05]
    out.mac(f"{pre}ZeroNSCells", "; ".join(f"t={c['rev']}, SD={c['sd']:.2f} ({c['mean']:+.4f})" for c in ns) or "none")


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--boot-factorial", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=6621)
    ap.add_argument("--features", default=None, help="CSV of 2024 provincial features for the semi-synthetic map")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.boot, a.boot_factorial = 20, 1000, 500
    t0 = time.time()
    cfg = Config(R=a.reps, seed=a.seed, n_boot=a.boot)
    conformant = a.reps == 100 and a.boot >= 10000 and a.boot_factorial >= 5000 and not a.quick
    cj = json.dumps(asdict(cfg), sort_keys=True)
    run_id = hashlib.sha256((code_hash() + cj + f"{a.boot_factorial}").encode()).hexdigest()[:12]
    stamp = f"run_id {run_id}; protocol_conformant={conformant}"
    O = Out(a.outdir)
    O.mac("RunID", run_id)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 99]))
    B = cfg.n_boot

    print(f"[{run_id}] baseline", flush=True)
    base = run_experiment(cfg)
    reg = base["regret"]
    rm = lambda k: reg[k]

    # ---- Table 3
    rows, csvr = [], []
    for k in ["psdf_adaptive", "ahp_k10", "ahp_static"]:
        lo, hi = boot_mean_ci(reg[k], rng, B)
        rows.append([NAMES[k], nm(reg[k].mean()), nm(lo), nm(hi), nm(reg[k].std(ddof=1))])
        csvr.append([k, reg[k].mean(), lo, hi, reg[k].std(ddof=1)])
        tag = {"psdf_adaptive": "Adaptive", "ahp_k10": "AHPTen", "ahp_static": "AHPStatic"}[k]
        for nme, val in [("Reg", reg[k].mean()), ("RegLo", lo), ("RegHi", hi), ("RegSD", reg[k].std(ddof=1))]:
            O.mac(f"{nme}{tag}", nm(val))
    O.tex("table3", "lrrrr", r"Method & Mean regret & 95\% CI lower & 95\% CI upper & Replication SD", rows)
    O.csv("table3_primary.csv", ["method", "mean", "ci_lo", "ci_hi", "rep_sd"], csvr)

    # phases
    x = base["regret_t"]["psdf_adaptive"]; y = base["regret_t"]["ahp_k10"]
    for nme, arr in [("Adaptive", x), ("AHPTen", y)]:
        O.mac(f"Pre{nme}", nm(arr[:, :30].mean())); O.mac(f"Post{nme}", nm(arr[:, 30:].mean()))
    O.mac("TransAdaptive", nm(x[:, 30:45].mean())); O.mac("LateAdaptive", nm(x[:, 45:60].mean()))

    # ---- Table 4 (seven comparisons, Holm over the family)
    fam = []
    for A_, B_, text in FAMILY:
        r = paired(reg[A_], reg[B_], rng, B); r.update(A=A_, B=B_, text=text); fam.append(r)
    for r, h in zip(fam, holm([r["p"] for r in fam])):
        r["holm"] = float(h)
    O.tex("table4", "lrcrr", r"Comparison & Mean diff. & 95\% CI & Paired $p$ & Holm $p$",
          [[r["text"], sg(r["mean"]), f"[{r['lo']:.4f}, {r['hi']:.4f}]".replace("-", "$-$"), fp(r["p"]), fp(r["holm"])] for r in fam])
    O.csv("table4_paired.csv", ["comparison", "mean_diff", "ci_lo", "ci_hi", "paired_p", "holm_p"],
          [[r["A"] + "-" + r["B"], r["mean"], r["lo"], r["hi"], r["p"], r["holm"]] for r in fam])
    O.mac("DiffAHPTen", sg(fam[0]["mean"])); O.mac("DiffAHPTenLo", f"{fam[0]['lo']:.4f}".replace("-", "$-$"))
    O.mac("DiffAHPTenHi", f"{fam[0]['hi']:.4f}".replace("-", "$-$"))
    O.mac("AddHolm", fp(fam[3]["holm"])); O.mac("AddDiff", sg(fam[3]["mean"]))
    O.mac("AddSig", "yes" if fam[3]["holm"] < 0.05 else "no")

    # ---- Table 5 weight recovery
    rows, csvr = [], []
    for k, lab in [("psdf_adaptive", "Adaptive PSDF"), ("ahp_k10", r"Periodic AHP ($k=10$)"), ("ahp_k5", r"Periodic AHP ($k=5$)"),
                   ("ahp_k20", r"Periodic AHP ($k=20$)"), ("ahp_static", "Static AHP")]:
        l1 = base["l1_t"][k]
        pre, post, ov = l1[:, :30].mean(), l1[:, 30:].mean(), l1.mean()
        rows.append([lab, f"{pre:.3f}", f"{post:.3f}", f"{ov:.3f}"]); csvr.append([k, pre, post, ov])
    # fixed-weight PSDF weights are not stored by the engine's L1 dict; recompute from a dedicated pass below
    O.csv("table5_weight_recovery.csv", ["method", "pre", "post", "overall"], csvr)
    O.mac("LOneZero", f"{base['l1_t']['psdf_adaptive'][:, 0].mean():.1f}")

    # ---- Table 6 ablation
    rows, csvr = [], []
    for k, lab in [("psdf_adaptive", "Adaptive weights, multiplicative risk"), ("psdf_fixed", "Fixed weights, multiplicative risk"),
                   ("psdf_additive", "Adaptive weights, additive risk"), ("psdf_ahpw", "Initial AHP weights, multiplicative PSDF")]:
        lo, hi = boot_mean_ci(reg[k], rng, B)
        d = "--" if k == "psdf_adaptive" else sg((reg[k] - reg["psdf_adaptive"]).mean())
        rows.append([lab, nm(reg[k].mean()), f"[{lo:.4f}, {hi:.4f}]", d]); csvr.append([k, reg[k].mean(), lo, hi, d])
    O.tex("table6", "lccr", r"Configuration & Mean regret & 95\% CI & Paired difference", rows)
    O.csv("table6_ablation.csv", ["config", "mean", "ci_lo", "ci_hi", "paired_diff"], csvr)
    O.mac("OracleMultRegret", nm(reg["diag_psdf_oracle_mult"].mean())); O.mac("OracleAddRegret", nm(reg["diag_oracle_additive"].mean()))

    # ---- Table 9 AHP diagnostics (macros) and secondary
    rr = [r for r in base["rounds"] if r[1] == "k10"]
    cr = np.array([r[4] for r in rr]); cl = np.array([r[5] for r in rr]); npair = rr[0][6]
    q1, q3 = np.percentile(cr, [25, 75])
    O.mac("CRMatrices", len(rr)); O.mac("CRMean", f"{cr.mean():.4f}"); O.mac("CRSD", f"{cr.std(ddof=1):.4f}")
    O.mac("CRMedian", f"{np.median(cr):.4f}"); O.mac("CRQOne", f"{q1:.4f}"); O.mac("CRQThree", f"{q3:.4f}")
    O.mac("CRProp", f"{(cr <= 0.10).mean():.3f}"); O.mac("ClipN", f"{int(cl.sum()):,}".replace(",", "{,}"))
    O.mac("ClipTotal", f"{npair * len(rr):,}".replace(",", "{,}")); O.mac("ClipPct", f"{100 * cl.sum() / (npair * len(rr)):.1f}")
    rows = []
    for sched in ["static", "k5", "k10", "k20"]:
        r_ = [r for r in base["rounds"] if r[1] == sched]
        c_, k_ = np.array([r[4] for r in r_]), np.array([r[5] for r in r_])
        rows.append([sched, len(r_), c_.mean(), c_.std(ddof=1), np.median(c_), (c_ <= .1).mean(), int(k_.sum()), npair * len(r_)])
    O.csv("table_ahp_diagnostics.csv", ["schedule", "matrices", "cr_mean", "cr_sd", "cr_median", "prop_cr_le_0.10", "clipped", "judgments"], rows)

    # ---- Table 7 partial feedback
    print("partial feedback", flush=True)
    rows, csvr = [], []
    for q, lab in [(0.25, "25\\% (8 of 30; achieved 26.7\\%)"), (0.50, "50\\% (15 of 30; achieved 50.0\\%)"),
                   (0.75, "75\\% (22 of 30; achieved 73.3\\%)"), (1.00, "100\\% (30 of 30; achieved 100\\%)")]:
        ex = base if q == 1.0 else run_experiment(replace(cfg, q=q))
        xx = ex["regret"]["psdf_adaptive"]; lo, hi = boot_mean_ci(xx, rng, B)
        rows.append([lab, nm(xx.mean()), nm(lo), nm(hi)]); csvr.append([q, xx.mean(), lo, hi, ex["fallback"][:, 1:].mean()])
    O.tex("table7", "lrrr", r"Target rate (observed per period) & Mean expected regret & 95\% CI lower & 95\% CI upper", rows)
    O.csv("table7_partial_feedback.csv", ["q", "mean", "ci_lo", "ci_hi", "fallback_rate"], csvr)

    # ---- Table 8 noise sensitivity
    print("noise sensitivity", flush=True)
    rows, csvr = [], []
    for s in [0.0, 0.05, 0.10, 0.20]:
        ex = base if s == 0.05 else run_experiment(replace(cfg, sd_outcome=s, sd_label=s, rank_metrics=False))
        p_, a_ = ex["regret"]["psdf_adaptive"], ex["regret"]["ahp_k10"]
        r = paired(a_, p_, rng, B)
        rows.append([f"{s:.2f}", nm(p_.mean()), nm(a_.mean()), sg(r["mean"]), f"[{r['lo']:.4f}, {r['hi']:.4f}]".replace("-", "$-$")])
        csvr.append([s, p_.mean(), a_.mean(), r["mean"], r["lo"], r["hi"]])
        if s in (0.0, 0.20):
            O.mac("NoisePSDF" + ("Zero" if s == 0 else "High"), nm(p_.mean()))
    O.tex("table8", "lrrrc", r"Noise SD & Adaptive PSDF & Periodic AHP ($k=10$) & Difference (AHP $-$ PSDF) & 95\% CI", rows)
    O.csv("table8_noise.csv", ["noise_sd", "psdf", "ahp10", "diff", "ci_lo", "ci_hi"], csvr)

    # ---- factorial regime map, synthetic (Tables 9, 11)
    print("factorial (synthetic)", flush=True)
    cells = regime_map(cfg, rng, a.boot_factorial, None, "syn")
    regime_table(O, cells, "table9"); regime_counts(O, cells, "Syn")
    rep = [(30, 0.0, 0.05), (37, 0.0, 0.05), (30, 1.0, 0.05), (37, 1.0, 0.20)]
    cols = [next(c for c in cells if c["rev"] == r_ and c["g"] == g_ and c["sd"] == s_)["exp"]["regret"] for r_, g_, s_ in rep]
    order = ["psdf_adaptive", "psdf_additive", "psdf_ewma", "psdf_expanding", "psdf_fixed", "psdf_critic", "psdf_uniform",
             "ahp_static", "ahp_k10", "ahp_k20", "psdf_ahpw"]
    O.tex("table11", "lrrrr", r"Method & $t=30$, $g=0$ & $t=37$, $g=0$ & $t=30$, $g=1$ & $t=37$, $g=1$, SD 0.20",
          [[NAMES[k]] + [nm(c[k].mean()) for c in cols] for k in order])
    O.csv("table11_all_methods.csv", ["method", "t30_g0", "t37_g0", "t30_g1", "t37_g1_sd20"], [[k] + [c[k].mean() for c in cols] for k in order])
    ew = paired(base["regret"]["psdf_ewma"], base["regret"]["psdf_adaptive"], rng, B)
    O.mac("EWDiff", sg(ew["mean"])); O.mac("EWLo", f"{ew['lo']:.4f}".replace("-", "$-$")); O.mac("EWHi", f"{ew['hi']:.4f}".replace("-", "$-$"))
    base_cell = next(c for c in cells if c["rev"] == 30 and c["g"] == 0.0 and c["sd"] == 0.05)
    O.mac("AHPShrinkLate", f"{abs(next(c for c in cells if c['rev'] == 37 and c['g'] == 0.0 and c['sd'] == 0.05)['mean']):.4f}")
    O.mac("AHPAdvEarly", f"{abs(base_cell['mean']):.4f}")

    # ---- fixed-weight L1 (Table 5 last row): dedicated pass
    from psdf_core import OutcomeWeights, make_rep_data, rep_seeds, true_weights, obs_mask
    W = true_weights(cfg); L1fix = []
    for ss in rep_seeds(cfg):
        d = make_rep_data(cfg, ss)
        mu = np.einsum("tmn,tn->tm", d.Z, W) - cfg.rho * d.R
        Yv = mu + cfg.sd_outcome * d.eps + cfg.sd_label * d.eta
        e = OutcomeWeights("static", cfg.n, cfg.L, cfg.alpha, cfg.nmin, cfg.gamma); ws = []
        for t in range(cfg.T):
            w, _ = e.weights(t); ws.append(np.abs(w - W[t]).sum())
            e.update(t, np.hstack([d.Z[t], d.R[t][:, None]]), Yv[t])
        L1fix.append(ws)
    L1fix = np.array(L1fix)
    rows = []
    for k, lab in [("psdf_adaptive", "Adaptive PSDF"), ("ahp_k10", r"Periodic AHP ($k=10$)"), ("ahp_k5", r"Periodic AHP ($k=5$)"),
                   ("ahp_k20", r"Periodic AHP ($k=20$)"), ("ahp_static", "Static AHP")]:
        l1 = base["l1_t"][k]; rows.append([lab, f"{l1[:, :30].mean():.3f}", f"{l1[:, 30:].mean():.3f}", f"{l1.mean():.3f}"])
    rows.append(["Fixed-weight PSDF", f"{L1fix[:, :30].mean():.3f}", f"{L1fix[:, 30:].mean():.3f}", f"{L1fix.mean():.3f}"])
    O.tex("table5", "lrrr", r"Method & Pre-reversal & Post-reversal & Overall", rows)

    # ---- semi-synthetic regime map (Table 10) only if provincial features are supplied
    if a.features:
        print("factorial (semi-synthetic)", flush=True)
        F = load_features(a.features)
        sc = semi_config(cfg)
        scells = regime_map(sc, rng, a.boot_factorial, F, "semi")
        regime_table(O, scells, "table10"); regime_counts(O, scells, "Semi")

    # ---- write macros, manifest
    with open(f"{a.outdir}/macros.tex", "w", encoding="utf-8") as f:
        f.write(f"% generated by run_experiments.py -- {stamp}\n")
        for k, v in O.macros.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    man = dict(run_id=run_id, protocol_conformant=conformant, config=asdict(cfg), boot=a.boot, boot_factorial=a.boot_factorial,
               seed=a.seed, seed_scheme="SeedSequence(seed).spawn(R); per replication 5 streams: features, outcome/label noise, "
               "AHP judgment noise, feedback-mask keys, preference-gap draw", semi_synthetic_run=bool(a.features),
               code_sha256=code_hash(), python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
               platform=platform.platform(), runtime_seconds=round(time.time() - t0, 1),
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(man, open(f"{a.outdir}/manifest.json", "w"), indent=2)
    print(f"done in {man['runtime_seconds']}s -> {a.outdir} ({stamp})")


if __name__ == "__main__":
    main()
