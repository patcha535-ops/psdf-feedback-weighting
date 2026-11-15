#!/usr/bin/env python3
"""
seed_robustness.py -- repeats the baseline experiment under independent master seeds to show which conclusions are
stable and which sit at the significance boundary (used for the Monte-Carlo-stability statements in the manuscript).

    python seed_robustness.py --outdir results --n-seeds 20
"""
import argparse
from dataclasses import replace

import numpy as np
from scipy import stats

from psdf_core import Config, holm, run_experiment

FAM = [("ahp_k10", "psdf_adaptive"), ("ahp_static", "psdf_adaptive"), ("psdf_fixed", "psdf_adaptive"),
       ("psdf_additive", "psdf_adaptive"), ("ahp_k5", "psdf_adaptive"), ("ahp_k20", "psdf_adaptive"),
       ("psdf_ahpw", "psdf_adaptive")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=1001)
    a = ap.parse_args()
    rows = []
    for s in range(a.first_seed, a.first_seed + a.n_seeds):
        e = run_experiment(Config(seed=s, rank_metrics=False))
        r = e["regret"]
        ps = [stats.ttest_rel(r[x], r[y]).pvalue for x, y in FAM]
        h = holm(ps)
        d = lambda x, y: float((r[x] - r[y]).mean())
        rows.append([s, d("ahp_k10", "psdf_adaptive"), d("psdf_additive", "psdf_adaptive"), float(h[3]),
                     d("psdf_ewma", "psdf_adaptive"), float(stats.ttest_rel(r["psdf_ewma"], r["psdf_adaptive"]).pvalue),
                     r["psdf_adaptive"].mean(), r["ahp_k10"].mean()])
        print(rows[-1], flush=True)
    hd = ["seed", "d_ahp10_minus_psdf", "d_additive_minus_mult", "holm_p_additive", "d_ewma_minus_rolling", "p_ewma_unadj", "psdf_regret", "ahp10_regret"]
    with open(f"{a.outdir}/seed_robustness.csv", "w") as f:
        f.write(",".join(hd) + "\n" + "\n".join(",".join(str(x) for x in r) for r in rows) + "\n")
    R = np.array(rows)
    n = len(R)
    mac = {
        "RobSeeds": n,
        "RobAHPTenAllNeg": int((R[:, 1] < 0).sum()),
        "RobAddLower": int((R[:, 2] < 0).sum()), "RobAddSig": int((R[:, 3] < 0.05).sum()),
        "RobAddMin": f"{R[:, 2].min():.4f}".replace("-", "$-$"), "RobAddMax": f"{R[:, 2].max():.4f}".replace("-", "$-$"),
        "RobEWLower": int((R[:, 4] < 0).sum()), "RobEWSig": int(((R[:, 5] < 0.05) & (R[:, 4] < 0)).sum()),
        "RobEWMin": f"{R[:, 4].min():.4f}".replace("-", "$-$"), "RobEWMax": f"{R[:, 4].max():.4f}".replace("-", "$-$"),
    }
    with open(f"{a.outdir}/macros_robust.tex", "w") as f:
        for k, v in mac.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(mac)


if __name__ == "__main__":
    main()
