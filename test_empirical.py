"""Tests for empirical_tourism.py on a SYNTHETIC panel (shape of the real one: 77 provinces x 45 months). Not a data result."""
import csv
import os
import sys
import tempfile

import numpy as np

import empirical_tourism as E


def synthetic_panel(seed=0, P=77, M=45):
    rng = np.random.default_rng(seed)
    base = np.exp(rng.normal(14, 1.0, P))
    season = 1 + 0.3 * np.sin(2 * np.pi * np.arange(M) / 12)
    vis = base[:, None] * season[None, :] * np.exp(rng.normal(0, 0.1, (P, M)))
    frn = vis * rng.uniform(0.01, 0.6, (P, 1)) * np.exp(rng.normal(0, 0.05, (P, M))).clip(0, 1.5)
    frn = np.minimum(frn, 0.95 * vis)
    rec = vis * np.exp(rng.normal(8.0, 0.4, (P, 1))) * np.exp(rng.normal(0, 0.1, (P, M))) / 1e6
    occ = rng.uniform(30, 85, (P, M))
    guests = vis * rng.uniform(0.2, 0.7, (P, 1))
    return dict(occupancy=occ, guests=guests, visitors_total=vis, visitors_foreign=frn, receipts_total=rec)


def test_shapes_and_decision_count():
    R = E.run(synthetic_panel())
    assert len(R["best10"]) == 32 and R["sel"].shape == (32, 10)
    assert set(R["reg10"]) >= {"psdf", "momentum", "signed_ridge", "critic", "uniform"}


def test_no_future_information_in_decisions():
    A = synthetic_panel(1)
    base = E.run(A)
    t0 = 30                                                       # month index; decisions at t <= t0 must not change
    B = {k: v.copy() for k, v in A.items()}
    rng = np.random.default_rng(5)
    for k in B:
        B[k][:, t0:] = B[k][:, t0:] * np.exp(rng.normal(0, 1.0, B[k][:, t0:].shape))
    B["visitors_foreign"] = np.minimum(B["visitors_foreign"], 0.95 * B["visitors_total"])
    new = E.run(B)
    k = t0 - 13 + 1                                               # decisions for t = 13 .. t0
    assert np.array_equal(base["sel"][:k], new["sel"][:k])
    assert not np.array_equal(base["sel"][k:], new["sel"][k:])    # the test has power


def test_outcome_is_within_month_zscore():
    A = synthetic_panel(2)
    y = E.outcome(A, 20)
    assert abs(y.mean()) < 1e-12 and abs(y.std(ddof=1) - 1) < 1e-12      # so random selection has expected outcome 0 and regret = best-ten mean


def test_end_to_end_tex_output():
    A = synthetic_panel(3)
    months = [f"{2023 + i // 12}-{i % 12 + 1:02d}" for i in range(45)]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "panel.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["province", "month", "occupancy", "guests", "visitors_total", "visitors_foreign", "receipts_total"])
            for p in range(77):
                for j, m in enumerate(months):
                    w.writerow([f"P{p}", m] + [A[k][p, j] for k in ["occupancy", "guests", "visitors_total", "visitors_foreign", "receipts_total"]])
        sys.argv = ["x", path, "--outdir", os.path.join(d, "o"), "--boot", "200"]
        E.main()
        t = open(os.path.join(d, "o", "tex", "table12.tex")).read()
        assert t.count("\\\\") >= 10 and "Random selection" in t


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as ex:  # noqa
                fails += 1; print("FAIL", name, repr(ex))
    sys.exit(1 if fails else 0)
