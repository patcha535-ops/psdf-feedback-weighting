"""Temporal-integrity tests (Sec. 6.2) and sanity checks. No external test framework needed:
    python test_temporal_integrity.py
"""
import dataclasses
import sys
import numpy as np
from psdf_core import (Config, OutcomeWeights, ahp_elicit, holm, make_rep_data, rep_seeds, run_replication,
                       stated_vstar, obs_mask)

CFG = Config(R=1, T=40)


def _data(cfg=CFG):
    return make_rep_data(cfg, rep_seeds(cfg)[0])


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def _same_through(a, b, t0):
    return all(np.array_equal(a["sel"][k][: t0 + 1], b["sel"][k][: t0 + 1]) for k in a["sel"])


def test_1_no_future_label_access():
    d, t0 = _data(), 20
    base = run_replication(CFG, d)
    rng = np.random.default_rng(1)
    d2 = dataclasses.replace(d, eps=d.eps.copy(), eta=d.eta.copy())
    d2.eps[t0:] = 20 * rng.standard_normal(d.eps[t0:].shape)
    d2.eta[t0:] = 20 * rng.standard_normal(d.eta[t0:].shape)
    new = run_replication(CFG, d2)
    assert _same_through(base, new, t0)
    for k in ("psdf_adaptive", "psdf_ewma", "psdf_expanding"):                       # the test has power
        assert not np.array_equal(base["sel"][k][t0 + 1:], new["sel"][k][t0 + 1:]), k


def test_2_no_future_judgment_access():
    d, t0 = _data(), 22
    base = run_replication(CFG, d)
    d2 = dataclasses.replace(d, ahp=d.ahp.copy())
    d2.ahp[t0 + 1:] *= 30
    new = run_replication(CFG, d2)
    assert _same_through(base, new, t0)
    assert not np.array_equal(base["sel"]["ahp_k5"][t0 + 1:], new["sel"]["ahp_k5"][t0 + 1:])


def test_3_weight_freezing():
    r = run_replication(CFG, _data())
    for s, v in r["v_hist"].items():
        for t in range(1, CFG.T):
            if s == "static" or t % s != 0:
                assert np.array_equal(v[t], v[t - 1]), (s, t)


def test_4_sequential_updates():
    e = OutcomeWeights("rolling", 5, 15, 0.1, 12, 0.9)
    X, y = np.random.rand(30, 6), np.random.rand(30)
    e.update(0, X, y)
    assert _raises(lambda: e.update(2, X, y))
    assert _raises(lambda: e.update(0, X, y))
    assert _raises(lambda: e.weights(5))


def test_5_snapshot_immutability():
    r = run_replication(CFG, _data())
    assert _raises(lambda: r["sel"]["psdf_adaptive"].__setitem__(0, 0))
    assert _raises(lambda: r["w_hist"]["psdf_adaptive"].__setitem__((0, 0), 1.0))


def test_6_feedback_timing():
    r = run_replication(CFG, _data())
    assert np.array_equal(r["n_hist"], np.arange(CFG.T))                              # decision t sees exactly periods < t


def test_7_reproducibility():
    a, b = run_replication(CFG, _data()), run_replication(CFG, _data())
    assert all(np.array_equal(a["reg"][k], b["reg"][k]) for k in a["reg"])


def test_min_observations_fallback():
    e = OutcomeWeights("rolling", 5, 15, 0.1, 12, 0.9)
    e.update(0, np.random.rand(8, 6), np.random.rand(8))
    w, info = e.weights(1)
    assert info["fallback"] and np.allclose(w, 0.2)                                   # 8 < Nmin = 12 -> uniform


def test_partial_feedback_counts():
    keys = np.random.default_rng(0).random((10, 30))
    assert [int(obs_mask(keys, q)[0].sum()) for q in (0.25, 0.5, 0.75, 1.0)] == [8, 15, 22, 30]


def test_static_ridge_frozen_after_first_estimate():
    r = run_replication(CFG, _data())
    # fixed-weight PSDF must choose from weights frozen at t=1: its regret path may vary only through Z,R, never via weights
    e = OutcomeWeights("static", 5, 15, 0.1, 12, 0.9)
    d = _data(); W = np.zeros(5)
    ws = []
    for t in range(10):
        w, _ = e.weights(t); ws.append(w.copy())
        e.update(t, np.hstack([d.Z[t], d.R[t][:, None]]), np.random.rand(30))
    assert all(np.array_equal(ws[1], ws[i]) for i in range(2, 10))


def test_sanity_oracle_and_aggregation_floor():
    r = run_replication(CFG, _data())
    assert r["reg"]["diag_oracle_additive"].max() < 1e-12
    assert r["reg"]["diag_psdf_oracle_mult"].mean() > 0


def test_sanity_ahp_exact_when_consistent_and_unclipped():
    w = (0.30, 0.25, 0.20, 0.15, 0.10)
    cfg = dataclasses.replace(CFG, w_pre=w, w_post=w, sd_ahp=0.0)
    r = run_replication(cfg, _data(cfg))
    assert r["reg"]["ahp_k10"].max() < 1e-12 and r["reg"]["ahp_static"].max() < 1e-12
    res = ahp_elicit(stated_vstar(cfg, np.array(w), np.zeros(5)), np.zeros(15), 0.0)
    assert abs(res.cr) < 1e-9 and res.clipped == 0 and abs(res.v.sum() - 1) < 1e-12


def test_sanity_default_weights_trigger_clipping():
    res = ahp_elicit(stated_vstar(CFG, np.array(CFG.w_pre), np.zeros(5)), np.zeros(15), 0.0)
    assert res.clipped >= 3                                                           # 14, 10 and rho/w5 = 10 exceed 9


def test_preference_gap_distorts_ahp_target_only():
    cfg = dataclasses.replace(CFG, gap=1.0)
    z = np.array([1.0, -1.0, 0.5, 0.0, -0.5])
    v = stated_vstar(cfg, np.array(cfg.w_pre), z)
    assert abs(v[-1] - cfg.rho / (1 + cfg.rho)) < 1e-12 and not np.allclose(v[:-1] * (1 + cfg.rho) / (1 + cfg.rho), np.append(cfg.w_pre, 0)[:5] / (1 + cfg.rho))


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as ex:  # noqa
                fails += 1; print("FAIL", name, repr(ex))
    print(f"{fails} failure(s)")
    sys.exit(1 if fails else 0)
