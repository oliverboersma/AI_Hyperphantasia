"""The imagination detector must be right about subjects whose internals we know."""

import numpy as np
import pytest

from hyperphantasia.detect import battery, verdict
from hyperphantasia.detect import world as W
from hyperphantasia.detect.subjects import Patch, SceneEngine, SurfaceHeuristic


def test_world_counterfactuals_do_what_they_claim():
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = W.sample(rng, 5)
        assert len(p.premises) == 4
        cf = W.scene_counterfactual(p, rng)
        assert cf is not None and set(cf.order) == set(p.order) and cf.answer != p.answer
        rc = W.rank_counterfactual(p, rng)
        if rc is not None:
            assert rc.rank[p.query[1]] == p.rank[p.query[1]] and rc.answer != p.answer


def test_surface_counts_cannot_place_the_middle():
    p = W.Problem((0, 1, 2, 3, 4), ((0, W.LT, 1), (2, W.GT, 1), (2, W.LT, 3), (4, W.GT, 3)), (1, 3))
    f = W.surface_features(p)
    assert (f[1][:2] == f[2][:2]).all() and (f[2][:2] == f[3][:2]).all()  # identical counts


def test_detector_recognises_a_real_scene():
    ev = battery.examine(SceneEngine(), quick=True)
    assert verdict.judge(ev).label == "standing scene"


def test_detector_is_not_fooled_by_a_surface_heuristic():
    ev = battery.examine(SurfaceHeuristic(), quick=True)
    v = verdict.judge(ev)
    assert v.label == "no imagination"
    # Its "scene" is a lossless copy of the premises, so transplanting it does move answers;
    # the detector must not mistake that for imagination: ranks are not decodable beyond counts.
    assert ev.tracing[(1, "scene")] > 0.8
    assert ev.probes[(1, "scene")]["acc"] - ev.probes[(1, "scene")]["surface"] < 0.05


def test_rank_subspace_patch_moves_the_engine():
    rng = np.random.default_rng(1)
    base, cf = [], []
    while len(base) < 50:
        p = W.sample(rng, 5)
        c = W.rank_counterfactual(p, rng)
        if c:
            base.append(p), cf.append(c)
    eng = SceneEngine()
    _, acts = eng.run(cf)
    pred, _ = eng.run(base, Patch(1, "X", acts[(1, "X")]))
    assert (pred == W.labels(cf)).all()


def test_network_patching_is_exact():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from hyperphantasia.detect import nets
    for kind in ("plain", "slots"):
        cfg = nets.NetConfig(kind=kind, d=32, layers=2, enc_layers=1, k_slots=4)
        params = nets.init(cfg, 0)
        probs = W.batch(np.random.default_rng(2), 16)
        toks = jnp.asarray(nets.encode(cfg, probs))
        logits, resid = nets.forward(params, toks, cfg)
        pos = jnp.asarray(nets.roles(cfg)["scene"])
        same, _ = nets.forward(params, toks, cfg, 1, pos, resid[1][:, pos], jnp.eye(cfg.d))
        assert np.allclose(np.asarray(logits), np.asarray(same), atol=1e-5)  # self-patch is a no-op
