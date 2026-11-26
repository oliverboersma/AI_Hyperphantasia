"""Turn evidence into a verdict, with thresholds fixed before looking at results."""

from __future__ import annotations

from dataclasses import dataclass

from .battery import Evidence

COMPOSE_OK = 0.90        # behaviour: questions needing composition
DECODE_ABS = 0.80        # representation: probe accuracy ...
DECODE_MARGIN = 0.20     # ... and margin over the best control (untrained net, surface counts)
TRANSFER_OK = 0.80       # causation: transplant makes the answer follow the other world
SUBSPACE_OK = 0.70       # transplanting only the rank subspace suffices ...
SUBSPACE_MARGIN = 0.40   # ... and beats a random subspace of the same size by this much

VERDICTS = {
    "standing scene": "Imagination in the mechanism: a model of the whole scene exists before the "
                      "question is read, and the answer is produced from it.",
    "on-demand scene": "Imagination on demand: there is no standing model of the scene, but when the "
                       "question arrives the positions of the asked-about items are built and used.",
    "epiphenomenal": "Scene information is present but idle: it can be decoded, yet editing it does "
                     "not change the answer. Something else does the work.",
    "something else": "Something else: the questions are answered correctly, but no scene "
                      "representation that this detector can find is used (it may compare relations "
                      "directly, or encode positions non-linearly).",
    "no imagination": "No working imagination: questions that need premises combined are not "
                      "answered, so behaviour is explained by stated pairs and surface statistics.",
}


@dataclass
class Verdict:
    label: str
    text: str
    reasons: list[str]


def _decodable(p) -> bool:
    controls = [c for c in (p["untrained"], p["surface"]) if c is not None]
    return p["acc"] >= DECODE_ABS and p["acc"] - max(controls, default=p["chance"]) >= DECODE_MARGIN


def judge(ev: Evidence) -> Verdict:
    b = ev.behaviour
    reasons = [f"composition accuracy {b['composition']:.2f} (threshold {COMPOSE_OK})"]
    if b["composition"] < COMPOSE_OK:
        return Verdict("no imagination", VERDICTS["no imagination"], reasons)

    scene_sites = [k for k, p in ev.probes.items() if k[1] == "scene" and _decodable(p)]
    standing = [k for k in scene_sites if ev.tracing.get(k, 0) >= TRANSFER_OK]
    if scene_sites:
        best = max(scene_sites, key=lambda k: ev.probes[k]["acc"])
        reasons.append(f"whole scene decodable before the question at layer {best[0]} "
                       f"(probe {ev.probes[best]['acc']:.2f})")
    if standing:
        k = max(standing, key=lambda k: ev.tracing[k])
        reasons.append(f"transplanting that scene site makes the answer follow the other world "
                       f"{ev.tracing[k]:.0%} of the time (layer {k[0]})")
        return Verdict("standing scene", VERDICTS["standing scene"], reasons)

    causal_x = [layer for layer, s in ev.subspace.items()
                if s["rank"] >= SUBSPACE_OK and s["rank"] - s["random"] >= SUBSPACE_MARGIN
                and _decodable(ev.probes[(layer, "X")])]
    if causal_x:
        layer = max(causal_x, key=lambda l: ev.subspace[l]["rank"])
        s = ev.subspace[layer]
        reasons.append(f"rank of the asked-about letter is decodable at layer {layer} "
                       f"(probe {ev.probes[(layer, 'X')]['acc']:.2f})")
        reasons.append(f"transplanting only that {s['dims']}-dimensional rank subspace moves the answer "
                       f"{s['rank']:.0%} of the time, versus {s['random']:.0%} for a random subspace")
        if scene_sites:
            reasons.append("the scene site holds the information but the answer is not read from it")
        return Verdict("on-demand scene", VERDICTS["on-demand scene"], reasons)

    if scene_sites or any(_decodable(p) for k, p in ev.probes.items() if k[1] in ("X", "Y")):
        reasons.append("decodable rank information exists, but transplanting it does not move the answer")
        return Verdict("epiphenomenal", VERDICTS["epiphenomenal"], reasons)
    reasons.append("no site carries decodable rank information beyond the controls")
    return Verdict("something else", VERDICTS["something else"], reasons)
