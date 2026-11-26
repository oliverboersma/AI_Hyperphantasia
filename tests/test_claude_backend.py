"""The Claude backend, exercised against a stub client (no network, no key)."""

import json
from types import SimpleNamespace

import pytest

from hyperphantasia import PhantasiaLoop, Status
from hyperphantasia.backends.claude import BackendRefusal, ClaudeBackend

SCENE = {
    "domain": "spatial", "title": "table", "row": False, "existential_import": True,
    "entities": [{"id": "cup", "label": "the cup"}, {"id": "plate", "label": "plate"},
                 {"id": "fork", "label": "fork"}],
    "relations": [{"kind": "left_of", "args": ["cup", "plate"], "source": "The cup is left of the plate."},
                  {"kind": "left_of", "args": ["fork", "plate"], "source": "The fork is left of the plate."},
                  {"kind": "left_of", "args": ["fork", "spoon"], "source": "The fork is left of a spoon."}],
    "query": {"kind": "left_of", "args": ["fork", "cup"], "text": "Is the fork left of the cup?"},
    "initial": [], "goal": [], "steps": [], "unparsed": [], "parse_confidence": 0.9,
}


class StubMessages:
    def __init__(self, replies, stop_reason="end_turn"):
        self.replies, self.calls, self.stop_reason = list(replies), [], stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.replies.pop(0)
        return SimpleNamespace(stop_reason=self.stop_reason, stop_details=None,
                               content=[SimpleNamespace(type="text", text=text)])


def stub(replies, **kw):
    msgs = StubMessages(replies, **kw)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def test_envision_requests_structured_scene_and_validates_it():
    client, msgs = stub([json.dumps(SCENE)])
    scene = ClaudeBackend(client=client, polish=False).envision("...")
    call = msgs.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in call["betas"]
    assert scene.entities["cup"] == "cup"
    assert len(scene.relations) == 2  # the dangling "spoon" premise is not silently trusted
    assert scene.unparsed and scene.parse_confidence < 0.9


def test_verbal_disagreement_triggers_a_perception_check():
    replies = [json.dumps(SCENE),                                     # envision
               json.dumps({"answer": "yes", "confidence": 0.8}),      # intuit (first impression)
               json.dumps({"faithful": False, "issues": ["spoon missing"]}),  # verify
               "Polished explanation."]                               # polish
    client, msgs = stub(replies)
    loop = PhantasiaLoop(ClaudeBackend(client=client), use_intuition=True)
    i = loop.think("The cup is left of the plate. The fork is left of the plate. Is the fork left of the cup?")
    assert i.status is Status.POSSIBLE  # the model decides, not the verbal guess
    assert next(c for c in i.channels if c.name == "verbal").verdict is Status.NECESSARY
    assert any("re-checked" in s.summary for s in i.trace)
    assert i.signals.perception < 0.5
    assert i.explanation == "Polished explanation."
    assert [c["output_config"]["effort"] for c in msgs.calls] == ["high", "low", "medium", "medium"]


def test_refusal_is_surfaced():
    client, _ = stub(["{}"], stop_reason="refusal")
    with pytest.raises(BackendRefusal):
        ClaudeBackend(client=client).envision("...")
