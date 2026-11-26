from hyperphantasia import ListenerModel, PhantasiaLoop, Status
from hyperphantasia.attune import CalibrationLedger, honesty_violations
from hyperphantasia.backends.offline import OfflineBackend

ANCHOR = "The cup is left of the plate. The fork is left of the plate. Is the fork left of the cup?"
ANCHOR2 = "The pen is left of the book. The key is left of the book. Is the key left of the pen?"


def test_self_model_learns_from_its_own_surprise(tmp_path):
    loop = PhantasiaLoop(state_dir=tmp_path)
    before = loop.me.glance_reliability("spatial")
    loop.think(ANCHOR)
    assert loop.me.glance_reliability("spatial") < before
    assert any("first picture" in l["text"] for l in loop.me.lessons)

    # A fresh loop reloads the self-model and recognises the same shape of problem.
    again = PhantasiaLoop(state_dir=tmp_path)
    second = again.think(ANCHOR2)
    attune = second.trace[0].summary
    assert "analog" in attune
    assert "same shape as an earlier problem" in second.explanation


def test_feedback_calibrates_confidence():
    ledger = CalibrationLedger()
    assert ledger.calibrate("spatial", 0.95) == 0.95  # no history: trust the raw signal
    for _ in range(10):
        ledger.record("spatial", 0.95, False)
    assert ledger.calibrate("spatial", 0.95) < 0.5
    assert ledger.calibrate("categorical", 0.95) == 0.95  # calibration is per domain


def test_feedback_blames_perception_when_reasoning_was_exact():
    loop = PhantasiaLoop()
    i = loop.think(ANCHOR)
    msg = loop.feedback(i, correct=False, note="the user meant a circular table")
    assert "envision" in msg
    assert loop.me.blamed


def test_honesty_invariant():
    assert honesty_violations("This is definitely right.", 0.8) == ["definitely"]
    assert honesty_violations("This is definitely right.", 0.99) == []
    assert honesty_violations("It necessarily follows.", 0.8) == []  # logical, not epistemic


class OverclaimingBackend(OfflineBackend):
    def polish(self, explanation, listener, confidence):
        return "This is certainly and definitely true."


def test_overclaiming_rewrite_is_rejected():
    i = PhantasiaLoop(OverclaimingBackend()).think(
        "The atlas is far to the left. The atlas is left of the bible. Is the atlas left of the bible?")
    # Perception was imperfect (one sentence not understood), so certainty was not earned.
    assert i.confidence < 0.97
    assert "definitely" not in i.explanation
    assert any("rejected" in s.summary for s in i.trace)


def test_listener_shapes_the_explanation():
    novice = PhantasiaLoop().think(ANCHOR, ListenerModel("picture", "novice"))
    words = PhantasiaLoop().think(ANCHOR, ListenerModel("words", "novice"))
    expert = PhantasiaLoop().think(ANCHOR, ListenerModel("picture", "expert"))
    assert "+---" in novice.explanation and "+---" not in words.explanation
    assert "fork  <  cup  <  plate" in words.explanation
    assert len(expert.explanation) < len(novice.explanation)
    assert novice.status is words.status is expert.status is Status.POSSIBLE
