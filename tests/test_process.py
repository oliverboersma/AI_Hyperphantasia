import random
from pathlib import Path

from hyperphantasia import PhantasiaLoop, Status, reason
from hyperphantasia.domains import ProcessDomain

TYRE = (Path(__file__).parent.parent / "examples" / "tyre.txt").read_text()


def test_tyre_fails_and_is_repaired():
    i = reason(TYRE)
    assert i.status is Status.FAILS
    assert i.glance is Status.WORKS and i.signals.surprise
    assert "move 'Loosen the nuts' before 'Jack up the car'" in i.answer
    fixed = i.verdict.repair
    assert ProcessDomain().run(i.scene, fixed).ok
    assert "At the front door: Loosen the nuts." in i.explanation  # rehearsed as a memory palace


def test_working_procedure_and_parallel_steps():
    i = reason("""Process: Make tea
Start: kettle empty, cup clean
Goal: tea ready
1. Fill the kettle - needs: kettle empty; makes: kettle full; ends: kettle empty
2. Boil the water - needs: kettle full; makes: water hot
3. Put a teabag in the cup - needs: cup clean; makes: bag in cup
4. Pour the water - needs: water hot, bag in cup; makes: tea ready
""")
    assert i.status is Status.WORKS and not i.signals.surprise
    timeline = next(c for c in i.channels if c.name == "timeline").rendering
    assert "Fill the kettle  ||  Put a teabag in the cup" in timeline


def test_missing_step_is_diagnosed():
    i = reason("""Process: Send a letter
Start: letter written
Goal: letter posted
1. Seal the envelope - needs: letter in envelope; makes: envelope sealed
2. Post it - needs: envelope sealed, stamp on; makes: letter posted
""")
    assert i.status is Status.FAILS and i.verdict.repair is None
    assert "nothing ever makes 'letter in envelope' true" in i.answer
    assert "nothing ever makes 'stamp on' true" in i.answer


def test_simulation_and_causal_links_agree_on_random_orders():
    loop = PhantasiaLoop()
    base = loop.backend.envision(TYRE)
    rng = random.Random(3)
    for _ in range(60):
        scene = loop.backend.envision(TYRE)
        rng.shuffle(scene.extras["steps"])
        insight = loop.think(scene)
        assert insight.signals.dissonance == 0
        if insight.verdict.repair:
            assert ProcessDomain().run(scene, insight.verdict.repair).ok
    assert base.extras["steps"][0] == "jack_up_the_car"
