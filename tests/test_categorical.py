from itertools import product

import pytest

from hyperphantasia import PhantasiaLoop, Scene, Status, reason
from hyperphantasia.scene import Query

FIGURES = {  # (major premise args, minor premise args) with S, M, P
    1: (("M", "P"), ("S", "M")),
    2: (("P", "M"), ("S", "M")),
    3: (("M", "P"), ("M", "S")),
    4: (("P", "M"), ("M", "S")),
}
MOODS = {"A": "all", "E": "no", "I": "some", "O": "some_not"}


def syllogism(major, minor, conclusion, figure, existential_import=True) -> Scene:
    scene = Scene("categorical", extras={"existential_import": existential_import})
    for t in ("S", "M", "P"):
        scene.entity(t)
    (a, b), (c, d) = FIGURES[figure]
    scene.add(MOODS[major], a.lower(), b.lower(), source=f"{major} {a}{b}")
    scene.add(MOODS[minor], c.lower(), d.lower(), source=f"{minor} {c}{d}")
    scene.query = Query(MOODS[conclusion], ("s", "p"))
    return scene


@pytest.mark.parametrize("existential_import, expected", [(True, 24), (False, 15)])
def test_exactly_the_classical_valid_syllogisms(existential_import, expected):
    """256 syllogistic forms; logic says 24 are valid with existential import, 15 without."""
    loop = PhantasiaLoop()
    valid = []
    for (ma, mi, co), fig in product(product("AEIO", repeat=3), FIGURES):
        i = loop.think(syllogism(ma, mi, co, fig, existential_import))
        assert i.signals.dissonance == 0  # Venn shading and enumeration agree on every form
        if i.status is Status.NECESSARY:
            valid.append(f"{ma}{mi}{co}-{fig}")
    assert len(valid) == expected
    for name in ("AAA-1", "EAE-1", "AII-1", "EIO-1", "EAE-2", "AOO-2", "OAO-3", "IAI-4"):
        assert name in valid  # Barbara, Celarent, Darii, Ferio, Cesare, Baroco, Bocardo, Dimaris


def test_classic_illusion_is_caught():
    i = reason("All artists are beekeepers. Some beekeepers are chemists. "
               "Does it follow that some artists are chemists?")
    assert i.glance is Status.NECESSARY  # the greedy first picture merges the individuals
    assert i.status is Status.POSSIBLE
    assert i.signals.surprise and i.signals.corroboration == 2
    assert "artists are chemists" in i.answer


def test_barbara_and_what_follows():
    assert reason("All cats are mammals. All mammals are animals. "
                  "Does it follow that all cats are animals?").status is Status.NECESSARY
    open_q = reason("Every artist is a beekeeper. All beekeepers are chemists. What follows?")
    assert open_q.status is Status.OPEN
    assert "All artists are chemists" in open_q.answer


def test_inconsistent_premises_are_named():
    i = reason("All poets are dreamers. No poets are dreamers. Some dreamers are poets. Are all poets dreamers?")
    assert i.status is Status.INCONSISTENT
    assert set(i.verdict.conflict_set) <= {"p1", "p2", "p3"} and len(i.verdict.conflict_set) == 2


def test_questions_without_a_verb_between_terms():
    i = reason("All cats are mammals. All mammals are animals. Are all cats animals?")
    assert i.status is Status.NECESSARY and not i.scene.unparsed
    j = reason("Some dogs are pets. Are some pets not dogs?")
    assert (j.scene.query.kind, j.status) == ("some_not", Status.POSSIBLE)
