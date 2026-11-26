import random
from itertools import permutations

from hyperphantasia import PhantasiaLoop, Scene, Status, reason
from hyperphantasia.scene import Query


def test_shared_anchor_is_indeterminate_and_surprising():
    i = reason("The cup is left of the plate. The fork is left of the plate. Is the fork left of the cup?")
    assert i.status is Status.POSSIBLE
    assert i.glance is Status.IMPOSSIBLE  # the first picture puts things in mention order
    assert i.signals.surprise
    assert i.verdict.witness is not None and i.verdict.counterexample is not None


def test_transitive_chain_is_necessary_with_proof_and_two_channels():
    i = reason("The fork is left of the cup. The cup is left of the plate. "
               "The knife is right of the plate. Is the fork left of the knife?")
    assert i.status is Status.NECESSARY
    assert i.signals.corroboration == 2 and i.signals.dissonance == 0
    assert "fork < cup < plate < knife" in i.explanation
    assert i.confidence > 0.95


def test_contradiction_is_localised_to_the_cycle():
    i = reason("The red box is left of the blue box. The blue box is left of the green box. "
               "The lamp is above the red box. The green box is left of the red box. "
               "Is the lamp above the blue box?")
    assert i.status is Status.INCONSISTENT
    assert i.verdict.conflict_set == ["p1", "p2", "p4"]  # p3 (the lamp) is not to blame


def test_two_dimensional_relative_query():
    loose = reason("The lamp is directly above the cup. The cup is left of the plate. "
                   "Where is the lamp relative to the plate?")
    # Nothing says how high the plate is, so only the horizontal relation is fixed.
    assert loose.status is Status.POSSIBLE
    assert "left of plate" in loose.answer and "undetermined vertically" in loose.answer
    tight = reason("The lamp is directly above the cup. The cup is left of the plate. "
                   "The plate is in the same row as the cup. Where is the lamp relative to the plate?")
    assert tight.status is Status.NECESSARY
    assert tight.answer == "lamp is left of plate; above plate."
    assert tight.signals.corroboration == 2  # possibility space and proof graph both vouch


def test_row_mode_adjacency_and_betweenness():
    i = reason("Five books are on a shelf in a row. The atlas is immediately left of the bible. "
               "The cookbook is between the atlas and the diary. The diary is right of the encyclopedia. "
               "The encyclopedia is left of the atlas. Is the bible next to the cookbook?")
    assert i.scene.unparsed == []  # the scene-setting sentence is context, not a missed premise
    assert i.status is Status.POSSIBLE
    assert i.imagination.vividness["encyclopedia"] == 1.0  # leftmost in every arrangement


def _brute(n_ents, constraints, query):
    """Independent oracle: strict orders over n slots via permutations."""
    verdicts = set()
    for perm in permutations(range(n_ents)):
        if all(perm[a] < perm[b] for a, b in constraints):
            verdicts.add(perm[query[0]] < perm[query[1]])
    if not verdicts:
        return Status.INCONSISTENT
    if verdicts == {True}:
        return Status.NECESSARY
    if verdicts == {False}:
        return Status.IMPOSSIBLE
    return Status.POSSIBLE


def test_random_orderings_match_an_independent_oracle():
    rng = random.Random(7)
    loop = PhantasiaLoop()
    for _ in range(150):
        n = rng.randint(3, 6)
        names = [f"e{k}" for k in range(n)]
        cons = [tuple(rng.sample(range(n), 2)) for _ in range(rng.randint(1, n + 1))]
        q = tuple(rng.sample(range(n), 2))
        scene = Scene("spatial", extras={"row": True})
        for nm in names:
            scene.entity(nm)
        for a, b in cons:
            scene.add("left_of", names[a], names[b], source=f"{names[a]} left of {names[b]}")
        scene.query = Query("left_of", (names[q[0]], names[q[1]]))
        insight = loop.think(scene)
        assert insight.status is _brute(n, cons, q)
        assert insight.signals.dissonance == 0  # closure and enumeration never disagree


def test_weak_orders_in_two_dimensions_keep_channels_consistent():
    rng = random.Random(11)
    kinds = ["left_of", "right_of", "above", "below", "same_column", "same_row", "directly_above"]
    loop = PhantasiaLoop()
    for _ in range(120):
        n = rng.randint(3, 5)
        scene = Scene("spatial")
        names = [scene.entity(f"o{k}") for k in range(n)]
        for _ in range(rng.randint(1, n + 1)):
            a, b = rng.sample(names, 2)
            scene.add(rng.choice(kinds), a, b, source="random")
        a, b = rng.sample(names, 2)
        scene.query = Query(rng.choice(kinds + ["relative"]), (a, b))
        insight = loop.think(scene)
        assert insight.signals.dissonance == 0, scene.describe()
