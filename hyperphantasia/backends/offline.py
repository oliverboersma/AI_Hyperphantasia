"""Offline perception: a small restricted-English parser.

It understands three registers:

  spatial      "The cup is left of the plate. Is the fork left of the knife?"
  categorical  "All artists are beekeepers. Some beekeepers are chemists.
                Does it follow that some artists are chemists?"
  process      Process: Change a tyre
               Start: car on ground, nuts tight
               Goal: spare on
               1. Loosen the nuts - needs: nuts tight, car on ground; makes: nuts loose; ends: nuts tight

It is deliberately strict. A sentence it cannot represent is recorded as a
hole in the scene rather than guessed at, which lowers perception confidence.
Use ``ClaudeBackend`` for open-ended language.
"""

from __future__ import annotations

import re

from ..schema import perception_confidence
from ..scene import Query, Scene, display, plural, singular, slug

SPATIAL_PHRASES = [
    (r"directly to the left of|directly left of", "directly_left_of"),
    (r"directly to the right of|directly right of", "directly_right_of"),
    (r"immediately to the left of|immediately left of|just to the left of|just left of", "immediately_left_of"),
    (r"immediately to the right of|immediately right of|just to the right of|just right of", "immediately_right_of"),
    (r"directly above|directly over|right above", "directly_above"),
    (r"directly below|directly beneath|directly under|right below", "directly_below"),
    (r"in the same column as", "same_column"),
    (r"in the same row as", "same_row"),
    (r"to the left of|left of|west of", "left_of"),
    (r"to the right of|right of|east of", "right_of"),
    (r"on top of|above|over|north of", "above"),
    (r"underneath|beneath|below|under|south of", "below"),
    (r"next to|beside|adjacent to", "next_to"),
]
REL_ALT = "|".join(f"(?:{p})" for p, _ in SPATIAL_PHRASES)
SPATIAL_WORDS = re.compile(r"\b(left|right|above|below|under|over|beneath|beside|next to|between|"
                           r"north|south|east|west|column|row|adjacent)\b", re.IGNORECASE)
ROW_CUE = re.compile(r"\b(in a row|on a shelf|in a line|lined? up|in a queue|side by side)\b", re.IGNORECASE)
VERB = r"(?:is|are|sits|sit|stands|stand|lies|lie|was|were)"

QUERY_PREFIX = re.compile(r"^(?:does it follow that|therefore,?|so,?|hence,?|is it true that|"
                          r"can we conclude that|must it be (?:true )?that|does that mean that|"
                          r"does this mean that|it follows that)\s+", re.IGNORECASE)
CATEGORICAL = [
    (re.compile(r"^(?:all|every|each)\s+(?P<x>.+?)\s+(?:are|is)\s+(?:an?\s+)?(?P<y>.+)$", re.I), "all"),
    (re.compile(r"^no\s+(?P<x>.+?)\s+(?:are|is)\s+(?:an?\s+)?(?P<y>.+)$", re.I), "no"),
    (re.compile(r"^some\s+(?P<x>.+?)\s+(?:are|is)\s+not\s+(?:an?\s+)?(?P<y>.+)$", re.I), "some_not"),
    (re.compile(r"^not\s+(?:all|every)\s+(?P<x>.+?)\s+(?:are|is)\s+(?:an?\s+)?(?P<y>.+)$", re.I), "some_not"),
    (re.compile(r"^some\s+(?P<x>.+?)\s+(?:are|is)\s+(?:an?\s+)?(?P<y>.+)$", re.I), "some"),
]
PROCESS_KEYS = {"needs": "needs", "requires": "needs", "makes": "makes", "gives": "makes",
                "produces": "makes", "ends": "ends", "removes": "ends", "consumes": "ends",
                "uses up": "ends"}
KEY_RE = re.compile(r"\b(needs|requires|makes|gives|produces|ends|removes|consumes|uses up)\s*:", re.I)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.findall(r"[^.?!;\n]+[.?!;]?", text) if s.strip(" .;")]


def strip_end(s: str) -> str:
    return s.strip().rstrip(".?!;").strip()


def is_question(s: str) -> bool:
    return s.strip().endswith("?") or bool(QUERY_PREFIX.match(s.strip())) or \
        bool(re.match(r"^(is|are|does|do|where|what|which|can|must)\b", s.strip(), re.I))


class OfflineBackend:
    """Perception without a language model. Implements ``envision`` only."""

    name = "offline"

    def envision(self, text: str) -> Scene:
        if re.search(r"^\s*(process|procedure)\s*:", text, re.I | re.M) or KEY_RE.search(text):
            return self._process(text)
        sents = sentences(text)
        for s in sents:
            body = QUERY_PREFIX.sub("", strip_end(s))
            if any(p.match(body) for p, _ in CATEGORICAL):
                return self._categorical(sents)
        return self._spatial(sents)

    # ------------------------------------------------------------- spatial
    def _spatial(self, sents: list[str]) -> Scene:
        scene = Scene(domain="spatial")
        unparsed = []
        for s in sents:
            body = strip_end(s)
            if ROW_CUE.search(body):
                scene.extras["row"] = True
            if is_question(s):
                if not self._spatial_query(scene, body):
                    unparsed.append(s)
                continue
            if self._spatial_statement(scene, body, s):
                continue
            if ROW_CUE.search(body) and not re.search(r"\b(left|right|above|below|between|next)\b", body, re.I):
                continue  # scene-setting ("five books on a shelf in a row"), not a premise
            if SPATIAL_WORDS.search(body):
                unparsed.append(s)  # looks like a premise we failed to read
        scene.unparsed = unparsed
        scene.parse_confidence = perception_confidence(1.0, len(unparsed))
        if not scene.relations and not scene.query:
            scene.unparsed = [s for s in sents]  # nothing here could be pictured
            scene.parse_confidence = 0.0
        return scene

    def _spatial_statement(self, scene: Scene, body: str, source: str) -> bool:
        m = re.match(rf"^(?P<a>.+?)\s+{VERB}\s+between\s+(?P<b>.+?)\s+and\s+(?P<c>.+)$", body, re.I)
        if m:
            scene.add("between", scene.entity(m["a"]), scene.entity(m["b"]), scene.entity(m["c"]),
                      source=source)
            scene.extras["row"] = True
            return True
        m = re.match(rf"^(?P<a>.+?)\s+and\s+(?P<b>.+?)\s+{VERB}\s+in\s+the\s+same\s+(?P<w>row|column)$",
                     body, re.I)
        if m:
            scene.add("same_" + m["w"].lower(), scene.entity(m["a"]), scene.entity(m["b"]), source=source)
            return True
        m = re.match(rf"^(?P<a>.+?)\s+{VERB}\s+(?:placed\s+|located\s+|positioned\s+|on the\s+)?"
                     rf"(?P<rel>{REL_ALT})\s+(?P<b>.+)$", body, re.I)
        if m:
            kind = next(k for p, k in SPATIAL_PHRASES if re.fullmatch(p, m["rel"], re.I))
            scene.add(kind, scene.entity(m["a"]), scene.entity(m["b"]), source=source)
            return True
        return False

    def _spatial_query(self, scene: Scene, body: str) -> bool:
        m = re.match(r"^where\s+is\s+(?P<a>.+?)\s+(?:relative to|in relation to|compared to|"
                     r"with respect to)\s+(?P<b>.+)$", body, re.I)
        if m:
            scene.query = Query("relative", (scene.entity(m["a"]), scene.entity(m["b"])), body)
            return True
        m = re.match(r"^(?:is|are)\s+(?P<a>.+?)\s+between\s+(?P<b>.+?)\s+and\s+(?P<c>.+)$", body, re.I)
        if m:
            scene.query = Query("between", tuple(scene.entity(m[k]) for k in "abc"), body)
            scene.extras["row"] = True
            return True
        m = re.match(rf"^(?:is|are)\s+(?P<a>.+?)\s+(?P<rel>{REL_ALT})\s+(?P<b>.+)$", body, re.I)
        if m:
            kind = next(k for p, k in SPATIAL_PHRASES if re.fullmatch(p, m["rel"], re.I))
            scene.query = Query(kind, (scene.entity(m["a"]), scene.entity(m["b"])), body)
            return True
        return False

    # --------------------------------------------------------- categorical
    def _categorical(self, sents: list[str]) -> Scene:
        scene = Scene(domain="categorical")
        unparsed, questions = [], []
        for s in sents:  # premises first, so questions can be read against known terms
            body = strip_end(s)
            core = QUERY_PREFIX.sub("", body)
            if re.match(r"^what (follows|can (we|be) conclude)", core, re.I):
                continue  # open question: report what follows
            if is_question(s):
                questions.append((s, body, core))
                continue
            for pattern, kind in CATEGORICAL:
                m = pattern.match(core)
                if m:
                    scene.add(kind, self._term(scene, m["x"]), self._term(scene, m["y"]), source=s)
                    break
            else:
                unparsed.append(s)
        for s, body, core in questions:
            parsed = self._categorical_question(scene, core)
            if parsed:
                scene.query = Query(parsed[0], parsed[1:], body)
            else:
                unparsed.append(s)
        scene.unparsed = unparsed
        scene.parse_confidence = perception_confidence(1.0, len(unparsed))
        return scene

    def _categorical_question(self, scene: Scene, core: str):
        for pattern, kind in CATEGORICAL:
            m = pattern.match(core)
            if m:
                return kind, self._term(scene, m["x"]), self._term(scene, m["y"])
        # "Are all cats animals?" has no verb between the terms: split at known terms.
        m = re.match(r"^(?:are|is)\s+(all|every|some|no)\s+(.+)$", core, re.I)
        if not m:
            return None
        quant, words = m.group(1).lower(), m.group(2).split()
        negated = False
        for i in range(1, len(words)):
            left, right = words[:i], words[i:]
            if quant == "some" and right and right[0].lower() == "not":
                right, negated = right[1:], True
            x, y = slug(singular(" ".join(left))), slug(singular(" ".join(right)))
            if x in scene.entities and y in scene.entities:
                kind = {"all": "all", "every": "all", "no": "no"}.get(quant, "some_not" if negated else "some")
                return kind, x, y
            negated = False
        return None

    @staticmethod
    def _term(scene: Scene, phrase: str) -> str:
        """Terms are identified by their singular form and displayed in the plural."""
        one = singular(phrase)
        eid = scene.entity(one)
        scene.entities[eid] = plural(scene.entities[eid]) if scene.entities[eid] == display(one) \
            else scene.entities[eid]
        return eid

    # ------------------------------------------------------------- process
    def _process(self, text: str) -> Scene:
        scene = Scene(domain="process")
        steps, unparsed = [], []
        facts = lambda v: [scene.entity(f) for f in re.split(r",|\band\b", v) if f.strip(" .")]
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^(process|procedure)\s*:\s*(.+)$", line, re.I)
            if m:
                scene.title = m.group(2).strip()
                continue
            m = re.match(r"^(start|initially|given|initial state)\s*:\s*(.*)$", line, re.I)
            if m:
                scene.extras["initial"] = facts(m.group(2))
                continue
            m = re.match(r"^(goal|end|target)\s*:\s*(.*)$", line, re.I)
            if m:
                scene.extras["goal"] = facts(m.group(2))
                continue
            m = re.match(r"^(?:\d+[.)]|[-*]|step\s*\d*\s*[:.)]?)\s*(.+)$", line, re.I)
            if not m:
                unparsed.append(line)
                continue
            body = m.group(1)
            k = KEY_RE.search(body)
            label = (body[:k.start()] if k else body).strip(" -—–|:,;.")
            sid = scene.entity(label)
            steps.append(sid)
            if k:
                for part in re.split(r";", body[k.start():]):
                    km = re.match(r"\s*(needs|requires|makes|gives|produces|ends|removes|consumes|uses up)"
                                  r"\s*:\s*(.*)$", part, re.I)
                    if not km:
                        if part.strip(" ."):
                            unparsed.append(part.strip())
                        continue
                    for f in facts(km.group(2)):
                        scene.add(PROCESS_KEYS[km.group(1).lower()], sid, f, source=line)
        scene.extras["steps"] = steps
        scene.extras.setdefault("initial", [])
        scene.extras.setdefault("goal", [])
        scene.query = Query("achieves", (), "Does the procedure reach its goal?")
        scene.unparsed = unparsed
        scene.parse_confidence = perception_confidence(1.0, len(unparsed))
        return scene
