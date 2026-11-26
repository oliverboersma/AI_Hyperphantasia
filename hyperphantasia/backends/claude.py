"""Perception and voice through Claude.

The language model does what it is good at: reading open-ended language into a
structured scene (ENVISION), giving a fast verbal first impression (an
independent "verbal" channel), re-reading the scene against the original text
when channels disagree (perception check), and rewriting the explanation for a
particular listener (TRANSLATE). It never decides the answer: the imagined
model does, and the loop checks the rewrite for unearned certainty.

Requires ``pip install anthropic`` and credentials (ANTHROPIC_API_KEY or an
``ant auth login`` profile).
"""

from __future__ import annotations

import json
from typing import Any

from ..attune import ListenerModel, hedge
from ..schema import ALL_QUERY_KINDS, ALL_RELATION_KINDS, scene_from_dict
from ..scene import Scene

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_STR_LIST = {"type": "array", "items": {"type": "string"}}
SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["domain", "title", "entities", "relations", "query", "row", "existential_import",
                 "initial", "goal", "steps", "unparsed", "parse_confidence"],
    "properties": {
        "domain": {"type": "string", "enum": ["spatial", "categorical", "process"]},
        "title": {"type": "string"},
        "entities": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "label"],
            "properties": {"id": {"type": "string"}, "label": {"type": "string"}}}},
        "relations": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["kind", "args", "source"],
            "properties": {"kind": {"type": "string", "enum": ALL_RELATION_KINDS},
                           "args": _STR_LIST, "source": {"type": "string"}}}},
        "query": {"type": "object", "additionalProperties": False, "required": ["kind", "args", "text"],
                  "properties": {"kind": {"type": "string", "enum": ALL_QUERY_KINDS},
                                 "args": _STR_LIST, "text": {"type": "string"}}},
        "row": {"type": "boolean"},
        "existential_import": {"type": "boolean"},
        "initial": _STR_LIST,
        "goal": _STR_LIST,
        "steps": _STR_LIST,
        "unparsed": _STR_LIST,
        "parse_confidence": {"type": "number"},
    },
}

ENVISION_SYSTEM = """You are the perception stage of a reasoning system. You turn a problem statement \
into a structured scene that a separate engine will imagine and reason over. Do not solve the problem; \
represent it faithfully so the engine can.

Pick one domain:
- spatial: objects placed relative to each other. Kinds (args are entity ids): left_of(a,b), right_of(a,b), \
above(a,b), below(a,b), directly_above/directly_below(a,b) (same column), directly_left_of/directly_right_of(a,b) \
(same row), same_column(a,b), same_row(a,b). For things in a single row of distinct slots (shelf, queue, seats) \
set row=true; then you may also use immediately_left_of(a,b), immediately_right_of(a,b), next_to(a,b), \
between(a,b,c) meaning a is between b and c. Query kinds: any of these, or relative(a,b) for "where is a \
relative to b".
- categorical: statements about categories. Kinds: all(x,y), some(x,y), no(x,y), some_not(x,y). Entities are \
the categories (plural labels, e.g. "artists"). Query kinds: the same four. Use query kind "none" for \
"what follows?". Set existential_import=true unless the text says a category may be empty.
- process: an ordered procedure. Entities are steps and facts. Relations: needs(step,fact), makes(step,fact), \
ends(step,fact) (a fact the step makes false). steps = step ids in the stated order; initial = facts true at \
the start; goal = facts that must hold at the end. Query kind "achieves" with no args.

Rules: ids are short lowercase snake_case and every id used in relations must appear in entities (for \
process, facts may be listed as entities too). Give each relation the sentence it came from as source. \
Represent every premise. If something cannot be expressed in this vocabulary, put the sentence in unparsed \
rather than approximating it. Fill unused fields with empty values (row=false, empty lists). \
parse_confidence is your honest probability that the scene captures the problem exactly."""

INTUIT_SYSTEM = """Answer the question in the problem immediately, from first impression, without working \
it out. Reply with one of: yes, no, undetermined (the premises allow both), contradiction (the premises \
conflict), works / fails (for a procedure). Rate your confidence honestly."""

INTUIT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answer", "confidence"],
    "properties": {"answer": {"type": "string", "enum": ["yes", "no", "undetermined", "contradiction",
                                                         "works", "fails"]},
                   "confidence": {"type": "number"}},
}

VERIFY_SYSTEM = """You check a perception step. Given an original problem and the structured premises \
extracted from it, list every discrepancy: premises that were missed, added, reversed, or that change \
meaning. Be strict about direction (left/right, above/below, all/some). If it is faithful, say so."""

VERIFY_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["faithful", "issues"],
    "properties": {"faithful": {"type": "boolean"}, "issues": _STR_LIST},
}

POLISH_SYSTEM = """You rewrite an explanation for a specific listener. Keep every fact, the verdict, and \
any diagrams exactly as given (copy indented diagram lines verbatim). Do not add claims, reasons or \
examples that are not in the original. Express certainty exactly at the level stated: use the phrase \
"{hedge}" and no stronger certainty words. Listener: {expertise}, prefers {prefers} \
(picture = lead with the diagram; words = prose, no diagram; steps = numbered walk-through). Output only \
the rewritten explanation."""


class BackendRefusal(RuntimeError):
    pass


class ClaudeBackend:
    name = "claude"

    def __init__(self, model: str = MODEL, client: Any = None, polish: bool = True):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.do_polish = client, model, polish

    def _call(self, system: str, user: str, schema: dict | None, effort: str) -> str:
        output_config: dict[str, Any] = {"effort": effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
            betas=[FALLBACK_BETA],
            fallbacks="default",  # re-run a policy decline on Anthropic's recommended fallback model
        )
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise BackendRefusal(f"request declined ({getattr(details, 'category', None)})")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("response truncated at max_tokens")
        return "".join(b.text for b in response.content if b.type == "text")

    # ENVISION: language -> scene. Parsing accuracy is where errors live, so think hard.
    def envision(self, text: str) -> Scene:
        data = json.loads(self._call(ENVISION_SYSTEM, text, SCENE_SCHEMA, effort="high"))
        scene = scene_from_dict(data)
        scene.extras["source_text"] = text
        return scene

    # A fast verbal first impression: an independent channel built without imagery.
    def intuit(self, text: str) -> tuple[str, float]:
        data = json.loads(self._call(INTUIT_SYSTEM, text, INTUIT_SCHEMA, effort="low"))
        return data["answer"], float(data["confidence"])

    # Perception re-check, triggered when the verbal channel disagrees with the model.
    def verify(self, text: str, scene: Scene) -> tuple[bool, list[str]]:
        user = f"Original problem:\n{text}\n\nExtracted scene:\n{scene.describe()}"
        data = json.loads(self._call(VERIFY_SYSTEM, user, VERIFY_SCHEMA, effort="medium"))
        return bool(data["faithful"]), list(data["issues"])

    # TRANSLATE: adapt the explanation's voice; the loop rejects unearned certainty.
    def polish(self, explanation: str, listener: ListenerModel, confidence: float) -> str | None:
        if not self.do_polish:
            return None
        system = POLISH_SYSTEM.format(hedge=hedge(confidence), expertise=listener.expertise,
                                      prefers=listener.prefers)
        return self._call(system, explanation, None, effort="medium").strip() or None
