"""Prompt builders for the RAGU-style extraction GRPO line (unlabeled corpus).

Построители промптов для линии GRPO-extraction в стиле RAGU (корпус без разметки).

Three task stages, mirroring RAGU's extractors
(:class:`~ragu.triplet.llm_artifact_extractor.ArtifactsExtractorLLM` one-stage
and :class:`~ragu.triplet.two_stage_extractor.TwoStageArtifactsExtractorLLM`
two-stage) / Три стадии задач, зеркалящие экстракторы RAGU:

* ``one_stage`` — entities + relations in a single JSON response
  (system: the vendored ``DEFAULT_ARTIFACTS_EXTRACTOR_SYSTEM``);
  entities+relations одним JSON-ответом;
* ``stage1`` — entity extraction only (system: the vendored
  ``DEFAULT_TWO_STAGE_ENTITIES_EXTRACTOR_SYSTEM``); только сущности;
* ``stage2`` — relation extraction conditioned on the entity list that
  stage 1 produced (and that passed the deterministic verification). The
  system template here is NEW: the vendored relation templates were
  deliberately simplified to drop every reference to a "provided list of
  entities" (see the deviation note in ``_ragu_templates`` — the ICL
  formatters never emit the list). Our stage-2 builder DOES put the entity
  list into the user message, so the entity-list language is restored in a
  fresh template instead of editing the vendored one. Only relations BETWEEN
  THE PROVIDED ENTITIES are requested; relation extraction conditioned on the
  list of entities from stage 1 that passed deterministic verification
  (the list is included in the user prompt, the template is written anew —
  restoring the entity-list language dropped during vendoring).

Two deviations from upstream, both deliberate / Два deliberate-отклонения от
upstream:

1. **The JSON output schema is spelled out inline** in every system prompt
   (``Output JSON schema:`` block). Upstream relies on server-side
   ``output_schema`` structured decoding; the policy must learn the envelope
   from the prompt alone (the same decision as ``docsem_quant``'s champion
   PoT prompt).
2. **A compact prompt variant per stage** (``variant=1``) for instruction
   variability — the corpus generator alternates formulations so GRPO does
   not overfit one phrasing.
"""

from __future__ import annotations

from typing import Any

from jinja2 import Environment, StrictUndefined

from llm_training.extraction.ontology import ExtractionSchema, get_ontology
from llm_training.loaders._ragu_templates import (
    DEFAULT_ARTIFACTS_EXTRACTOR_SYSTEM,
    DEFAULT_TWO_STAGE_ENTITIES_EXTRACTOR_SYSTEM,
)

# Jinja2 environment matching RAGU's settings (StrictUndefined, trim/lstrip)
# — the same as the vendored template module.
# Jinja2-окружение как в RAGU (StrictUndefined, trim/lstrip).
_ENV = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)

#: Stage identifiers used across the corpus generator, the reward and the
#: anchors. / Идентификаторы стадий (генератор корпуса, награда, якоря).
STAGE_ONE = "one_stage"
STAGE_ENTITIES = "stage1"
STAGE_RELATIONS = "stage2"
STAGES: tuple[str, ...] = (STAGE_ONE, STAGE_ENTITIES, STAGE_RELATIONS)

#: Target-language prompt values. / Значения целевого языка в промпте.
LANGUAGE_VALUES: dict[str, str] = {
    "ru": "Russian",
    "en": "English",
}

# The NEW stage-2 relation template (entity-list language restored — see the
# module docstring). / НОВЫЙ шаблон стадии 2 (язык про список сущностей
# возвращён — см. докстринг модуля).
STAGE2_RELATIONS_SYSTEM = """
You are an expert relation extraction system.
You are given a source text and the list of entities that were previously
extracted from it. Your task is to extract all meaningful relationships
BETWEEN THE PROVIDED ENTITIES only.

## TASK
Determine ALL pairs (**source_entity**, **target_entity**) of entities from
the provided list that are *explicitly or clearly implicitly connected* in
the text.

For each relationship, provide:
1. **source_entity** — The name of the source entity. MUST be one of the
   provided entities (use the name exactly as given in the list).
2. **target_entity** — The name of the target entity. MUST be one of the
   provided entities (use the name exactly as given in the list).
3. **relation_type** — A concise, descriptive label for the type of relationship.
4. **description** — A detailed, context-rich description of the relationship *as it is expressed in the text*. Include specifics: roles, actions, conditions, temporal aspects, causality, and any qualifiers mentioned.
5. **relationship_strength** — An integer from 1 to 5 indicating how strong/direct/significant the connection is.

## RULES
- **Only the provided entities**: the `source_entity` and `target_entity`
  values MUST be taken from the provided entity list. Do NOT introduce new
  entities, rename the provided ones, or use aliases not present in the list.
- **Exhaustive extraction**: identify ALL relationships between the provided
  entities present in the text. Do not stop at obvious ones — also capture
  indirect, causal, temporal, spatial, and hierarchical relationships.
- **Directionality matters**: choose source and target to reflect the natural
  direction of the relationship (source = the agent/subject, target = the
  object/recipient). For symmetric relationships, pick a consistent direction
  but do NOT duplicate the pair in reverse.
- **Multiple relations per pair**: the SAME entity pair CAN have MULTIPLE
  different relation types. Extract each distinct relationship as a separate entry.
- **Grounded descriptions**: every claim in the description MUST be supported
  by the source text. Do not hallucinate or infer beyond what is stated or
  clearly implied.
{% if relation_types %}
- You MUST use one of the following predefined relation types: {{ relation_types }}
- Every extracted relation MUST be assigned one of these types.
{% else %}
- Since no predefined relation types are provided, infer appropriate, concise,
  and consistent UPPER_SNAKE_CASE labels (e.g., WORKS_FOR, LOCATED_IN,
  PRODUCES, PART_OF, CAUSES).
{% endif %}

Provide the answer in the following language: {{ language }}
Return the result as valid JSON matching the provided schema.
"""

# Compact variants (instruction variability, variant=1).
# Компактные варианты (вариативность формулировок, variant=1).
COMPACT_ONE_STAGE_SYSTEM = """
Extract named entities and relations from the given text.
Entity types allowed: {{ entity_types }}.
Relation types allowed: {{ relation_types }}.
For each entity give: entity_name (normalized: base form, proper
capitalization), entity_type, description (grounded in the text).
For each relation give: source_entity, target_entity (both must be extracted
entities), relation_type, description, relationship_strength (integer 0-5).
Answer in {{ language }}. Return ONLY valid JSON matching the schema.
"""

COMPACT_STAGE1_SYSTEM = """
Extract every meaningful entity from the given text.
Entity types allowed: {{ entity_types }}.
For each entity give: entity_name (the normalized canonical name: base form,
coreferences resolved, no pronouns), entity_type, description (detailed,
grounded in the text). Only entities supported by the text — no hallucination.
Answer in {{ language }}. Return ONLY valid JSON matching the schema.
"""

COMPACT_STAGE2_SYSTEM = """
You are given a text and a list of entities already extracted from it.
Find all relationships between the PROVIDED entities that the text states or
clearly implies. Relation types allowed: {{ relation_types }}.
For each relation give: source_entity, target_entity (exactly as in the
provided list), relation_type, description (grounded in the text),
relationship_strength (integer 1-5).
Answer in {{ language }}. Return ONLY valid JSON matching the schema.
"""

#: Inline JSON schemas per stage (the envelope the policy must learn from the
#: prompt — upstream relies on server-side structured decoding, we cannot).
#: JSON-схемы по стадиям, вписанные в промпт.
_JSON_SCHEMAS: dict[str, str] = {
    STAGE_ONE: (
        '{"entities": [{"entity_name": str, "entity_type": str, '
        '"description": str}], "relations": [{"source_entity": str, '
        '"target_entity": str, "relation_type": str, "description": str, '
        '"relationship_strength": int}]}'
    ),
    STAGE_ENTITIES: (
        '{"entities": [{"entity_name": str, "entity_type": str, '
        '"description": str}]}'
    ),
    STAGE_RELATIONS: (
        '{"relations": [{"source_entity": str, "target_entity": str, '
        '"relation_type": str, "description": str, '
        '"relationship_strength": int}]}'
    ),
}

#: Public alias of the inline envelope schemas (the agent-suite builder and
#: other consumers outside this module read THEM, not private names).
#: Публичный алиас инлайновых схем (построитель набора и другие потребители
#: вне модуля читают его, а не приватное имя).
JSON_SCHEMAS = _JSON_SCHEMAS


def render_system(
    stage: str,
    schema: ExtractionSchema,
    language: str,
    variant: int = 0,
    with_signatures: bool = False,
) -> str:
    """Render the system message of one extraction stage.

    Рендерит системное сообщение стадии извлечения.

    Args:
        stage: one of :data:`STAGES` (``one_stage`` / ``stage1`` / ``stage2``).
        schema: the ontology subset rendered into the type lists.
        language: prompt language value — ``"Russian"`` / ``"English"`` /
            or a free-form instruction (e.g. "the same language as the
            source text").
        variant: ``0`` = the RAGU-verbatim (or new stage-2) template,
            ``1`` = the compact variability variant.
        with_signatures: render each predicate with its
            ``[DOMAIN -> RANGE]`` signature (RAGU's ``show_type_signatures``;
            requires ``schema.signatures`` to be populated).
    """
    if stage not in STAGES:
        raise ValueError(
            f"Unknown extraction stage {stage!r}; must be one of {STAGES}."
            f"\nНеизвестная стадия извлечения {stage!r}."
        )
    entity_types = get_ontology(schema.ontology).render_entity_types(
        list(schema.entity_types)
    )
    relation_types = get_ontology(schema.ontology).render_relation_types(
        list(schema.relation_types), with_signatures=with_signatures
    )
    if stage == STAGE_ONE:
        template = (
            DEFAULT_ARTIFACTS_EXTRACTOR_SYSTEM if variant == 0
            else COMPACT_ONE_STAGE_SYSTEM
        )
    elif stage == STAGE_ENTITIES:
        template = (
            DEFAULT_TWO_STAGE_ENTITIES_EXTRACTOR_SYSTEM if variant == 0
            else COMPACT_STAGE1_SYSTEM
        )
    else:
        template = (
            STAGE2_RELATIONS_SYSTEM if variant == 0 else COMPACT_STAGE2_SYSTEM
        )
    rendered = _ENV.from_string(template).render(
        language=language,
        entity_types=entity_types,
        relation_types=relation_types,
    )
    schema_block = (
        f"\nOutput JSON schema (return exactly this envelope):\n"
        f"{_JSON_SCHEMAS[stage]}\nReturn ONLY the JSON object, "
        f"no markdown fences, no extra text."
    )
    return rendered.rstrip() + "\n" + schema_block


def _format_entity_list(entities: list[dict[str, Any]]) -> str:
    lines = []
    for e in entities:
        name = str(e.get("entity_name", "")).strip()
        etype = str(e.get("entity_type", "")).strip()
        if not name:
            continue
        lines.append(f"- {name} ({etype})" if etype else f"- {name}")
    return "\n".join(lines) if lines else "(none)"


def build_user_message(
    stage: str,
    text: str,
    provided_entities: list[dict[str, Any]] | None = None,
) -> str:
    """Build the user message of one extraction stage.

    Строит пользовательское сообщение стадии. One-stage and stage 1 carry only
    the source text (RAGU's ``Text:\\n{...}`` layout); stage 2 prepends the
    verified entity list from stage 1 (the list stage 2 is conditioned on).
    """
    if stage == STAGE_RELATIONS:
        if provided_entities is None:
            raise ValueError(
                "stage2 requires provided_entities (the verified stage-1 list)."
                "\nstage2 требует provided_entities (проверенный список стадии 1)."
            )
        block = _format_entity_list(provided_entities)
        return f"Entities extracted from the text:\n{block}\n\nText:\n{text}"
    return f"Text:\n{text}"


def build_extraction_prompt(
    stage: str,
    text: str,
    schema: ExtractionSchema,
    language: str = "Russian",
    variant: int = 0,
    with_signatures: bool = False,
    provided_entities: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the full message-list prompt of one extraction stage.

    Строит полный промпт-список сообщений стадии извлечения.

    Returns ``[{role: "system", content: ...}, {role: "user", content: ...}]``
    — the Meno-Lite chat format without the assistant turn (ready for the
    chat-template rendering used across the training/eval paths).
    """
    system = render_system(stage, schema, language, variant, with_signatures)
    user = build_user_message(stage, text, provided_entities)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
