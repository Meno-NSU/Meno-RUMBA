"""Unified dataset schema — TRL-compatible example with task-class metadata.

Схема унифицированного набора данных: TRL-совместимый пример с метаданными класса задачи.

The unified dataset is consumed by TRL's ``GRPOTrainer``. A row must carry a
conversational ``prompt`` column (``list[{role, content}]``); every extra column
is preserved by ``GRPOConfig`` (``remove_unused_columns=False`` by default) and
forwarded as a keyword argument to the reward function — so ``task_class`` and
``reference`` reach the multi-task reward dispatcher automatically.

Унифицированный набор данных потребляется ``GRPOTrainer`` из TRL. Строка обязана нести
колонку ``prompt`` в разговорном формате (``list[{role, content}]``); все доп.
колонки сохраняются и пробрасываются в reward-функцию как kwargs, поэтому
``task_class`` и ``reference`` автоматически доходят до диспетчера наград.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A chat message in the Qwen2.5 / Meno-Lite chat format.
# Сообщение чата в формате Qwen2.5 / Meno-Lite.
Message = dict[str, Any]


# Canonical list of all 25 task classes. The reward function branches on this
# string and ``grpo_train.py``'s ``--val-task`` / ``--test-task`` choose hold-outs
# among them.
# Канонический список всех 25 классов задач. По этой строке ветвится reward, а
# аргументы ``--val-task`` / ``--test-task`` скрипта grpo_train.py выбирают hold-out.
TASK_CLASSES: tuple[str, ...] = (
    # --- ShareGPT catalog (9) / Каталог ShareGPT ---
    "coder",
    "creative_writer",
    "data_analyst",
    "deep_research",
    "factual_qa",
    "instruction_following",
    "math_reasoner",
    "rewriter",
    "summarizer",
    # --- Menon QA (1) ---
    "menon_qa",
    # --- mt-RAGEval (1, 8 sub-types merged) / 8 подтипов слиты ---
    "mtrageval",
    # --- RAGU ICL (6) ---
    "ragu_artifact_extraction",
    "ragu_artifact_validation",
    "ragu_entity_extraction",
    "ragu_entity_validation",
    "ragu_relation_extraction",
    "ragu_relation_validation",
    # --- Other sources (one class each) / Прочие источники ---
    "asr_summarization",
    "tool_calls_multiturn",
    "multihop_rag",
    "rummlu_pro",
    "self_cognition",
    "opencode_gen",
    # --- Meno-stand distillation (3; GPT-4o through the NSU "Menon") ---
    # --- Дистилляция Менон-стенда (3; GPT-4o через «Менон» НГУ) ---
    "meno_rerank",
    "meno_rewrite",
    "meno_final",
    # --- DocSem quantitative reasoning (1; distilled check-181) ---
    # --- DocSem: количественные вопросы по документу (дистилляция check-181) ---
    "docsem_quant",
    # --- NEREL_Instruct (HF runtime; 6 configs merged) / с HF, 6 конфигов слиты ---
    "nerel_instruct",
)
# NOTE (frozen contract): the GRPO-EXTRACTION line's classes (3 extraction
# stages, 3 summary kinds, fineweb_qa) deliberately do NOT join this
# registry: TASK_CLASSES gates UnifiedExample (which requires a reference —
# the extraction corpus is reference-free by design), the loaders, and
# run_eval.py's default task list (no loaders/scoring for them there).
# The extraction line dispatches by its own constants
# (llm_training.extraction.prompts.STAGES, .summary_prompts.SUMMARY_STAGES,
# .qa_prompts.QA_TASK_CLASS); RewardConfig's fluency allowlists validate
# against TASK_CLASSES ∪ EXTRACTION_LINE_TASK_CLASSES (see below).
assert len(TASK_CLASSES) == 28, f"Expected 28 task classes, got {len(TASK_CLASSES)}"

#: Extra task classes of the GRPO-EXTRACTION line (the 8-class design): the 3
#: consolidation kinds + the QA class. The 3 extraction stage classes reuse
#: existing TASK_CLASSES names (ragu_*_extraction), so they are not listed
#: here. These classes DO NOT join TASK_CLASSES (see the frozen-contract note
#: above), but RewardConfig's fluency allowlists accept them on top of
#: TASK_CLASSES (validation = TASK_CLASSES ∪ this tuple) — consolidation and
#: QA are NL tasks where the grammar-classifier signal applies.
#: Дополнительные классы задач линии GRPO-extraction (дизайн 8 классов):
#: 3 консолидации + класс вопросо-ответа.
EXTRACTION_LINE_TASK_CLASSES: tuple[str, ...] = (
    "ragu_entity_summary",
    "ragu_relation_summary",
    "ragu_community_summary",
    "fineweb_qa",
)
assert len(set(EXTRACTION_LINE_TASK_CLASSES)) == 4


@dataclass
class UnifiedExample:
    """One row of the unified dataset, ready to feed TRL ``GRPOTrainer``.

    Одна строка унифицированного набора данных, готовая для ``GRPOTrainer`` из TRL.

    Attributes / Атрибуты:
        task_class: task-class identifier (one of :data:`TASK_CLASSES`).
            Identifies the class of the task, for reward dispatch & hold-out split.
            идентификатор класса задачи — для ветвления reward и разбиения.
        prompt: conversational prompt = chat messages WITHOUT the reference
            assistant turn. TRL applies the Meno-Lite chat template to it.
            разговорный промпт = сообщения БЕЗ референсного assistant-хода;
            TRL применит к нему chat_template модели Meno-Lite.
        reference: the human reference answer (last assistant content for plain
            tasks; the full assistant-generated content for tool-use tasks).
            эталонный ответ человека (фидит reward/judge; не SFT-метка).
        tools: tool schemas for the tool-use classes (``opencode_gen``,
            ``tool_calls_multiturn``), ``None`` otherwise. Kept as structured
            metadata for the future GRPO-loop tool executor; the tools are
            baked into ``prompt``'s system message for TRL rendering.
            схемы инструментов для tool-use классов; в system они «впекаются»
            отдельно, здесь хранятся как метаданные для будущего executor'а.
        reference_messages: assistant/tool turns appended to ``prompt`` to
            reconstruct the full conversation (used for masking & rendering).
            ходы assistant/tool, дополняющие prompt до полной беседы
            (для рендера и маскирования).
        source: provenance string (file path / HF id / sub-type).
            строка происхождения примера.
        extra: optional per-source metadata (e.g. mtrageval sub-type, ragu
            quality_rating, nerel config).
            необязательные метаданные источника.
    """

    task_class: str
    prompt: list[Message]
    reference: str
    tools: list[Message] | None = None
    reference_messages: list[Message] = field(default_factory=list)
    source: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_class not in TASK_CLASSES:
            raise ValueError(
                f"Unknown task_class {self.task_class!r}; "
                f"must be one of {TASK_CLASSES}."
                f"\nНеизвестный task_class {self.task_class!r}."
            )
        if not self.reference:
            raise ValueError(
                "UnifiedExample.reference must be non-empty."
                "\nUnifiedExample.reference должен быть непустым."
            )
        if not isinstance(self.prompt, list) or not self.prompt:
            raise ValueError(
                "UnifiedExample.prompt must be a non-empty list of messages."
                "\nUnifiedExample.prompt должен быть непустым списком сообщений."
            )


def to_dataset_row(example: UnifiedExample) -> dict[str, Any]:
    """Flatten a :class:`UnifiedExample` into a ``datasets.Dataset`` row.

    Превращает :class:`UnifiedExample` в строку ``datasets.Dataset``.

    Extra columns (``task_class``, ``reference``, ``tools``, ``source``,
    ``extra``) are preserved by ``GRPOConfig`` and reach the reward function.
    """
    return {
        "prompt": list(example.prompt),
        "task_class": example.task_class,
        "reference": example.reference,
        "tools": example.tools,
        "reference_messages": list(example.reference_messages),
        "source": example.source,
        "extra": dict(example.extra),
    }
