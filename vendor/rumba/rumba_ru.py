"""
name:
MemoryBench QA (RUMBA)

abstract:
Custom QA benchmark for evaluating memory frameworks. Each row is a single QA sample.

languages:
russian

tags:
qa, memory, rag
"""
from aenum import extend_enum

from lighteval.metrics.metrics_sample import JudgeLLM
from lighteval.models.model_output import ModelResponse
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import SampleLevelMetric

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod

from evaluation.prompts import POLLUX_PROMPT, POLLUX_CRITERIA, POLLUX_CRITERIA_NAME

import numpy as np
import re
import json

LOG_PATH = "evaluation/results/judge_logs.jsonl"


def memorybench_prompt(line: dict, task_name: str = None) -> Doc:
    """
    Expected dataset columns (train split):
      - question: str
      - answer: str
      - (optional) id, conversation_idx, user_id, category, evidence, question_date
    """
    return Doc(
        id=line['id'],
        task_name=task_name,
        query=line['question'],
        choices=[line["answer"]],
        gold_index=0,
        specific={
            "user_id": line['user_id'], 
            "question_date": line["question_date"],
            "category": line["category"],
            "evidence": line["evidence"],
            "conversation": line["conversation"]
        }
    )


def judge_template(question: str, answer: str, gold: str | None = None, options=None) -> str:
    prompt = POLLUX_PROMPT.format(instruction=question,
                                reference_answer=gold,
                                answer=answer,
                                criteria_name=POLLUX_CRITERIA_NAME,
                                criteria_rubrics=POLLUX_CRITERIA)
    return [{"role": "user", "content": prompt}]

def process_judge_response(response) -> int:
    if isinstance(response, list):
        response = response[0]    
    match = re.search(r"\[RESULT\]\s*(-?\d+)\s*\[END\]", response)
    label = int(match.group(1)) if match else None
    if label in [0, 1]:
        return 0
    elif label in [2, 3]:
        return 1
    return 0


def extract_judge_feedback(text: str):
    if isinstance(text, list):
        text = text[0]
    m = re.search(r"\[FEEDBACK\]\s*(.*?)\s*(?=\[RESULT\])", text, flags=re.DOTALL)
    reasoning = m.group(1).strip() if m else None
    
    match = re.search(r"\[RESULT\]\s*(-?\d+)\s*\[END\]", text)
    verdict = int(match.group(1)) if match else None
    
    return reasoning, verdict

class JudgeLLMRUMBA(JudgeLLM):
    def __init__(self):
        super().__init__(
            judge_model_name="ai-forever/pollux-judge-32b",
            template=judge_template,
            process_judge_response=process_judge_response,
            judge_backend="transformers",
            short_judge_name="rumba-judge",
            max_tokens=4096,
        )
    def compute(self, doc: Doc, model_response: ModelResponse) -> bool:
        response = model_response.text[0]
        question_date = doc.specific['question_date']
        q_with_date = f"{doc.query} (текущая дата: {question_date.date()})"
        score, _, judge_response = self.judge.evaluate_answer(question=q_with_date, answer=response, gold=doc.choices[doc.gold_index])

        judge_reason, judge_vertict = extract_judge_feedback(judge_response)
        print(f"Q: {q_with_date} | G: {doc.choices[doc.gold_index]} | A: {response} | BV: {score} | R: {judge_reason}")
        record = {
            "question": doc.query,
            "gold": doc.choices[doc.gold_index],
            "answer": response,
            "verdict": judge_vertict,
            "binary_verdict": score,
            "judge_reason": judge_reason,
            "category": doc.specific['category']
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return score


rumba_metric = SampleLevelMetric(
    metric_name="llm-judge-accuracy",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=JudgeLLMRUMBA(),
    corpus_level_fn=np.mean
)

extend_enum(Metrics, "rumba_metric", rumba_metric)

memorybench_qa = LightevalTaskConfig(
    name="rumba",
    prompt_function=memorybench_prompt,
    hf_repo="ai-forever/RUMBA",
    hf_subset="ru",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    metrics=[Metrics.f1_score, Metrics.rumba_metric],
    stop_sequence=["\n"],
    version=0
)

TASKS_TABLE = [memorybench_qa]
