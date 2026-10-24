from pathlib import Path
from types import SimpleNamespace

from rumba_bench.data import ROOT, read_json
from rumba_bench.episodic import citation_repair, fact_documents, fact_prompt, parse_facts, parse_repaired, windows
from rumba_bench.pipeline import chunks, extract_memory_window


def sources():
    return [{"message_id": 0, "role": "user", "text": "Я купила красный самокат."},
            {"message_id": 1, "role": "assistant", "text": "Рекомендую купить шлем."},
            {"message_id": 2, "role": "user", "text": "Хочу купить шлем, но пока не купила."}]


def fact(statement, speaker, status, mid, quote):
    return {"statement": statement, "speaker": speaker, "status": status,
            "evidence": [{"message_id": mid, "quote": quote}]}


def test_evidence_must_match_visible_source_and_speaker():
    import json
    good = fact("Пользователь купила самокат", "user", "asserted", 0, "Я купила красный самокат.")
    recommendation = fact("Ассистент советовал шлем", "assistant", "recommendation", 1, "Рекомендую купить шлем.")
    bad_quote = fact("Пользователь купила шлем", "user", "asserted", 2, "Я купила шлем.")
    wrong_role = fact("Пользователь купила шлем", "user", "asserted", 1, "Рекомендую купить шлем.")
    missing_source = fact("Пользователь купила шлем", "user", "asserted", 999, "шлем")
    accepted, rejected = parse_facts(json.dumps({"facts": [good, recommendation, bad_quote, wrong_role, missing_source]}), sources())
    assert len(accepted) == 2
    assert {r["reason"] for r in rejected} == {"quote_not_in_visible_source", "speaker_not_supported_by_evidence"}


def test_plan_and_exact_source_are_preserved_in_memory():
    import json
    plan = fact("Пользователь хочет купить шлем, но ещё не купила", "user", "plan", 2,
                "Хочу купить шлем, но пока не купила.")
    facts, _ = parse_facts(json.dumps({"facts": [plan, plan]}), sources())
    records = [{"session_idx": 1, "timestamp": "2025-01-01", "chunk": 0, "facts": facts,
                "entities": [{"entity_name": "Шлем", "entity_type": "PRODUCT", "description": "BAD_DICTIONARY_DEFINITION"}],
                "relations": []}]
    docs = fact_documents("user_1", records)
    assert len(docs) == 1
    assert "plan" in docs[0]["text"] and "пока не купила" in docs[0]["text"]
    assert "BAD_DICTIONARY_DEFINITION" not in docs[0]["text"]
    assert docs[0]["turn_ids"] == [2]


def test_windows_keep_every_turn_role_and_bound_long_messages():
    session = {"conversation": [{"role": "user", "message": "Начало " + "слово " * 300},
                                {"role": "assistant", "message": "Какой цвет?"},
                                {"role": "user", "message": "Красный."}]}
    result = windows(session, chunks, max_chars=700)
    assert len(result) > 1
    assert {s["message_id"] for window in result for s in window} == {0, 1, 2}
    assert all(sum(len(s["text"]) + 60 for s in w) <= 700 for w in result)
    assert any(s["text"] == "Красный." for w in result for s in w)


def test_extraction_prompt_has_no_question_gold_or_evidence_annotations():
    session = {"session_datetime": "2025-01-01", "question": "SECRET_QUESTION", "answer": "SECRET_GOLD", "evidence": [12345]}
    prompt = fact_prompt("user_1", session, sources())
    assert all(secret not in str(prompt) for secret in ("SECRET_QUESTION", "SECRET_GOLD", "12345"))


def test_failed_nerel_annotation_does_not_erase_verified_fact(tmp_path):
    import json
    config = read_json(ROOT / "config.example.json")
    config["protocol"]["extraction_decoding"] = "structured"
    config["protocol"]["repetition_penalty"] = 1.03
    class Fake:
        count = 0
        def call(self, messages, *args, **kwargs):
            self.count += 1
            if self.count == 2:
                raise ValueError("NEREL output truncated")
            return {"text": json.dumps({"facts": [fact("Пользователь купила самокат", "user", "asserted", 0,
                                                     "Я купила красный самокат.")]}), "seconds": 0, "usage": {}}
    record = extract_memory_window("user_1", {"session_idx": 1, "session_datetime": "2025-01-01"}, 0,
                                   sources(), config, Fake(), tmp_path)
    assert len(record["facts"]) == 1 and "nerel_error" in record and "error" not in record
    assert len(fact_documents("user_1", [record])) == 1


def test_concurrent_cache_writes_are_atomic(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from rumba_bench.data import write_json
    target = tmp_path / "same-request.json"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: write_json(target, {"value": i}), range(32)))
    assert read_json(target)["value"] in range(32)
    assert not list(tmp_path.glob("*.tmp"))


def test_citation_repair_keeps_claim_and_requires_literal_evidence():
    import json
    proposed = fact("Пользователь купила самокат", "user", "asserted", 0, "Склеенная неверная цитата")
    candidates, _, schema = citation_repair([{"fact": proposed}], sources(), {})
    valid = {**proposed, "candidate_id": 0, "evidence": [{"message_id": 0, "quote": "Я купила красный самокат."}]}
    assert len(parse_repaired(json.dumps({"facts": [valid]}), candidates, sources())) == 1
    changed_claim = {**valid, "statement": "Пользователь купила шлем"}
    assert not parse_repaired(json.dumps({"facts": [changed_claim]}), candidates, sources())
    invented_quote = {**valid, "evidence": [{"message_id": 0, "quote": "Я купила шлем."}]}
    assert not parse_repaired(json.dumps({"facts": [invented_quote]}), candidates, sources())
    wrong_speaker = {**valid, "evidence": [{"message_id": 1, "quote": "Рекомендую купить шлем."}]}
    assert not parse_repaired(json.dumps({"facts": [wrong_speaker]}), candidates, sources())
