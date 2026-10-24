import ast
from pathlib import Path
from types import SimpleNamespace
import json

import nltk
import numpy as np
import pytest

from rumba_bench.data import ROOT, category, digest, pilot_ids, read_json
from rumba_bench.pipeline import (Generation, answer_question, chunks, parse_extraction,
                                  training_modules)
from rumba_bench.upstream import f1_score, load_official


@pytest.fixture
def config():
    return read_json(ROOT / "config.example.json")


def test_official_rumba_parser_and_date_prompt():
    official = load_official()
    for verdict, expected in [(0, 0), (1, 0), (2, 1), (3, 1), (4, 0)]:
        text = f"[FEEDBACK] explanation [RESULT] {verdict} [END]"
        assert official["process_judge_response"](text) == expected
    assert official["process_judge_response"]("malformed") == 0
    prompt = official["judge_template"]("Q (текущая дата: 2020-01-01)", "A", "G")
    assert "2020-01-01" in prompt[0]["content"]


def test_f1_matches_actual_upstream_method():
    tree = ast.parse((ROOT / "vendor/lighteval/f1_source.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_one_item")
    namespace = {"nltk": nltk}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "official-f1", "exec"), namespace)
    scorer = SimpleNamespace(normalize_gold=None, normalize_pred=None)
    for gold, pred in [("Да.", "да"), ("кот кот пёс", "кот"), ("", ""), ("a b", "b c"), ("Да", "Да")]:
        assert f1_score(gold, pred) == pytest.approx(namespace["compute_one_item"](scorer, gold, pred))


def test_pilot_is_fixed_and_covers_four_cells():
    questions = [{"id": f"{u}-{i}", "user_id": str(u),
                  "category": f"StaticUser, {['single','multi'][i%2]}, {['temporal','atemporal'][(i//2)%2]}"}
                 for u in range(6) for i in range(24)]
    ids = pilot_ids(questions)
    assert ids == pilot_ids(list(reversed(questions)))
    assert len(set(ids)) == 50
    picked = [q for q in questions if q["id"] in ids]
    assert len({category(q)[1:] for q in picked}) == 4
    assert len({q["user_id"] for q in picked}) == 3


def test_chunks_preserve_source_and_terminate():
    text = "a longer sentence " * 100
    parts = list(chunks(text, 200, 20))
    assert parts[0].startswith(text[:10]) and parts[-1].endswith(text[-10:])
    assert all(0 < len(part) <= 200 for part in parts)
    with pytest.raises(ValueError):
        list(chunks(text, 10, 10))


def test_extraction_reports_invalid_types_and_dangling_relations(config):
    ontology, *_ = training_modules(config)
    result = {"entities": [
        {"entity_name": "Анна", "entity_type": "PERSON", "description": "Пользователь Анна"},
        {"entity_name": "X", "entity_type": "UNKNOWN", "description": "bad"}],
        "relations": [{"source_entity": "Анна", "target_entity": "X", "relation_type": "WORKPLACE",
                       "description": "bad", "relationship_strength": 3}]}
    parsed = parse_extraction(json.dumps(result), ontology)
    assert len(parsed["entities"]) == 1
    assert not parsed["relations"]
    assert len(parsed["violations"]) == 2
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_extraction('{"entities": []}', ontology)


def test_retrieval_uses_same_hybrid_index_and_is_user_local(config):
    _, _, HybridIndex, _ = training_modules(config)
    vectors = {"кот дома": [1, 0], "работа библиотека": [0, 1], "где кот": [1, 0]}
    def encode(texts):
        return np.array([vectors[t] for t in texts])
    index = HybridIndex.build(["кот дома", "работа библиотека"], ["user-A/1", "user-A/2"], encode)
    assert index.search("где кот", 1)[0].doc_id == "user-A/1"
    assert all("user-B" not in hit.doc_id for hit in index.search("где кот", 20))


def test_answers_do_not_send_gold_evidence_or_categories(config):
    class Capture:
        def call(self, messages, *args, **kwargs):
            self.messages = messages
            return {"text": "Нет такой информации", "seconds": 0, "usage": {}}
    generator = Capture()
    row = {"id": "test", "question": "Где я живу?", "question_date": "2022-01-01",
           "answer": "GOLD_SECRET", "evidence": [98765], "category": "CATEGORY_SECRET"}
    result = answer_question(row, [], None, config, generator, load_official())
    prompt = json.dumps(generator.messages, ensure_ascii=False)
    assert all(secret not in prompt for secret in ("GOLD_SECRET", "98765", "CATEGORY_SECRET"))
    assert result["retrieval_recall"] == 0


def test_generation_cache_and_truncation(tmp_path, config):
    endpoint = config["models"]["meno-lite-0.2"]
    generator = Generation(endpoint, config["runtime"], tmp_path)
    class Fake:
        calls = 0
        def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"), finish_reason="stop")],
                                   usage=None, model="meno-lite-0.2", system_fingerprint="test")
    fake = Fake()
    generator.client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    messages = [{"role": "user", "content": "test"}]
    assert generator.call(messages, 32)["text"] == "answer"
    generator.call(messages, 32)
    assert fake.calls == 1
    generator.call(messages, 64)
    assert fake.calls == 2
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    generator.call(messages, 32, json_schema=schema)
    assert fake.calls == 3
    records = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert sum("structured_outputs" in r["request"]["extra_body"] for r in records) == 1
    cache = next(p for p in tmp_path.glob("*.json")
                 if read_json(p)["request"]["max_tokens"] == 32
                 and "structured_outputs" not in read_json(p)["request"]["extra_body"])
    record = json.loads(cache.read_text())
    record["finish_reason"] = "length"
    cache.write_text(json.dumps(record))
    tokens = record["request"]["max_tokens"]
    with pytest.raises(ValueError, match="incomplete"):
        generator.call(messages, tokens)
    assert generator.call(messages, tokens, allow_incomplete=True)["finish_reason"] == "length"


def test_penalty_and_temperature_are_explicit_and_isolated():
    from rumba_bench.pipeline import generation_parameters
    base = generation_parameters()
    trial = generation_parameters({"temperature": 0.2, "top_p": 0.95, "repetition_penalty": 1.1})
    assert trial["temperature"] == 0.2 and trial["top_p"] == 0.95
    assert trial["extra_body"]["repetition_penalty"] == 1.1
    assert base["extra_body"]["repetition_penalty"] == 1
    for settings in ({"repetition_penalty": 0}, {"temperature": -1}, {"temperature": float('nan')}, {"top_p": 2}):
        with pytest.raises(ValueError):
            generation_parameters(settings)


def test_calibration_selection_protects_required_repetition():
    from scripts.tune_decoding import choose_penalty
    def row(penalty, valid, qa, recall):
        return {"sampling": {"repetition_penalty": penalty},
                "regression": {"json_valid": valid},
                "dev": {"qa_correct": qa, "entity_recall": recall},
                "overall": {"duplicate_fraction": 0, "ontology_violation_fraction": 0}}
    choices = [row(1, .5, 1, 1), row(1.03, 1, 1, 1), row(1.1, 1, .5, 1)]
    assert choose_penalty(choices)["sampling"]["repetition_penalty"] == 1.03


def test_calibration_preserves_numeric_signs():
    from scripts.tune_decoding import assess
    task = {"kind": "qa", "expected": "56"}
    assert not assess(task, {"text": "-56", "finish_reason": "stop"}, None)["qa_correct"]
    assert assess(task, {"text": "56.", "finish_reason": "stop"}, None)["qa_correct"]


def test_schema_has_ontology_enums_and_configurable_bounds(config):
    from rumba_bench.schemas import extraction_schema
    ontology, *_ = training_modules(config)
    assert extraction_schema(ontology, {"extraction_decoding": "free"}) is None
    schema = extraction_schema(ontology, {"extraction_decoding": "structured", "structured_limits": {"entities": 8}})
    assert schema["properties"]["entities"]["maxItems"] == 8
    entity = schema["properties"]["entities"]["items"]["properties"]
    assert "PERSON" in entity["entity_type"]["enum"]
    assert "UNKNOWN" not in entity["entity_type"]["enum"]
    assert schema["additionalProperties"] is False
    with pytest.raises(ValueError):
        extraction_schema(ontology, {"extraction_decoding": "structured", "structured_limits": {"entities": -1}})


def test_both_modes_keep_separate_extraction_caches(tmp_path, config, monkeypatch):
    from copy import deepcopy
    from rumba_bench import cli
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "protocol_manifest", lambda c: {
        "training_source_hashes": {"prompts.py": "same-prompts"},
        "runner_hashes": {"pipeline.py": "same-pipeline", "schemas.py": "same-schema"},
        "dataset": {"histories_sha256": "same-data"}})
    free = cli.extraction_paths(config, "meno-lite-0.2")
    constrained = deepcopy(config)
    constrained["protocol"]["extraction_decoding"] = "structured"
    structured = cli.extraction_paths(constrained, "meno-lite-0.2")
    assert free != structured
    assert free.parent.name.startswith("free-")
    assert structured.parent.name.startswith("structured-")
    constrained["protocol"]["structured_limits"]["entities"] = 8
    assert cli.extraction_paths(constrained, "meno-lite-0.2") != structured


def test_bundled_training_sources_match_their_manifest(config):
    import hashlib
    folder = ROOT / config["training_source"]
    manifest = read_json(folder / "manifest.json")
    for name, info in manifest["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == info["sha256"]


def test_stages_work_when_generator_embedder_and_judge_are_swapped(tmp_path, config, monkeypatch):
    from rumba_bench import cli
    from rumba_bench.data import write_json
    model = "meno-lite-0.2"
    root, folder, extracted = tmp_path / "run", tmp_path / "run" / model, tmp_path / "extraction"
    question = {"id": "1|q0", "user_id": "user_1", "question": "Где я живу?", "answer": "Москва.",
                "question_date": "2022-01-02", "category": "StaticUser, single, atemporal", "evidence": [1]}
    write_json(extracted / "memories/user_1.json", {"documents": [
        {"id": "user_1/1/0/entities/0", "session_idx": 1, "timestamp": "2022-01-01",
         "text": '{"entity_name":"Москва","entity_type":"CITY","description":"Пользователь живёт в Москве."}'}]})
    write_json(extracted / "qc-pilot.json", {"questions": 1, "chunks": 1, "extraction_errors": 0,
                                            "ontology_violations": 0, "empty_extractions": 0})
    monkeypatch.setattr(cli, "job_paths", lambda c, m: (root, folder))
    monkeypatch.setattr(cli, "extraction_paths", lambda c, m: extracted)
    monkeypatch.setattr(cli, "selected_questions", lambda suite: [question])

    class FakeEncoder:
        def encode(self, texts, query=False):
            return np.ones((len(texts), 2), dtype=np.float32)
        def __call__(self, texts):
            return self.encode(texts, query=True)
    monkeypatch.setattr(cli, "Encoder", lambda c, r: FakeEncoder())
    cli.retrieve(config, model, "pilot")

    def embedding_is_off(*args):
        raise AssertionError("Embedding server is offline during answer and judge stages")
    monkeypatch.setattr(cli, "Encoder", embedding_is_off)

    class FakeGeneration:
        def __init__(self, endpoint, runtime, cache, sampling=None):
            self.model = endpoint["model"]
            self.client = SimpleNamespace(models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=model)])))
        def call(self, messages, max_tokens, **kwargs):
            text = "[FEEDBACK] Совпадает. [RESULT] 2 [END]" if "pollux" in self.model else "Москва."
            return {"text": text, "seconds": 0, "usage": {}}
    monkeypatch.setattr(cli, "Generation", FakeGeneration)
    cli.answer(config, model, "pilot")
    assert read_json(folder / "pilot/report.json")["status"] == "judge_pending"
    config["judge"] = {"base_url": "http://offline.invalid/v1", "model": "pollux",
                       "source_model": "ai-forever/pollux-judge-32b"}
    cli.score(config, model, "pilot")
    report = read_json(folder / "pilot/report.json")
    assert report["overall"]["f1"] == report["overall"]["rumba_metric"] == 1
    assert report["overall"]["retrieval_recall"] == 1
    assert report["status"] == "scored"
