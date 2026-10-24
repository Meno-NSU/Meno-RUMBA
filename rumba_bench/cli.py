"""Command-line driver. / Управление бенчмарком из командной строки."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from .data import ROOT, canonical, digest, prepare, read_json, read_jsonl, timestamp, write_json
from .pipeline import (Encoder, Generation, chunks, extract_user, generate_answer, index_user,
                       protocol_manifest, retrieve_question, training_modules)
from .report import comparison, make_report
from .upstream import load_official
from .schemas import extraction_schema


def selected_questions(suite):
    rows = read_jsonl(ROOT / "data/questions.jsonl")
    if suite == "pilot":
        ids = set(read_json(ROOT / "data/pilot50.json"))
        rows = [row for row in rows if row["id"] in ids]
        if len(rows) != 50:
            raise ValueError("Pilot must contain exactly 50 questions")
    return rows


def job_paths(config, label):
    manifest = protocol_manifest(config)
    identifier = digest(manifest)[:16]
    mode = config["protocol"].get("extraction_decoding", "free")
    root = ROOT / "runs" / f"{mode}-{identifier}"
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "protocol.json", manifest)
    endpoint = config["models"].get(label)
    if not endpoint:
        raise ValueError(f"Configure models.{label} first")
    folder = root / label
    path = folder / "model.json"
    if path.exists() and read_json(path) != endpoint:
        raise ValueError(f"Endpoint/weights changed under {folder}; use a separate model label")
    write_json(path, endpoint)
    return root, folder


def extraction_paths(config, label):
    """Extraction survives embedding endpoint changes. / Извлечение не зависит от сервера векторов."""
    full = protocol_manifest(config)
    keys = ("seed", "temperature", "extraction_stage", "prompt_variant", "with_signatures",
            "chunk_chars", "chunk_overlap_chars", "extraction_max_tokens", "extraction_decoding", "structured_limits",
            "top_p", "repetition_penalty")
    manifest = {"protocol": {k: config["protocol"][k] for k in keys}, "model": config["models"][label],
                "sources": full["training_source_hashes"],
                "pipeline_source": full["runner_hashes"]["pipeline.py"],
                "schema_source": full["runner_hashes"]["schemas.py"],
                "dataset_histories_sha256": full["dataset"]["histories_sha256"]}
    mode = config["protocol"].get("extraction_decoding", "free")
    folder = ROOT / "runs/extraction" / f"{mode}-{digest(manifest)[:16]}" / label
    write_json(folder / "extraction-protocol.json", manifest)
    return folder


def preflight(config, label):
    endpoint = config["models"].get(label)
    if not endpoint:
        raise ValueError(f"No endpoint for {label}")
    generator = Generation(endpoint, config["runtime"], ROOT / "runs/preflight/requests", sampling=config["protocol"])
    models = generator.client.models.list().model_dump()
    if endpoint["model"] not in {item["id"] for item in models["data"]}:
        raise ValueError(f"Server does not serve {endpoint['model']}: {models}")
    response = generator.call([{"role": "user", "content": "Ответь одним словом: сколько будет два плюс два?"}], 32)
    result = {"endpoint": endpoint, "models": models, "response": response}
    write_json(ROOT / "runs/preflight" / f"{label}.json", result)
    print(canonical(result))


def estimate(config, suite):
    rows = selected_questions(suite)
    users = sorted({row["user_id"] for row in rows})
    sessions = 0
    jobs = 0
    for user in users:
        history = read_json(ROOT / "data/conversations" / f"{user}.json")
        sessions += len(history)
        for session in history:
            text = "\n".join(f"{m['role']}: {m['message']}" for m in session["conversation"])
            jobs += sum(1 for _ in chunks(text, config["protocol"]["chunk_chars"], config["protocol"]["chunk_overlap_chars"]))
    result = {"suite": suite, "questions": len(rows), "users": len(users), "sessions": sessions,
              "extraction_calls_per_model": jobs, "answer_calls_per_model": len(rows),
              "judge_calls_per_model": len(rows), "concurrency": config["runtime"]["concurrency"],
              "extraction_output_token_upper_bound_per_model": jobs * config["protocol"]["extraction_max_tokens"]}
    write_json(ROOT / "runs" / f"estimate-{suite}.json", result)
    print(canonical(result))


def smoke(config, label):
    """One real extraction request, independent of embeddings. / Одно реальное извлечение."""
    ontology, build_prompt, _, _ = training_modules(config)
    row = selected_questions("pilot")[0]
    session = read_json(ROOT / "data/conversations" / f"{row['user_id']}.json")[0]
    text = "\n".join(f"{m['role']}: {m['message']}" for m in session["conversation"])
    part = next(chunks(text, config["protocol"]["chunk_chars"], config["protocol"]["chunk_overlap_chars"]))
    source = f"User ID: {row['user_id']}\nSession: {session['session_idx']}\nTimestamp: {session['session_datetime']}\n{part}"
    prompt = build_prompt("one_stage", source, ontology.full_schema(), language="Russian",
                          variant=config["protocol"]["prompt_variant"], with_signatures=config["protocol"]["with_signatures"])
    endpoint = config["models"][label]
    generator = Generation(endpoint, config["runtime"], ROOT / "runs/smoke/requests", sampling=config["protocol"])
    response = generator.call(prompt, config["protocol"]["extraction_max_tokens"],
                              json_schema=extraction_schema(ontology, config["protocol"]))
    from .pipeline import parse_extraction
    parsed = parse_extraction(response["text"], ontology)
    mode = config["protocol"].get("extraction_decoding", "free")
    write_json(ROOT / "runs/smoke" / f"{label}-{mode}.json", {"response": response, "parsed": parsed})
    print(canonical({"seconds": response["seconds"], "usage": response["usage"],
                     "entities": len(parsed["entities"]), "relations": len(parsed["relations"]),
                     "violations": len(parsed["violations"])}))


def full_gate(config, label, suite):
    if suite != "full":
        return
    root, folder = job_paths(config, label)
    qc_path = folder / "pilot/qc.json"
    if not qc_path.exists() or not read_json(qc_path)["pipeline_complete"]:
        raise ValueError("Complete and debug pilot50 under the SAME protocol before full run")


def extract(config, label, suite):
    full_gate(config, label, suite)
    questions = selected_questions(suite)
    folder = extraction_paths(config, label)
    generator = Generation(config["models"][label], config["runtime"], folder / "requests", sampling=config["protocol"])
    available = generator.client.models.list()
    if config["models"][label]["model"] not in {m.id for m in available.data}:
        raise ValueError("Expected served model not present")
    all_records = []
    for user in sorted({q["user_id"] for q in questions}):
        records, documents = extract_user(user, config, generator, folder)
        all_records.extend(records)
    qc = {"questions": len(questions), "chunks": len(all_records),
          "extraction_decoding": config["protocol"].get("extraction_decoding", "free"),
          "extraction_errors": sum("error" in r for r in all_records),
          "ontology_violations": sum(len(r["violations"]) for r in all_records),
          "empty_extractions": sum(not r["entities"] and not r["relations"] for r in all_records)}
    write_json(folder / f"qc-{suite}.json", qc)
    print(canonical(qc))


def retrieve(config, label, suite):
    full_gate(config, label, suite)
    root, folder = job_paths(config, label)
    source = extraction_paths(config, label)
    encoder = Encoder(config["embedding"], config["runtime"])
    official = load_official()
    grouped = defaultdict(list)
    for question in selected_questions(suite):
        grouped[question["user_id"]].append(question)
    for user, rows in sorted(grouped.items()):
        path = source / "memories" / f"{user}.json"
        if not path.exists():
            raise ValueError(f"Run extract first; missing {path}")
        documents = read_json(path)["documents"]
        index = index_user(user, documents, config, encoder, folder)
        for row in rows:
            path = folder / "retrieval" / suite / f"{row['id']}.json"
            if not path.exists():
                write_json(path, retrieve_question(row, documents, index, config, official))
        print(f"{user}: retrieval saved / поиск сохранён", flush=True)


def answer(config, label, suite):
    full_gate(config, label, suite)
    root, folder = job_paths(config, label)
    source = extraction_paths(config, label)
    generator = Generation(config["models"][label], config["runtime"], folder / "requests", sampling=config["protocol"])
    available = generator.client.models.list()
    if config["models"][label]["model"] not in {m.id for m in available.data}:
        raise ValueError("Expected served model not present")
    questions = selected_questions(suite)

    def process(row):
        path = folder / "answers" / suite / f"{row['id']}.json"
        if not path.exists():
            retrieved = read_json(folder / "retrieval" / suite / f"{row['id']}.json")
            result = generate_answer(row, retrieved, config, generator)
            write_json(path, result)
        print(f"{label} {suite}: {row['id']} answered / ответ получен", flush=True)
    with ThreadPoolExecutor(max_workers=config["runtime"]["concurrency"]) as pool:
        list(pool.map(process, questions))

    report = make_report(folder, suite)
    extraction_qc = read_json(source / f"qc-{suite}.json")
    qc = {**extraction_qc, "answers": report["overall"]["n"],
          "pipeline_complete": report["overall"]["n"] == len(questions) and extraction_qc["extraction_errors"] == 0,
          "judge_pending": report["overall"]["judged"] < len(questions)}
    write_json(folder / suite / "qc.json", qc)
    print(canonical(qc))
    print(comparison(root, suite, config["models"]))


def run(config, label, suite):
    extract(config, label, suite)
    retrieve(config, label, suite)
    answer(config, label, suite)


def score(config, label, suite):
    if not config.get("judge"):
        raise ValueError("Configure independent judge endpoint first; no self-judging")
    judge = config["judge"]
    if judge["model"] == config["models"][label]["model"]:
        raise ValueError("Judge must be independent of the evaluated model")
    if judge.get("source_model", judge["model"]) != "ai-forever/pollux-judge-32b":
        raise ValueError("Official RU rumba_metric requires Pollux; configure judge.source_model")
    root, folder = job_paths(config, label)
    generator = Generation(judge, config["runtime"], folder / "judge-requests")
    identity_path = folder / "scores" / "judge.json"
    if identity_path.exists() and read_json(identity_path) != judge:
        raise ValueError("Judge changed; do not mix scores in the same run")
    write_json(identity_path, judge)
    official = load_official()
    paths = sorted((folder / "answers" / suite).glob("*.json"))
    expected = len(selected_questions(suite))
    if len(paths) != expected:
        raise ValueError(f"Expected {expected} answers, found {len(paths)}")

    def process(path):
        row = read_json(path)
        output = folder / "scores" / suite / path.name
        if output.exists():
            return
        question = f"{row['question']} (текущая дата: {timestamp(row['question_date']).date()})"
        messages = official["judge_template"](question, row["prediction"], row["answer"])
        response = generator.call(messages, config["protocol"]["judge_max_tokens"])
        reason, verdict = official["extract_judge_feedback"](response["text"])
        result = {"id": row["id"], "prediction": row["prediction"], "judge": judge,
                  "rumba_metric": official["process_judge_response"](response["text"]),
                  "judge_verdict": verdict, "judge_reason": reason, "judge_parse_error": verdict not in (0, 1, 2, 3)}
        write_json(output, result)
        print(f"{row['id']}: {result['rumba_metric']}", flush=True)
    with ThreadPoolExecutor(max_workers=config["runtime"]["concurrency"]) as pool:
        list(pool.map(process, paths))
    make_report(folder, suite)
    print(comparison(root, suite, config["models"]))


def main():
    parser = argparse.ArgumentParser(description="RUMBA benchmark / Бенчмарк RUMBA")
    parser.add_argument("command", choices=["preflight", "prepare", "estimate", "smoke", "extract", "retrieve", "answer", "run", "score", "report"])
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--model", default="meno-lite-0.2")
    parser.add_argument("--suite", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--extraction-decoding", choices=["free", "structured"],
                        help="Extraction protocol; use the same mode at every stage / Режим извлечения для всех этапов")
    args = parser.parse_args()
    if not args.config.exists():
        parser.error("Copy config.example.json to config.json and configure endpoints / Скопируйте шаблон и укажите адреса серверов")
    config = read_json(args.config)
    config["protocol"].setdefault("extraction_decoding", "free")
    config["protocol"].setdefault("structured_limits", {"entities": 16, "relations": 16, "name_chars": 80, "description_chars": 200})
    config["protocol"].setdefault("top_p", 1.0)
    config["protocol"].setdefault("repetition_penalty", 1.0)
    if args.extraction_decoding:
        config["protocol"]["extraction_decoding"] = args.extraction_decoding
    if config["protocol"]["extraction_stage"] != "one_stage" or config["protocol"]["temperature"] != 0:
        raise ValueError("This frozen protocol implements greedy one-stage extraction only")
    if args.command == "prepare":
        prepare()
    elif args.command == "preflight":
        preflight(config, args.model)
    elif args.command == "estimate":
        estimate(config, args.suite)
    elif args.command == "smoke":
        smoke(config, args.model)
    elif args.command == "run":
        run(config, args.model, args.suite)
    elif args.command == "extract":
        extract(config, args.model, args.suite)
    elif args.command == "retrieve":
        retrieve(config, args.model, args.suite)
    elif args.command == "answer":
        answer(config, args.model, args.suite)
    elif args.command == "score":
        score(config, args.model, args.suite)
    else:
        root, folder = job_paths(config, args.model)
        make_report(folder, args.suite)
        print(comparison(root, args.suite, config["models"]))


if __name__ == "__main__":
    main()
