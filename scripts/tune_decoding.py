"""Resume-able technical calibration, separate from scored RUMBA.

Возобновляемая техническая калибровка, отдельно от зачётных результатов RUMBA.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rumba_bench.data import canonical, digest, read_json, write_json
from rumba_bench.pipeline import Generation, parse_extraction, training_modules


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold().replace("ё", "е")))


def assess(task, response, ontology):
    text = response["text"]
    result = {"complete": response["finish_reason"] == "stop", "json_valid": None,
              "entity_recall": None, "qa_correct": None, "duplicate_fraction": None,
              "ontology_violation_fraction": None}
    if task["kind"] == "qa":
        if "expected_json" in task:
            try:
                correct = json.loads(text.strip()) == task["expected_json"]
            except (ValueError, TypeError):
                correct = False
        else:
            # Preserve signs and internal punctuation; normalization must not turn -56 into 56.
            # Сохранять знаки: нормализация не должна превращать -56 в 56.
            correct = text.strip().casefold().rstrip(".!?").strip() == task["expected"].strip().casefold().rstrip(".!?").strip()
        result["qa_correct"] = result["complete"] and correct
        return result
    try:
        parsed = parse_extraction(text, ontology)
        raw = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))
    except (ValueError, TypeError):
        result["json_valid"] = False
        if task.get("expected_entities"):
            result["entity_recall"] = 0.0
        return result
    result["json_valid"] = result["complete"]
    keys = [(normalized(e["entity_name"]), e["entity_type"]) for e in parsed["entities"]]
    keys += [(r["source_entity"], r["target_entity"], r["relation_type"], r["description"]) for r in parsed["relations"]]
    result["duplicate_fraction"] = (len(keys) - len(set(keys))) / len(keys) if keys else 0.0
    count = len(raw["entities"]) + len(raw["relations"])
    result["ontology_violation_fraction"] = len(parsed["violations"]) / count if count else 0.0
    if task.get("expected_entities"):
        actual = {(normalized(e["entity_name"]), e["entity_type"]) for e in parsed["entities"]}
        matched = sum(any((normalized(alias), gold["type"]) in actual for alias in gold["aliases"])
                      for gold in task["expected_entities"])
        result["entity_recall"] = matched / len(task["expected_entities"])
    return result


def load_tasks(config):
    ontology, build_prompt, *_ = training_modules(config)
    dev = read_json(ROOT / "fixtures/decoding_dev.json")
    tasks = []
    for row in dev:
        if row["kind"] == "extraction":
            messages = build_prompt("one_stage", row["text"], ontology.full_schema(), language="Russian",
                                    variant=config["protocol"]["prompt_variant"], with_signatures=True)
        else:
            messages = [{"role": "user", "content": row["prompt"]}]
        tasks.append({**row, "group": "synthetic-dev", "messages": messages,
                      "max_tokens": 2048 if row["kind"] == "extraction" else 128})
    failed, controls, seen = [], [], set()
    for path in sorted((ROOT / "runs/extraction").glob("*/*/requests/*.json")):
        record = read_json(path)
        request = record["request"]
        if "structured_outputs" in request.get("extra_body", {}):
            continue
        key = digest(request["messages"])
        if key in seen:
            continue
        seen.add(key)
        row = {"id": "regression-" + key[:12], "kind": "extraction", "group": "regression",
               "messages": request["messages"], "max_tokens": 4096,
               "baseline_finish_reason": record["finish_reason"]}
        if record["finish_reason"] == "length":
            failed.append(row)
        else:
            controls.append(row)
    return tasks + failed[:5] + controls[:3], ontology


def aggregate(rows):
    result = {"n": len(rows)}
    for metric in ("complete", "json_valid", "entity_recall", "qa_correct", "duplicate_fraction", "ontology_violation_fraction"):
        values = [r[metric] for r in rows if r.get(metric) is not None]
        result[metric] = mean(values) if values else None
    result["mean_tokens"] = mean(r["usage"].get("completion_tokens", 0) for r in rows)
    return result


def summarize(rows):
    grouped = {}
    for row in rows:
        key = canonical(row["sampling"])
        grouped.setdefault(key, []).append(row)
    result = []
    for key, items in grouped.items():
        result.append({"sampling": json.loads(key), "overall": aggregate(items),
                       "regression": aggregate([r for r in items if r["group"] == "regression"]) if any(r["group"] == "regression" for r in items) else None,
                       "dev": aggregate([r for r in items if r["group"] == "synthetic-dev"])})
    return result


def choose_penalty(summaries):
    baseline = next(r for r in summaries if r["sampling"]["repetition_penalty"] == 1.0)
    def quality(row):
        d = row["dev"]
        return (d["qa_correct"] or 0, d["entity_recall"] or 0)
    bqa, brecall = quality(baseline)
    eligible = [r for r in summaries if quality(r)[0] >= bqa - 0.05 and quality(r)[1] >= brecall - 0.10]
    return max(eligible, key=lambda r: ((r["regression"] or r["overall"])["json_valid"] or 0,
               sum(quality(r)), -(r["overall"]["duplicate_fraction"] or 0),
               -(r["overall"]["ontology_violation_fraction"] or 0), -r["sampling"]["repetition_penalty"]))


def run_grid(tasks, settings, seeds, config, folder, ontology, workers):
    jobs = [(task, sampling, seed) for sampling in settings for seed in seeds for task in tasks]
    clients = {canonical(s): Generation(config["models"]["meno-lite-0.2"], config["runtime"], folder.parent / "requests", sampling=s) for s in settings}
    def process(job):
        task, sampling, seed = job
        path = folder / "rows" / f"{digest({'task':task,'sampling':sampling,'seed':seed})}.json"
        if path.exists():
            return read_json(path)
        response = clients[canonical(sampling)].call(task["messages"], task["max_tokens"], seed=seed, allow_incomplete=True)
        row = {"id": task["id"], "group": task["group"], "kind": task["kind"], "sampling": sampling,
               "seed": seed, "finish_reason": response["finish_reason"], "seconds": response["seconds"],
               "usage": response["usage"], **assess(task, response, ontology)}
        write_json(path, row)
        print(f"{task['id']} {sampling} seed={seed}: {row['finish_reason']}", flush=True)
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(process, jobs))
    for client in clients.values():
        client.client.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--penalties", type=float, nargs="+", default=[1.0, 1.03, 1.05, 1.10, 1.15])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    config = read_json(args.config)
    tasks, ontology = load_tasks(config)
    manifest = {"tasks": tasks, "penalties": args.penalties, "temperature_stage": [0.2, 0.5],
                "seeds": [42, 43], "script_sha256": digest(Path(__file__).read_text()),
                "model": config["models"]["meno-lite-0.2"],
                "note": "Provisional technical calibration; no Pollux and no scored RUMBA answer tuning"}
    folder = ROOT / "runs/decoding-calibration" / digest(manifest)[:16]
    write_json(folder / "manifest.json", manifest)
    settings = [{"temperature": 0, "top_p": 1.0, "repetition_penalty": p} for p in args.penalties]
    rows = run_grid(tasks, settings, [42], config, folder, ontology, args.workers)
    summaries = summarize(rows)
    best = choose_penalty(summaries)
    write_json(folder / "penalty-summary.json", {"candidates": summaries, "provisional_candidate": best})
    print("PENALTY_CANDIDATE " + canonical(best), flush=True)
    dev = [task for task in tasks if task["group"] == "synthetic-dev"]
    r = best["sampling"]["repetition_penalty"]
    sampled = [{"temperature": t, "top_p": 1.0, "repetition_penalty": r} for t in (0.2, 0.5)]
    temperature_rows = run_grid(dev, sampled, [42, 43], config, folder, ontology, args.workers)
    temperature_summary = summarize(temperature_rows)
    sampled_candidate = max(temperature_summary, key=lambda item: (item["dev"]["qa_correct"] or 0,
                           item["dev"]["entity_recall"] or 0, -(item["dev"]["ontology_violation_fraction"] or 0),
                           -item["sampling"]["temperature"]))
    write_json(folder / "summary.json", {"status": "provisional_without_judge", "penalty_candidates": summaries,
               "penalty_candidate": best, "temperature_candidates": temperature_summary, "optional_sampling_candidate": sampled_candidate,
               "limitations": ["Only one model", f"{len(dev)} synthetic dev cases and {len(tasks)-len(dev)} cached technical regressions",
                               "No semantic judge; entity recall checks names/types only", "No general release optimum established"]})
    write_json(folder / "generation_config.greedy.patch.json", {"do_sample": False, "repetition_penalty": r})
    write_json(folder / "generation_config.sampling.patch.json", {"do_sample": True, **sampled_candidate["sampling"]})
    print("REPORT " + str(folder / "summary.json"), flush=True)


if __name__ == "__main__":
    main()
