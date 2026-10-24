"""Bounded candidate protocol; never mixes with benchmark runs.

Кандидат ограниченного декодирования; результаты не смешиваются с бенчмарком.
"""
import argparse
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rumba_bench.data import digest, read_json, write_json
from rumba_bench.pipeline import client_for, parse_extraction, training_modules
from rumba_bench.schemas import extraction_schema


def bounded_schema(ontology):
    return extraction_schema(ontology, {"extraction_decoding": "structured"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2)
    args = parser.parse_args()
    config = read_json(ROOT / "config.json")
    ontology, *_ = training_modules(config)
    candidates = []
    for path in sorted((ROOT / "runs/extraction").glob("*/*/requests/*.json")):
        record = read_json(path)
        if record["finish_reason"] == "length":
            candidates.append((path, record))
    client = client_for(config["models"]["meno-lite-0.2"], config["runtime"])
    for path, original in candidates[:args.count]:
        schema = bounded_schema(ontology)
        request = {**original["request"], "extra_body": {**original["request"].get("extra_body", {}),
                                                       "structured_outputs": {"json": schema}}}
        output = ROOT / "runs/structured-probe" / f"{digest(request)}.json"
        if output.exists():
            result = read_json(output)
        else:
            start = time.monotonic()
            response = client.chat.completions.create(**request)
            choice = response.choices[0]
            result = {"baseline": str(path), "request": request, "text": choice.message.content,
                      "finish_reason": choice.finish_reason, "seconds": time.monotonic() - start,
                      "usage": response.usage.model_dump()}
            if choice.finish_reason == "stop":
                try:
                    result["parsed"] = parse_extraction(choice.message.content, ontology)
                except ValueError as error:
                    result["parse_error"] = str(error)
            write_json(output, result)
        parsed = result.get("parsed", {})
        print({"output": str(output), "finish_reason": result["finish_reason"], "seconds": result["seconds"],
               "usage": result["usage"], "entities": len(parsed.get("entities", [])),
               "relations": len(parsed.get("relations", [])), "violations": len(parsed.get("violations", []))}, flush=True)


if __name__ == "__main__":
    main()
