"""Compare identical pilot questions across protocols. / Сравнение тех же вопросов между протоколами."""
import argparse
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rumba_bench.data import canonical, category, read_json, write_json


def compare(before, after, suite="pilot"):
    old = {row["id"]: row for row in read_json(before / suite / "predictions.json")}
    new = {row["id"]: row for row in read_json(after / suite / "predictions.json")}
    if set(old) != set(new):
        raise ValueError("Runs must contain identical question IDs")
    rows = []
    for key in sorted(old):
        a, b = old[key], new[key]
        if any(a[field] != b[field] for field in ("question", "answer", "category", "user_id")):
            raise ValueError(f"Dataset changed for {key}")
        rows.append({"id": key, "category": b["category"], "question": b["question"], "gold": b["answer"],
                     "before": a["prediction"], "after": b["prediction"], "f1_before": a["f1"], "f1_after": b["f1"],
                     "delta": b["f1"] - a["f1"], "retrieval_before": a["retrieval_recall"],
                     "retrieval_after": b["retrieval_recall"]})
    def group(items):
        return {"n": len(items), "f1_before": mean(r["f1_before"] for r in items),
                "f1_after": mean(r["f1_after"] for r in items), "f1_delta": mean(r["delta"] for r in items),
                "improved": sum(r["delta"] > 0 for r in items), "worsened": sum(r["delta"] < 0 for r in items),
                "unchanged": sum(r["delta"] == 0 for r in items)}
    categories = sorted({category(r)[1:] for r in rows})
    return {"before": str(before), "after": str(after), "overall": group(rows),
            "temporal_x_session_scope": {f"{temporal} × {scope}": group([r for r in rows if category(r)[1:] == (scope, temporal)])
                                         for scope, temporal in categories},
            "note": "Development pilot: both memory pipeline and repetition penalty changed; no causal separation or representative full-set estimate.",
            "questions": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True, type=Path)
    p.add_argument("--after", required=True, type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "runs/memory-comparison.json")
    args = p.parse_args()
    result = compare(args.before, args.after)
    write_json(args.output, result)
    print(canonical({key: value for key, value in result.items() if key != "questions"}))


if __name__ == "__main__":
    main()
