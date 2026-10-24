"""Metrics and comparison tables. / Метрики и сравнительные таблицы."""
import csv
from collections import defaultdict
from statistics import mean

from .data import category, read_json, write_json
from .upstream import f1_score


def summary(rows):
    def average(key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return mean(values) if values else None
    return {"n": len(rows), "judged": sum(row.get("rumba_metric") is not None for row in rows),
            "rumba_metric": average("rumba_metric"), "f1": average("f1"),
            "retrieval_recall": average("retrieval_recall"),
            "all_evidence_retrieved": average("all_evidence_retrieved"), "seconds": average("seconds")}


def make_report(folder, suite):
    answers = sorted((folder / "answers" / suite).glob("*.json"))
    rows = []
    for path in answers:
        row = read_json(path)
        row["f1"] = f1_score(row["answer"], row["prediction"])
        score_path = folder / "scores" / suite / path.name
        if score_path.exists():
            score = read_json(score_path)
            if score.get("prediction") != row["prediction"]:
                raise ValueError(f"Score belongs to a different prediction: {score_path}")
            row.update({k: score[k] for k in ("rumba_metric", "judge_verdict", "judge_parse_error")})
        rows.append(row)
    grouped, semantic = defaultdict(list), defaultdict(list)
    for row in rows:
        kind, scope, temporal = category(row)
        grouped[f"{temporal} × {scope}"].append(row)
        semantic[kind].append(row)
    report = {"model": folder.name, "suite": suite,
              "status": "scored" if rows and all("rumba_metric" in r for r in rows) else "judge_pending",
              "overall": summary(rows), "temporal_x_session_scope": {k: summary(v) for k, v in sorted(grouped.items())},
              "semantic": {k: summary(v) for k, v in sorted(semantic.items())},
              "judge_parse_errors": sum(r.get("judge_parse_error", False) for r in rows)}
    write_json(folder / suite / "report.json", report)
    write_json(folder / suite / "predictions.json", rows)
    return report


def comparison(root, suite, model_labels):
    reports = {}
    for label in model_labels:
        path = root / label / suite / "report.json"
        if path.exists():
            reports[label] = read_json(path)
    groups = ["overall", "atemporal × single", "atemporal × multi", "temporal × single", "temporal × multi"]
    table = ["| Категория | " + " | ".join(model_labels) + " |",
             "|---|" + "---|" * len(model_labels)]
    csv_rows = []
    for group in groups:
        cells = []
        for label in model_labels:
            report = reports.get(label, {})
            metrics = report.get("overall", {}) if group == "overall" else report.get("temporal_x_session_scope", {}).get(group, {})
            n, judged = metrics.get("n", 0), metrics.get("judged", 0)
            f1, score = metrics.get("f1"), metrics.get("rumba_metric")
            # No partial judge mean presented as a complete result. / Неполная оценка явно помечена.
            judge_text = f"{score:.4f}" if score is not None else "ожидается"
            if judged < n:
                judge_text += f" ({judged}/{n})"
            cells.append(f"RUMBA {judge_text}; F1 {f1:.4f}; n={n}" if f1 is not None else "ожидается")
            csv_rows.append({"category": group, "model": label, **metrics})
        table.append("| " + group + " | " + " | ".join(cells) + " |")
    prefix = "# RUMBA RU — " + suite + "\n\n"
    if suite == "pilot":
        prefix += "Пилот для отладки: 50 вопросов из нескольких пользователей с покрытием четырёх групп. Числа не являются оценкой всей выборки.\n\n"
    text = prefix + "\n".join(table) + "\n\nRUMBA: официальный промпт Pollux и разбор оценки; F1: LightEval 0.13.0, без нормализации.\n"
    (root / f"comparison-{suite}.md").write_text(text)
    keys = sorted(set().union(*(r.keys() for r in csv_rows)))
    with (root / f"comparison-{suite}.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(csv_rows)
    return text
