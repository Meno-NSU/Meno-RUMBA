"""Paired RUMBA comparisons and independent available-answer summaries."""
import csv
from collections import defaultdict
from pathlib import Path

from .data import category, digest, read_json, write_json
from .report import make_report, summary

MENO01 = 'meno-lite-0.1'
MENO02 = 'meno-lite-0.2'
QWEN = 'qwen2.5-7b-instruct'


def define_cohorts(ids):
    if not ids[MENO02] <= ids[MENO01] or not ids[MENO02] <= ids[QWEN]:
        raise ValueError('The Meno0.2 anchor is not fully present in both other models')
    if not ids[QWEN] <= ids[MENO01]:
        raise ValueError('The Qwen anchor is not fully present in Meno0.1')
    return [
        {'name': 'meno02_shared', 'paired': True, 'anchor': MENO02,
         'models': [MENO02, MENO01, QWEN], 'ids': sorted(ids[MENO02])},
        {'name': 'qwen_shared', 'paired': True, 'anchor': QWEN,
         'models': [QWEN, MENO01], 'ids': sorted(ids[QWEN])},
        {'name': 'available', 'paired': False, 'models': [MENO01, QWEN, MENO02],
         'ids_by_model': {m: sorted(s) for m, s in ids.items()}},
    ]


def build_reports(folders, output):
    folders = {m: Path(p) for m, p in folders.items()}
    rows = {}
    for model, folder in folders.items():
        make_report(folder, 'full')
        values = read_json(folder / 'full/predictions.json')
        rows[model] = {r['id']: r for r in values}
        if len(rows[model]) != len(values): raise ValueError('Duplicate answer ID')
    cohorts = define_cohorts({m: set(v) for m, v in rows.items()})
    result = {'status': 'judge_pending', 'cohorts': {},
              'warning': 'Stopped runs; anchor cohorts are not a random sample of the full test set.'}
    csv_rows, markdown = [], ['# RUMBA: matched question cohorts', '', result['warning'], '']
    for cohort in cohorts:
        item = {'paired': cohort['paired'], 'anchor': cohort.get('anchor'), 'models': {}}
        markdown.extend(['## ' + cohort['name'], '', '| Model | n | Judged | RUMBA | F1 |', '|---|---:|---:|---:|---:|'])
        reference = None
        for model in cohort['models']:
            ids = cohort['ids'] if cohort['paired'] else cohort['ids_by_model'][model]
            selected = [rows[model][i] for i in ids]
            gold = {r['id']: tuple(r[k] for k in ['question', 'answer', 'question_date', 'category']) for r in selected}
            if cohort['paired']:
                if reference is not None and gold != reference: raise ValueError('Paired questions differ across models')
                reference = gold
            grouped = defaultdict(list)
            for row in selected:
                _, scope, temporal = category(row); grouped[f'{temporal} × {scope}'].append(row)
            metrics = {'question_ids_sha256': digest(ids), 'overall': summary(selected),
                       'temporal_x_session_scope': {k: summary(v) for k, v in sorted(grouped.items())}}
            item['models'][model] = metrics
            overall = metrics['overall']; value = overall['rumba_metric']
            # A numeric cohort score is final only when every selected row is judged.
            display = f'{value:.4f}' if value is not None and overall['judged'] == overall['n'] else 'pending'
            markdown.append(f"| {model} | {overall['n']} | {overall['judged']} | {display} | {overall['f1']:.4f} |")
            for group, values in [('overall', overall), *metrics['temporal_x_session_scope'].items()]:
                csv_rows.append({'cohort': cohort['name'], 'paired': cohort['paired'], 'category': group, 'model': model, **values})
        item['status'] = 'scored' if all(m['overall']['judged'] == m['overall']['n'] for m in item['models'].values()) else 'judge_pending'
        result['cohorts'][cohort['name']] = item
        markdown.extend(['', 'Paired question IDs are identical.' if cohort['paired'] else 'Different available sample sizes; this is not a paired comparison.', ''])
    if all(v['status'] == 'scored' for v in result['cohorts'].values()): result['status'] = 'scored'
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'cohort-plan.json', {'folders': {m: str(p.resolve()) for m, p in folders.items()}, 'cohorts': cohorts})
    write_json(output / 'reports.json', result)
    (output / 'comparison.md').write_text('\n'.join(markdown) + '\n')
    with (output / 'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in csv_rows))))
        writer.writeheader(); writer.writerows(csv_rows)
    return result
