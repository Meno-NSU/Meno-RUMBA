import json
from pathlib import Path
import importlib.util
from types import SimpleNamespace

import pytest

from rumba_bench.cli import scoring_coverage
from rumba_bench.report import make_report


def row(number):
    return {'id': f'0|q{number}', 'user_id': 'user_0', 'question': f'Question {number}',
            'question_date': '2025-01-01 00:00:00', 'answer': 'answer',
            'prediction': 'answer', 'finish_reason': 'stop', 'truncated': False,
            'category': 'StaticUser, single, atemporal',
            'raw_response_hash': 'saved-wire-hash'}


def test_partial_scoring_is_explicit_and_preserves_missing_ids():
    questions = [row(0), row(1)]
    with pytest.raises(ValueError, match='allow-partial'):
        scoring_coverage([row(0)], questions)
    coverage = scoring_coverage([row(0)], questions, allow_partial=True)
    assert coverage['status'] == 'partial'
    assert coverage['expected_questions'] == 2 and coverage['included_questions'] == 1
    assert coverage['missing_ids'] == ['0|q1']


def test_partial_flag_does_not_allow_corrupt_or_foreign_answers():
    for invalid in [{**row(0), 'id': 'foreign'}, {**row(0), 'truncated': True},
                    {**row(0), 'answer': 'changed reference'}]:
        with pytest.raises(ValueError):
            scoring_coverage([invalid], [row(0), row(1)], allow_partial=True)


def test_export_ignores_unclosed_graph_and_hashes_exact_saved_answer(tmp_path):
    spec = importlib.util.spec_from_file_location('exporter', Path(__file__).resolve().parents[1] / 'scripts/export_partial_results.py')
    exporter = importlib.util.module_from_spec(spec); spec.loader.exec_module(exporter)
    root = tmp_path / 'ragu-example'; root.mkdir(); (root / 'manifest.json').write_text('{}')
    folder = root / 'model'; graph = folder / 'graphs/user_0'; graph.mkdir(parents=True)
    (graph / 'build_report.json').write_text(json.dumps({'consistent': True, 'all_source_chunks_stored': True, 'llm_errors': []}))
    (graph / '.checkpoint-ready.json').write_text(json.dumps({'suite': 'full', 'graph_closed': True, 'answers': ['0_q0.json']}))
    answers = folder / 'answers/full'; answers.mkdir(parents=True)
    payload = json.dumps(row(0)).encode(); (answers / '0_q0.json').write_bytes(payload)
    (folder / 'graphs/user_unclosed').mkdir()
    questions = tmp_path / 'questions.jsonl'; questions.write_text('\n'.join(map(json.dumps, [row(0), row(1)])))
    files, metadata = exporter.collect(SimpleNamespace(checkpoints=None, run_folder=folder, questions=questions, model=['model']))
    assert files['ragu-example/model/answers/full/0_q0.json'] == payload
    assert metadata['model']['included_questions'] == 1 and metadata['model']['missing_ids'] == ['0|q1']
    receipts = json.loads(files['files.sha256.json'])
    import hashlib
    assert next(r for r in receipts if r['path'].endswith('0_q0.json'))['sha256'] == hashlib.sha256(payload).hexdigest()


def test_anchor_cohorts_use_exactly_the_same_question_ids():
    from rumba_bench.cohorts import define_cohorts, MENO01, MENO02, QWEN
    plan = define_cohorts({MENO01: {'a', 'b', 'c'}, MENO02: {'a'}, QWEN: {'a', 'b'}})
    assert plan[0]['ids'] == ['a'] and set(plan[0]['models']) == {MENO01, MENO02, QWEN}
    assert plan[1]['ids'] == ['a', 'b'] and set(plan[1]['models']) == {MENO01, QWEN}
    assert not plan[2]['paired']
    with pytest.raises(ValueError, match='anchor'):
        define_cohorts({MENO01: {'b'}, MENO02: {'a'}, QWEN: {'a', 'b'}})


def test_staged_cohorts_judge_each_model_answer_once(tmp_path, monkeypatch):
    from rumba_bench import cli
    from rumba_bench.cohorts import define_cohorts, build_reports, MENO01, MENO02, QWEN
    from rumba_bench.data import write_json
    questions = [row(i) for i in range(3)]
    counts = {MENO01: 3, MENO02: 1, QWEN: 2}
    folders = {}
    for model, count in counts.items():
        folders[model] = tmp_path / model
        for q in questions[:count]: write_json(folders[model] / 'answers/full' / (q['id'] + '.json'), q)
    calls = []
    class Judge:
        def __init__(self, *args, **kwargs): pass
        def call(self, messages, maximum):
            calls.append(messages)
            return {'text': '[FEEDBACK] Верно. [RESULT] 2 [END]'}
    monkeypatch.setattr(cli, 'Generation', Judge)
    monkeypatch.setattr(cli, 'selected_questions', lambda suite: questions)
    config = {'judge': {'model': 'pollux', 'source_model': 'ai-forever/pollux-judge-32b'},
              'models': {m: {'model': m} for m in counts}, 'runtime': {'concurrency': 1},
              'protocol': {'judge_max_tokens': 4096}}
    plan = define_cohorts({m: {q['id'] for q in questions[:n]} for m, n in counts.items()})
    for cohort in plan:
        for model in cohort['models']:
            ids = cohort['ids'] if cohort['paired'] else cohort['ids_by_model'][model]
            cli.score(config, model, 'full', folder_override=folders[model], allow_partial=True, question_ids=ids)
    assert len(calls) == sum(counts.values()) == 6
    report = build_reports(folders, tmp_path / 'reports')
    assert report['status'] == 'scored'
    assert [d['overall']['n'] for d in report['cohorts']['meno02_shared']['models'].values()] == [1, 1, 1]
    assert [d['overall']['n'] for d in report['cohorts']['qwen_shared']['models'].values()] == [2, 2]


def test_public_readme_drops_private_fields_and_hides_incomplete_judge_mean():
    spec = importlib.util.spec_from_file_location('renderer', Path(__file__).resolve().parents[1] / 'scripts/render_benchmark_readme.py')
    renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)
    private = {'status': 'judge_pending', 'private_path': '/private/raw', 'cohorts': {
        'meno02_shared': {'paired': True, 'status': 'judge_pending', 'ids': ['private-id'], 'models': {
            'meno-lite-0.2': {'question_ids_sha256': 'private-identity', 'overall': {
                'n': 2, 'judged': 1, 'rumba_metric': 1.0, 'f1': 0.5, 'prediction': 'private-fact'},
                'temporal_x_session_scope': {}}}}}}
    public = renderer.sanitize(private)
    serialized = json.dumps(public)
    assert not any(value in serialized for value in ['private-fact', 'private-identity', 'private-id', '/private/raw'])
    assert public['cohorts']['meno02_shared']['models']['meno-lite-0.2']['overall']['rumba_metric'] is None
