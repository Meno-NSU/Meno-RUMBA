import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'entity_dedup', Path(__file__).resolve().parents[1] / 'scripts/run_ragu_entity_dedup.py')
dedup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dedup)


def test_homogeneous_duplicates_collapsed_and_audited(tmp_path):
    class Summarizer:
        async def _summarize_by_cluster_if_needed(self, descriptions):
            return descriptions
    audit = tmp_path / 'audit.jsonl'
    dedup.install_dedup_patch(Summarizer, audit, 'repair', lambda: 'user_2')
    source = ['В апреле посетил Францию.'] * 9
    result = asyncio.run(Summarizer()._summarize_by_cluster_if_needed(source))
    assert result == [source[0]]
    assert len(source) == 9
    event = json.loads(audit.read_text())
    assert event['input'] == source and event['retained'] == result
    assert event['unique_values_preserved'] and event['context'] == 'user_2'


def test_mixed_lists_remain_byte_identical_even_with_duplicate_facts(tmp_path):
    class Summarizer:
        async def _summarize_by_cluster_if_needed(self, descriptions):
            return descriptions
    audit = tmp_path / 'audit.jsonl'
    dedup.install_dedup_patch(Summarizer, audit, 'repair')
    source = ['В апреле посетил Францию.', 'В апреле посетил Францию.',
              'В мае НЕ посетил Францию.', 'В апреле посетил Францию. ']
    assert asyncio.run(Summarizer()._summarize_by_cluster_if_needed(source)) == source
    assert not audit.exists()


def test_ragu_grouping_keeps_all_sources_and_original_summary_threshold(tmp_path, monkeypatch):
    from ragu.graph.artifacts_summarizer import EntitySummarizer
    from ragu.graph.types import Entity
    original = EntitySummarizer._summarize_by_cluster_if_needed
    monkeypatch.setattr(EntitySummarizer, '_summarize_by_cluster_if_needed', original)
    dedup.install_dedup_patch(EntitySummarizer, tmp_path / 'audit.jsonl', 'repair')
    entities = [Entity(entity_name='Франция', entity_type='COUNTRY', description='Страна в Европе.',
                       source_chunk_id=[f'chunk{i}'], documents_id=[f'doc{i}']) for i in range(9)]
    grouped = EntitySummarizer.group_entities(entities)
    assert grouped.iloc[0]['duplicate_count'] == 9
    assert set(grouped.iloc[0]['source_chunk_id']) == {f'chunk{i}' for i in range(9)}
    summarizer = EntitySummarizer(use_llm_summarization=False, summarize_only_if_more_than=7)
    result = asyncio.run(summarizer.summarize_entities(grouped))
    assert result[0].description == 'Страна в Европе.'
    assert set(result[0].source_chunk_id) == {f'chunk{i}' for i in range(9)}
    assert set(result[0].documents_id) == {f'doc{i}' for i in range(9)}


def test_failed_summary_retries_once_preserving_other_fields_and_original_raw(tmp_path):
    descriptions = ['Был в апреле.', 'Был в апреле.', 'НЕ был в мае.']
    message = [{'role': 'user', 'content': 'Entity name: X, Description: ' + ' '.join(descriptions)
                + '\n\nProvide the answer in russian'}]
    request = {'model': 'meno-lite-0.2', 'messages': message, 'temperature': 0, 'seed': 42,
               'max_tokens': 4096, 'extra_body': {'repetition_penalty': 1.03,
               'structured_outputs': {'json': {'title': 'EntityDescriptionModel'}}}}
    raw_path = tmp_path / 'original.json'
    raw_path.write_text(json.dumps({'request': request, 'finish_reason': 'length', 'text': 'unfinished'}))
    original_bytes = raw_path.read_bytes()
    class Generation:
        def __init__(self): self.calls = []
        def call(self, messages, max_tokens, seed=42):
            self.calls.append((messages, max_tokens, seed))
            if len(self.calls) == 1:
                raise ValueError('Generation incomplete (length); raw response: ' + str(raw_path))
            return {'request': {**request, 'messages': messages}, 'finish_reason': 'stop', 'text': '{}'}
    registry = {('user_21', ' '.join(descriptions)): descriptions}
    audit = tmp_path / 'audit.jsonl'
    dedup.install_failed_summary_retry(Generation, registry, audit, 'repair', lambda: 'user_21')
    generation = Generation()
    result = generation.call(message, 4096, seed=42)
    assert len(generation.calls) == 2 and generation.calls[1][1:] == (4096, 42)
    assert result['request']['messages'][0]['content'] == 'Entity name: X, Description: Был в апреле. НЕ был в мае.\n\nProvide the answer in russian'
    assert raw_path.read_bytes() == original_bytes
    assert message == request['messages']
    events = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [e['outcome'] for e in events] == ['started', 'stop']
    assert events[-1]['result_request_sha256'] == events[0]['retry_request_sha256']
    assert events[0]['unique_values_preserved'] and events[0]['generation_fields_unchanged']


def test_successful_summary_is_not_changed_or_retried(tmp_path):
    class Generation:
        def call(self, messages, max_tokens): return messages
    audit = tmp_path / 'audit.jsonl'
    dedup.install_failed_summary_retry(Generation, {}, audit, 'repair', lambda: 'user_21')
    original = [{'role': 'user', 'content': 'original duplicate duplicate'}]
    assert Generation().call(original, 4096) is original
    assert not audit.exists()


def test_non_summary_failure_is_not_retried(tmp_path):
    source = tmp_path / 'failed.json'
    messages = [{'role': 'user', 'content': 'Description: A A\n\nProvide the answer'}]
    source.write_text(json.dumps({'request': {'model': 'meno-lite-0.2', 'messages': messages,
        'extra_body': {'structured_outputs': {'json': {'title': 'EntitiesExtractionModel'}}}}, 'finish_reason': 'length'}))
    class Generation:
        calls = 0
        def call(self, messages, max_tokens):
            self.calls += 1
            raise ValueError('Generation incomplete (length); raw response: ' + str(source))
    dedup.install_failed_summary_retry(Generation, {('u', 'A A'): ['A', 'A']}, tmp_path/'audit', 'repair', lambda:'u')
    generation = Generation()
    with pytest.raises(ValueError, match='Generation incomplete'):
        generation.call(messages, 4096)
    assert generation.calls == 1


def test_recovery_failure_propagates_after_one_retry(tmp_path):
    source = tmp_path / 'failed.json'
    messages = [{'role': 'user', 'content': 'Description: A A B\n\nProvide the answer'}]
    source.write_text(json.dumps({'request': {'model': 'meno-lite-0.2', 'messages': messages,
        'extra_body': {'structured_outputs': {'json': {'title': 'EntityDescriptionModel'}}}}, 'finish_reason': 'length'}))
    class Generation:
        calls = 0
        def call(self, messages, max_tokens):
            self.calls += 1
            if self.calls == 1:
                raise ValueError('Generation incomplete (length); raw response: ' + str(source))
            raise RuntimeError('Retry also failed')
    audit = tmp_path / 'audit.jsonl'
    dedup.install_failed_summary_retry(Generation, {('u', 'A A B'): ['A', 'A', 'B']}, audit, 'repair', lambda:'u')
    generation = Generation()
    with pytest.raises(RuntimeError, match='Retry also failed'):
        generation.call(messages, 4096)
    assert generation.calls == 2
    assert [json.loads(line)['outcome'] for line in audit.read_text().splitlines()] == ['started', 'failed']


def test_explicit_summary_prompt_retains_every_unique_description_and_sampling(tmp_path):
    descriptions = ['В апреле был в Польше.', 'В мае НЕ был в Польше.']
    messages = [{'role': 'user', 'content': 'Entity name: user_21, Description: '
                + ' '.join(descriptions) + '\n\nProvide the answer'}]
    request = {'model': 'meno-lite-0.2', 'messages': messages, 'max_tokens': 4096, 'seed': 42,
               'temperature': 0, 'extra_body': {'repetition_penalty': 1.03,
               'structured_outputs': {'json': {'title': 'EntityDescriptionModel'}}}}
    source = tmp_path / 'failed.json'
    source.write_text(json.dumps({'request': request, 'finish_reason': 'length'}))
    class Generation:
        calls = 0
        def call(self, messages, max_tokens, seed):
            self.calls += 1
            if self.calls == 1:
                raise ValueError('Generation incomplete (length); raw response: ' + str(source))
            assert messages == dedup.summary_retry_messages('user_21', descriptions)
            assert (max_tokens, seed) == (4096, 42)
            return {'request': {**request, 'messages': messages}, 'finish_reason': 'stop'}
    audit = tmp_path / 'audit.jsonl'
    dedup.install_failed_summary_retry(Generation, {('u', ' '.join(descriptions)): descriptions},
                                       audit, 'repair', lambda:'u', clarify_prompt=True)
    result = Generation().call(messages, 4096, seed=42)
    assert result['finish_reason'] == 'stop'
    assert json.dumps(descriptions, ensure_ascii=False, indent=2) in result['request']['messages'][1]['content']
    assert [json.loads(line)['outcome'] for line in audit.read_text().splitlines()] == ['started', 'stop']
