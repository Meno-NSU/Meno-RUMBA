import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'entity_dedup', Path(__file__).resolve().parents[1] / 'scripts/run_ragu_entity_dedup.py')
dedup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dedup)


def test_exact_duplicates_removed_but_dates_negations_and_order_preserved(tmp_path):
    class Summarizer:
        async def _summarize_by_cluster_if_needed(self, descriptions):
            return descriptions
    audit = tmp_path / 'audit.jsonl'
    dedup.install_dedup_patch(Summarizer, audit, 'repair', lambda: 'user_2')
    source = ['В апреле посетил Францию.', 'В апреле посетил Францию.',
              'В мае НЕ посетил Францию.', 'В апреле посетил Францию. ']
    result = asyncio.run(Summarizer()._summarize_by_cluster_if_needed(source))
    assert result == [source[0], source[2], source[3]]
    assert len(source) == 4
    event = json.loads(audit.read_text())
    assert event['input'] == source and event['retained'] == result
    assert event['unique_values_preserved'] and event['context'] == 'user_2'


def test_unique_descriptions_are_unchanged_without_an_audit_event(tmp_path):
    class Summarizer:
        async def _summarize_by_cluster_if_needed(self, descriptions):
            return descriptions
    audit = tmp_path / 'audit.jsonl'
    dedup.install_dedup_patch(Summarizer, audit, 'repair')
    assert asyncio.run(Summarizer()._summarize_by_cluster_if_needed(['A', 'a'])) == ['A', 'a']
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
