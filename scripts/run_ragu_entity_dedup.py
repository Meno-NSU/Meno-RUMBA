#!/usr/bin/env python3
"""Resume a worker with exact description deduplication before summarization.

Sampling, schemas, prompts, extraction results and provenance are unchanged.
Only repeated, byte-identical entries in EntitySummarizer's description list
are removed before the existing summarization prompt is rendered. The original
duplicate count and every source/document reference remain intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def install_dedup_patch(summarizer, audit_path, lineage_id, context=lambda: None):
    original = summarizer._summarize_by_cluster_if_needed

    async def deduplicated(self, descriptions):
        if not isinstance(descriptions, list) or any(not isinstance(x, str) for x in descriptions):
            raise ValueError('Entity descriptions must be a list of strings')
        unique = list(dict.fromkeys(descriptions))
        if len(unique) != len(descriptions):
            event = {'time': time.time(), 'kind': 'exact_entity_description_dedup',
                     'lineage_id': lineage_id, 'context': context(),
                     'input_count': len(descriptions), 'unique_count': len(unique),
                     'input_sha256': sha256(descriptions), 'unique_sha256': sha256(unique),
                     'input': descriptions, 'retained': unique,
                     'unique_values_preserved': set(descriptions) == set(unique),
                     'generation_fields_unchanged': True}
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open('a') as output:
                output.write(json.dumps(event, ensure_ascii=False) + '\n')
        return await original(self, unique)

    summarizer._summarize_by_cluster_if_needed = deduplicated
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--lineage', type=Path, required=True)
    parser.add_argument('worker', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    worker = args.worker[1:] if args.worker[:1] == ['--'] else args.worker
    if worker[:3] != ['-m', 'rumba_bench.ragu_harness', 'run']:
        raise ValueError('Only an explicit RAGU run command may be adapted')
    def value(flag):
        return worker[worker.index(flag) + 1]
    if value('--model') != 'meno-lite-0.2' or value('--suite') != 'full':
        raise ValueError('This repair is scoped to the authorized Meno0.2 full recovery')
    config_path = Path(value('--config')).resolve()
    config = json.loads(config_path.read_text())
    lineage = json.loads(args.lineage.read_text())
    if lineage['adapter_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Recovery adapter differs from its audited source')
    if lineage['config_sha256'] != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise ValueError('Recovery configuration differs from its audited baseline')
    protocol = config['protocol']
    if (config.get('judge') is not None or protocol['temperature'] != 0
            or protocol['seed'] != 42 or protocol['repetition_penalty'] != 1.03):
        raise ValueError('Recovery must keep the authorized no-judge generation protocol')
    from ragu.common.global_parameters import Settings
    from ragu.graph.artifacts_summarizer import EntitySummarizer
    from rumba_bench import ragu_harness as harness

    install_dedup_patch(EntitySummarizer, args.audit, lineage['lineage_id'],
                        context=lambda: str(Settings.storage_folder))
    original_build = harness.Harness.build

    async def audited_build(self, graph, user, sessions, **kwargs):
        result = await original_build(self, graph, user, sessions, **kwargs)
        target = Path(Settings.storage_folder) / 'pipeline-repair.json'
        target.write_text(json.dumps({**lineage, 'user': user, 'built_at': time.time(),
                                      'dedup_audit': str(args.audit)}, ensure_ascii=False,
                                     indent=2) + '\n')
        return result

    harness.Harness.build = audited_build
    sys.argv = ['rumba_bench.ragu_harness', *worker[2:]]
    harness.main()


if __name__ == '__main__':
    main()
