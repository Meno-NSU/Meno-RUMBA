#!/usr/bin/env python3
"""Render a public benchmark README from aggregate cohort metrics only."""
import argparse
import json
from pathlib import Path

LABELS = {'meno02_shared': 'Общие 345 вопросов Meno-lite-0.2',
          'qwen_shared': 'Общие 1026 вопросов Qwen',
          'available': 'Все доступные ответы каждой модели'}


def sanitize(report):
    result = {'status': report['status'], 'partial_dataset': True, 'expected_test_questions': 1543, 'cohorts': {}}
    for name, cohort in report['cohorts'].items():
        saved = {'status': cohort['status'], 'paired': cohort['paired'], 'models': {}}
        for model, metrics in cohort['models'].items():
            def clean(values):
                out = {k: values.get(k) for k in ['n', 'judged', 'rumba_metric', 'f1']}
                if out['judged'] != out['n']: out['rumba_metric'] = None
                return out
            saved['models'][model] = {'overall': clean(metrics['overall']),
                                     'temporal_x_session_scope': {g: clean(v) for g, v in metrics['temporal_x_session_scope'].items()}}
        result['cohorts'][name] = saved
    return result


def render(report):
    lines = ['# RUMBA RU: результаты бенчмаркинга', '',
             'Прогоны остановлены по времени. Это частичные результаты test: доступны 1508 ответов Meno-lite-0.1, 1026 Qwen2.5-7B-Instruct и 345 Meno-lite-0.2 из 1543 вопросов.', '',
             'Общие выборки выровнены по точным ID вопросов. Вопросы Meno-lite-0.2 полностью представлены у двух других моделей; вопросы Qwen полностью представлены у Meno-lite-0.1. Эти выборки состоят из завершённых пользовательских графов и не являются случайной выборкой полного test.', '',
             'Генерация: greedy, temperature=0, seed=42, repetition_penalty=1.03, FP16. Native-контекст 32768; отдельный YaRN-контекст 65536 используется только для длинных запросов. Извлечение — до 4096 токенов, ответ — до 256.', '',
             'Память: RAGU TwoStageArtifactsExtractorLLM с онтологией NEREL, BM25+dense, local search и реранкер. Судья — ai-forever/pollux-judge-32b; используются официальный RU judge_template и разбор rumba_metric. F1 — официальный вариант LightEval 0.13.0, без нормализации.', '',
             'Для Meno-lite-0.2 применялись исправления суммаризации описаний и дополнительные реплики генератора. Для Meno-lite-0.1 сохранялась семантика JSON при экранировании управляющих символов. Условия исполнения и исправления пайплайна следует учитывать при интерпретации: одинаковые вопросы сами по себе не устраняют эти различия.', '',
             'Каждая пара модель–вопрос оценивается один раз. Готовые оценки переиспользуются в трёх отчётах. Числовой RUMBA score показывается только после оценки всей указанной выборки; промежуточное число оценённых ответов видно в колонке «Оценено».', '']
    for name, cohort in report['cohorts'].items():
        lines.extend(['## ' + LABELS.get(name, name), '',
                      'Сравнение на одинаковых вопросах.' if cohort['paired'] else 'Независимые выборки разного размера; этот раздел не является парным сравнением.', '',
                      '| Модель | n | Оценено | RUMBA | F1 |', '|---|---:|---:|---:|---:|'])
        for model, metrics in cohort['models'].items():
            m = metrics['overall']; score = f"{m['rumba_metric']:.4f}" if m['rumba_metric'] is not None else 'ожидается'
            lines.append(f"| {model} | {m['n']} | {m['judged']}/{m['n']} | {score} | {m['f1']:.4f} |")
        lines.extend(['', '### Temporal × session-scope', '', '| Категория | Модель | n | Оценено | RUMBA | F1 |', '|---|---|---:|---:|---:|---:|'])
        for group in ['atemporal × single', 'atemporal × multi', 'temporal × single', 'temporal × multi']:
            for model, metrics in cohort['models'].items():
                if group not in metrics['temporal_x_session_scope']: continue
                m = metrics['temporal_x_session_scope'][group]; score = f"{m['rumba_metric']:.4f}" if m['rumba_metric'] is not None else 'ожидается'
                lines.append(f"| {group} | {model} | {m['n']} | {m['judged']}/{m['n']} | {score} | {m['f1']:.4f} |")
        lines.append('')
    lines.append('Сырые диалоги, личные факты, ответы моделей и причины судейских оценок в репозиторий не публикуются. Здесь сохранены только агрегаты.')
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports', type=Path, required=True)
    p.add_argument('--readme', type=Path, required=True)
    p.add_argument('--aggregates', type=Path, required=True)
    a = p.parse_args(); report = sanitize(json.loads(a.reports.read_text()))
    a.readme.parent.mkdir(parents=True, exist_ok=True); a.aggregates.parent.mkdir(parents=True, exist_ok=True)
    a.readme.write_text(render(report))
    a.aggregates.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__': main()
