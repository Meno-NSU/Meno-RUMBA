# Полный RUMBA на другом кластере с A100

Этот runbook запускает тот же RAGU-пайплайн, что в пилотах, на **всех 1543 вопросах** русского `test`: сначала память, поиск и ответы без судьи; отдельным запуском — Pollux над сохранёнными ответами. Для каждого кандидата строятся собственные графы. Код, данные, кэши и ответы остаются на новом кластере; соединение с прежним сервером не требуется.

Готовые файлы уже в репозитории: [workflow](scripts/cluster_a100.sh) и [шаблон Slurm](scripts/cluster_a100.sbatch). Менять пути в исходниках не нужно. Скрипты `launch_meno*.sh`, `launch_pollux_judge.sh` и другие прежние launchers предназначены для прежнего сервера; под Slurm используйте файлы из этого runbook.

## 1. Ресурсы и выбор GPU

| Профиль `A100_LAYOUT` | Выделение на одном узле | Первая стадия | Судейская стадия |
|---|---|---|---|
| `single80` | 1 × A100 80 GB | Два экземпляра кандидата + embedder + reranker на одной карте | Pollux BF16, TP=1 |
| `dual40` | 2 × A100 40 GB | Native + embedder на первой выделенной карте; long + reranker на второй | Pollux BF16, TP=2 на обеих картах |

**Одна A100 40 GB не является поддержанным профилем этого runbook.** Для двух копий кандидата с окнами 32K/64K и для Pollux 32B BF16 требуется другой объём ресурсов. Квантизация и уменьшение контекста здесь не подменяют исходный протокол. Для `dual40` обе карты должны находиться на одном узле; предпочтительно соединение NVLink.

При ручном запуске workflow всегда задаёт `CUDA_VISIBLE_DEVICES=1`; доступен профиль `single80`. Под Slurm используется назначенный планировщиком `CUDA_VISIBLE_DEVICES`: для двух карт сервисы получают только соответствующее подмножество выделенных устройств. Не подставляйте физические номера GPU вручную внутри задания. Такое переназначение локальных индексов описано в [Slurm GRES](https://slurm.schedmd.com/gres.html#GPU_Management).

Шаблон запрашивает 16 CPU, 128 GB RAM и 48 часов; partition, account, лимит времени и тип GRES замените на разрешённые в вашем кластере. Примеры ниже предполагают, что очередь `a100` существует; это **пример имени**, а не требование. Распознавание GPU и доступная память проверяются до запуска сервисов.

Профили памяти — отправная конфигурация, а не измеренный результат на новом кластере. Проверены CPU-тесты и синтаксис; фактический запуск на A100 следует проверить коротким пилотом. Время H100 нельзя напрямую переносить на A100.

## 2. Подготовка каталога и окружений

Выполните на узле, где разрешены GitHub/Hugging Face и установка пакетов. Репозиторий, окружения, модели и `runs/` разместите в постоянном каталоге, доступном compute nodes, например в project/scratch-хранилище с достаточной квотой. Не используйте очищаемый при завершении задания `$TMPDIR` как единственное место хранения результатов. Для локального Qdrant нужна файловая система с рабочими файловыми блокировками; один и тот же граф одновременно не открывают два процесса.

```bash
git clone --recurse-submodules https://github.com/Meno-NSU/Meno-RUMBA.git
cd Meno-RUMBA
git submodule update --init --recursive

uv venv --python 3.12 --seed .venv-ragu
uv pip install --python .venv-ragu/bin/python -r requirements-ragu.lock

# Отдельное CUDA-окружение: не ставить vLLM в окружение RAGU.
uv venv --python 3.12 --seed .venv-vllm
uv pip install --python .venv-vllm/bin/python 'vllm==0.29.0'

mkdir -p runs
git rev-parse HEAD > runs/cluster-code-revision.txt
.venv-vllm/bin/python -m pip freeze > runs/cluster-server-packages.txt
```

Если в кластере уже есть подходящее venv с vLLM, вместо установки задайте `CLUSTER_VLLM_ENV=/absolute/path/to/venv`: нужны `bin/python` и `bin/vllm`. CUDA wheel должен подходить драйверу узла; используйте согласованный с администраторами CUDA module или окружение. Не заменяйте закреплённую версию и зависимости посреди серии. О совместимости wheel/PyTorch/CUDA — [официальная установка vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

Модели ожидаются в `../models/` относительно репозитория:

```text
models/
  meno-lite-0.1/
  meno-lite-0.2/
  Qwen3.5-9B/                 # только если нужен этот кандидат
  bge-m3/
  bge-reranker-v2-m3/
  pollux-judge-32b/           # нужен только второй стадии
Meno-RUMBA/
```

Если веса уже есть, достаточно `export MODELS_DIR=/absolute/path/to/models`. Папки внутри должны называться как в дереве. Для скачивания недостающих весов можно использовать следующие команды; укажите ревизии, если нужен воспроизводимый снимок:

```bash
export MODELS_DIR="$(cd .. && pwd)/models"
.venv-ragu/bin/hf download bond005/meno-lite-0.2 --local-dir "$MODELS_DIR/meno-lite-0.2"
.venv-ragu/bin/hf download bond005/meno-lite-0.1 --local-dir "$MODELS_DIR/meno-lite-0.1"
.venv-ragu/bin/hf download BAAI/bge-m3 --local-dir "$MODELS_DIR/bge-m3"
.venv-ragu/bin/hf download BAAI/bge-reranker-v2-m3 --local-dir "$MODELS_DIR/bge-reranker-v2-m3"
.venv-ragu/bin/hf download ai-forever/pollux-judge-32b --local-dir "$MODELS_DIR/pollux-judge-32b"
# Необязательный третий кандидат:
# .venv-ragu/bin/hf download Qwen/Qwen3.5-9B --local-dir "$MODELS_DIR/Qwen3.5-9B"
```

Для судьи нужен **generative `pollux-judge-32b`**, не regression-вариант с суффиксом `-r`. Диск должен вмещать выбранные веса (у одного Pollux около 65 GB в двухбайтовом формате), два Python-окружения, кэш и растущие графы/сырой вывод. Оценку квоты делайте с запасом; размер конечных графов полного прогона ещё не измерен.

## 3. Данные и BM25 до GPU-задания

```bash
.venv-ragu/bin/python -m rumba_bench.cli prepare --config config.example.json

# Подготовить полный snapshot BM25, включая файлы проверки локального кэша.
.venv-ragu/bin/python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
from ragu.models.sparse_embedder import BM25
cache = str(Path('.cache/fastembed').resolve())
snapshot_download('Qdrant/bm25', revision='22b8d2af71a76161e18dd432d2cee0eefa66e412', cache_dir=cache)
BM25(language='russian', cache_dir=cache)
print('BM25 ready')
PY

.venv-ragu/bin/python - <<'PY'
import json
from pathlib import Path
rows = [json.loads(s) for s in Path('data/questions.jsonl').read_text().splitlines()]
assert len(rows) == len({r['id'] for r in rows}) == 1543
print('Questions:', len(rows), 'users:', len({r['user_id'] for r in rows}))
PY
```

Датасет закреплён на `d0480d86d4ef20701d8538dd3d50528ee065ab63`, RAGU — `abd3f2958c186dc624bbd42560ad6e1e7033efaf`. При полностью закрытом доступе compute nodes к сети заранее проверьте загрузку BM25 именно с `HF_HUB_OFFLINE=1` и тем же `.cache/fastembed` в compute-окружении. Сам по себе этот флаг не заполняет недостающие файлы.

## 4. Первая стадия: полный прогон без судьи

Основная команда для одной A100 80 GB, **из корня репозитория**:

```bash
export MODELS_DIR=/absolute/path/to/models
export A100_LAYOUT=single80
export BENCH_SUITE=full
sbatch --partition=a100 scripts/cluster_a100.sbatch inference meno-lite-0.2
```

Вариант с двумя A100 40 GB на одном узле:

```bash
export A100_LAYOUT=dual40
export BENCH_SUITE=full
sbatch --partition=a100 --gres=gpu:2 scripts/cluster_a100.sbatch inference meno-lite-0.2
```

Добавьте `--account=...`, замените `--partition`, при необходимости используйте принятый в кластере typed GRES (`gpu:a100:2` и т. п.). `--time=...` может переопределить 48 часов шаблона. Slurm-переменные в `#SBATCH` не раскрываются как shell-переменные; параметры передавайте в командной строке `sbatch`. См. [документацию sbatch](https://slurm.schedmd.com/sbatch.html).

Без Slurm, на выделенной A100 80 GB с физическим номером 1:

```bash
export MODELS_DIR=/absolute/path/to/models
A100_LAYOUT=single80 BENCH_SUITE=full bash scripts/cluster_a100.sh inference meno-lite-0.2
```

Workflow сам поднимает четыре API на loopback своего compute node, ждёт готовности и вызывает preflight, затем `ragu_harness run --suite full`. Ручной запуск держите в `tmux`/`screen`; отдельный локальный компьютер для клиента не требуется. Не запускайте модели на login node.

Порты 9022–9026 должны быть свободны на этом узле. Если порт занят неизвестным сервисом, workflow остановится с диагностикой; чужие задания он не завершает.

| Порт | Модель / назначение | Окно |
|---|---|---:|
| 9022 | Выбранный кандидат, native | 32768 |
| 9025 | Те же веса, длинный endpoint | 65536 |
| 9023 | `bge-m3`, dense embeddings | 8192 |
| 9024 | `bge-reranker-v2-m3`, reranker | 8192 |

У Meno YaRN ×2 включён **только на 9025**; запросы с полным бюджетом ≤32768 идут на 9022 без YaRN. У Qwen3.5-9B оба окна native; thinking выключен. Лимит считается точным `/tokenize` вместе с резервом ответа. Превышение 65536 вызывает ошибку, а не скрытое усечение.

Протокол кандидата: greedy, seed 42, `repetition_penalty=1.03`; извлечение максимум 4096 токенов, ответ максимум 256; прежние NEREL/ICL/TwoStage/BM25+dense/local search/reranker. В кластере `runtime.concurrency=4`, серверные `max-num-seqs=4`, `max-num-batched-tokens=8192` одинаковы для кандидатов. Это профиль ресурсов нового запуска; параметры декодирования пилотов сохранены.

Для предварительной проверки нового окружения замените `BENCH_SUITE=full` на `BENCH_SUITE=pilot` и запустите тот же workflow: 50 вопросов, затем проверьте QC. Следующий `full` содержит **все 1543**, включая эти 50; это не добавочная выборка из 1493. Готовые графы пользователей и кэш могут переиспользоваться.

Для Meno 0.1 замените последний аргумент на `meno-lite-0.1`, для Qwen3.5 — на `qwen3.5-9b`. Модели запускаются последовательно. При планировании нескольких заданий используйте зависимости `--dependency=afterok:JOB_ID`, чтобы задания не открывали один и тот же локальный кэш/граф одновременно. Для каждого кандидата сохраняется отдельный каталог.

## 5. Проверка и сохранение первой стадии

```bash
squeue -u "$USER"
tail -n 100 slurm-JOB_ID.out
cat runs/cluster-jobs/JOB_ID/run-folder.txt
```

Последний файл содержит точный каталог, например `.../runs/ragu-HASH/meno-lite-0.2`. **Сохраните его; не угадывайте HASH и не берите каталог другого запуска.** Сгенерированный конфиг и подробные логи находятся рядом в `runs/cluster-jobs/JOB_ID/`: `config.json`, `workflow.log`, `native.log`, `long.log`, `embedder.log`, `reranker.log`. При ручном запуске вместо JOB_ID — каталог `manual-...`.

В каталоге модели первая стадия оставляет:

- `graphs/<user>/build_report.json`, графы, индексы и provenance;
- `answers/full/*.json`: 1543 исходных ответа;
- `full/predictions.json`, `full/report.json`, `full/qc.json`;
- в родительском каталоге — `manifest.json` и `comparison-full.md/.csv`.

Успех — `INFERENCE QC PASSED: 1543 answers, judge_pending` в логе и нулевой exit code. Проверяются точные ID, число ответов, отсутствие обрезаний и ошибок LLM при извлечении, согласованность графов и сохранность исходных фрагментов. F1 уже доступен; `rumba_metric` пока пустой. Эталоны используются для оценки, а не подмешиваются в извлечение или ответы.

После окончания workflow завершает только собственные записанные PID. Под Slurm завершение задания освобождает allocation; судью запускают следующим отдельным заданием. Не используйте `pkill vllm` или убийство всех процессов GPU на общем кластере.

## 6. Вторая стадия: поднять Pollux и оценить сохранённые ответы

Для примера, первая стадия закончилась в `runs/ragu-HASH/meno-lite-0.2`. Передайте **родительский** `runs/ragu-HASH` и явно перечислите модели, которые требуется оценить:

```bash
export MODELS_DIR=/absolute/path/to/models
export A100_LAYOUT=single80
export BENCH_SUITE=full
# JOB_ID — номер успешно завершённого inference-задания.
RUN_FOLDER=$(cat runs/cluster-jobs/JOB_ID/run-folder.txt)
RUN_ROOT=$(dirname "$RUN_FOLDER")

sbatch --partition=a100 scripts/cluster_a100.sbatch \
  judge "$RUN_ROOT" meno-lite-0.2
```

Если нужен судья **только для двух Meno**, после завершения полных прогонов обеих моделей:

```bash
sbatch --partition=a100 scripts/cluster_a100.sbatch \
  judge "$RUN_ROOT" meno-lite-0.1 meno-lite-0.2
```

Модели должны иметь полные ответы в одном указанном `RUN_ROOT`. Для разных HASH запускайте отдельную оценку каждого каталога. Для `dual40` задайте `A100_LAYOUT=dual40` и добавьте `--gres=gpu:2`; судья автоматически получает `--tensor-parallel-size 2`. Эта схема поддерживается [vLLM tensor parallel serving](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/#single-node-deployment).

Без Slurm на GPU 1 / A100 80 GB:

```bash
A100_LAYOUT=single80 BENCH_SUITE=full bash scripts/cluster_a100.sh \
  judge "$RUN_ROOT" meno-lite-0.1 meno-lite-0.2
```

Workflow проверяет полноту сохранённых ответов, поднимает **только** `pollux-judge-32b` на `127.0.0.1:9026` и проверяет формат вердиктов на пяти сохранённых ответах каждой указанной модели с покрытием четырёх категорий. Эти запросы сохраняются в кэш. Только после успешного `JUDGE SMOKE PASSED` выполняется `ragu_harness score --suite full --run-folder ...` отдельно для указанных моделей. Генератор, эмбеддер и реранкер на этой стадии не запускаются. Сам Pollux использует BF16, 8192 контекста без YaRN, память 0.92, максимум четыре активных последовательности; greedy, seed 42, `repetition_penalty=1.0`, максимум 4096 токенов на вердикт. Извлечение, retrieval и ответы кандидатов повторять не требуется.

Официальные RU-промпт и parser закреплены в `vendor/rumba`: оценка 0/1 → бинарный 0, 2/3 → бинарная 1. После завершения нужны `status=scored`, `judged=1543`, `judge_parse_errors=0`; ошибки parser не скрываются как успешный QC. Артефакты: `judge-requests/` с сырым выводом и запросами, `scores/full/*.json` с вердиктами, обновлённые `full/report.json`, `full/predictions.json`, `full/qc.json`, `comparison-full.md/.csv`. Разбивка temporal × session-scope находится в `report.json`.

## 7. Повторный запуск, перенос результатов и диагностика

При ограничении времени Slurm повторите **ту же** команду с тем же кодом, весами, протоколом и каталогами. Готовые ответы и судейские оценки пропускаются, завершённые графы и генерации берутся из кэша. При смене compute node localhost-адреса остаются одинаковыми. Не меняйте содержимое весов при прежнем alias, не обновляйте зависимости и не редактируйте протокол для продолжения того же запуска.

Прерывание прямо во время построения графа требует проверки этого пользователя: незавершённое хранилище не является гарантированно атомарной контрольной точкой. Если отсутствует корректный `build_report.json` или граф несогласован, сохраните повреждённый каталог для диагностики и восстановите граф отдельно; не принимайте старые ответы по нему за проверенный результат. Аналогично, уже сохранённые ошибки разбора судьи не пересчитываются автоматически: их сначала анализируют и явно планируют повторную оценку.

При OOM смотрите конкретный `native.log`/`long.log`/`judge.log`, фактическую VRAM и layout. Для одиночной 40 GB не пытайтесь применить профиль 80 GB. Сначала можно уменьшить batching/concurrency, фиксируя одно изменение для всех кандидатов; не уменьшайте окно и не включайте YaRN на коротких промптах для скрытого обхода ошибки. После изменения протокола это отдельный эксперимент.

Не запускайте `score` по случайному HASH: `judge RUN_ROOT MODEL...` специально требует явный каталог и список. На новом кластере инструкции не оценивают Qwen автоматически.

Для продолжения работы сохраняйте весь соответствующий `runs/ragu-HASH/` вместе с manifest и `runs/cache/ragu/`; для повторной индексации также нужны `data/` и `.cache/fastembed/`. Для анализа оценок достаточно исходных ответов, `judge-requests/`, `scores/`, отчётов и manifest. Сырые ответы/истории и рабочие конфиги остаются вне Git; публикуются только проверенные агрегаты.
