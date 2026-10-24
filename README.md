# Meno-RUMBA

Бенчмарк памяти Meno-Lite-0.1, Meno-Lite-0.2 и Qwen3.5-9B на русском `test` [RUMBA](https://github.com/ai-forever/RUMBA): 1 543 вопроса. Основной эксперимент использует настоящий [RAGU](https://github.com/RaguTeam/RAGU), подключённый субмодулем в `third_party/RAGU` на ревизии `abd3f2958c186dc624bbd42560ad6e1e7033efaf`. Qwen3.5-9B заменяет ранее запланированный Qwen2.5-7B-Instruct; исторический отчёт Meno-Lite-0.2 сохраняет исходные метаданные.

Истории → `TwoStageArtifactsExtractorLLM` с NEREL → граф и гибридный индекс BM25 + BGE-M3 → `LocalSearchEngine` с BGE reranker → ответ оцениваемой модели. Все нейросетевые модели работают на удалённых серверах. Для каждой оцениваемой модели извлечение выполняется независимо; каждый пользователь имеет отдельный граф.

Пилоты Meno-Lite-0.2 и Qwen3.5-9B завершены на одинаковых 50 вопросах: greedy, seed 42, repetition_penalty 1.03; F1 — 0.1902 и 0.3018 соответственно. Ошибок извлечения и обрезанных ответов — 0. Официальный rumba_metric ожидает судью после пилота Meno-Lite-0.1. [Общая таблица и оценка времени полного прогона](reports/ragu-pilot-comparison.md), [Meno-Lite-0.2](reports/ragu-pilot-meno-lite-0.2.md), [Qwen3.5-9B](reports/ragu-pilot-qwen3.5-9b.md).

## Установка и запуск

```sh
git clone --recurse-submodules https://github.com/Meno-NSU/Meno-RUMBA.git
cd Meno-RUMBA
# Для уже существующей копии: git submodule update --init --recursive
uv venv --python 3.12 .venv-ragu
uv pip install --python .venv-ragu/bin/python -r requirements-ragu.lock
cp config.ragu.example.json config.ragu.json
# Укажите реальные адреса генератора, эмбеддера и реранкера в config.ragu.json.
.venv-ragu/bin/python -m rumba_bench.cli prepare --config config.example.json
.venv-ragu/bin/python -m rumba_bench.ragu_harness preflight
.venv-ragu/bin/python -m pytest -q
.venv-ragu/bin/python -m rumba_bench.ragu_harness smoke
.venv-ragu/bin/python -m rumba_bench.ragu_harness run --suite pilot
# После проверки пилота:
.venv-ragu/bin/python -m rumba_bench.ragu_harness run --suite full
```

Реальный конфиг исключён из Git; пример содержит localhost. RAGU требует отдельное окружение из-за ограничений pandas. `requirements-ragu.lock` фиксирует проверенные версии; `requirements-ragu.txt` позволяет разрешить зависимости заново.

### Контекст и YaRN

Короткие запросы обслуживает отдельный экземпляр модели без YaRN; длинные направляются в другой экземпляр тех же весов. Для Meno на Qwen2 используется YaRN ×2 только у длинного endpoint. Для Qwen3.5-9B оба лимита 32768/65536 помещаются в штатное окно 262144, поэтому оба endpoint работают без YaRN. Общий бюджет запроса остаётся одинаковым для трёх моделей. Историческое имя маршрута `yarn` в текущем JSON означает второй endpoint; фактический RoPE-профиль берётся из команды запуска и `long_context_models.*.rope_profile`.

`models` содержит обычные endpoint, `long_context_models` — длинные endpoint с тем же served model name и теми же весами. Клиент вызывает `/tokenize` обычного сервера с реальными chat messages и generation prompt, кэширует точный счётчик и прибавляет `max_tokens`:

- бюджет ≤ 32 768 → обычный сервер;
- 32 768 < бюджет ≤ 65 536 → длинный сервер;
- бюджет > 65 536 → явная ошибка, без скрытого усечения.

Это правило действует на извлечение, проверки, суммаризацию и ответы. В ответах и отчётах сборки сохраняется выбранный маршрут. `protocol.context_routing` фиксирует границы одинаково для всех трёх моделей; кэши обычного и YaRN-сервера разделены по endpoint и RoPE-профилю.

При транспортном сбое `/tokenize` клиент повторяет запрос согласно `runtime.max_retries` (по умолчанию 2 повтора). После исчерпания повторов он возвращает ошибку; маршрут выбирается только по успешно полученному точному числу токенов.

Все серверные скрипты находятся в репозитории, в `scripts/`, и жёстко задают `CUDA_VISIBLE_DEVICES=1` до загрузки конфигурации и при каждом запуске vLLM. Переменная `GPU_ID` не используется. Файлы скачиваются на сервер непосредственно с GitHub; копировать их с локальной машины или клонировать репозиторий не требуется.

- [launch_meno2.sh](scripts/launch_meno2.sh) — короткая команда запуска текущего профиля Meno.
- [launch_meno1.sh](scripts/launch_meno1.sh) — Meno-Lite-0.1 из `../models/meno-lite-0.1`, served name `meno-lite-0.1`.
- [launch_qwen35_9b.sh](scripts/launch_qwen35_9b.sh) — Qwen3.5-9B из `../models/Qwen3.5-9B`, served name `Qwen3.5-9B`.
- [restart_meno2_split_context.sh](scripts/restart_meno2_split_context.sh) — окружение, checkpoint и PID-файлы Meno-Lite-0.2.
- [launch_split_context.sh](scripts/launch_split_context.sh) — обычный и YaRN экземпляры одних весов.
- [launch_reranker_8192.sh](scripts/launch_reranker_8192.sh) — перезапуск реранкера с контекстом 8192.
- [launch_embedder_8192.sh](scripts/launch_embedder_8192.sh) — BGE-M3 dense embeddings, порт 9023, контекст 8192.
- [server_process.py](scripts/server_process.py) — состояние процесса, очистка устаревших PID и проверка портов.
- [server_logging.sh](scripts/server_logging.sh) — логирование этапов и ошибок запуска.

В серверном каталоге, где уже находится `vllm_0_29_0/`, выполните:

```sh
(
set -e
LAUNCHER_REF=main
LAUNCHER_API=https://api.github.com/repos/Meno-NSU/Meno-RUMBA/contents/scripts
for script in server_process.py server_logging.sh launch_split_context.sh restart_meno2_split_context.sh launch_meno2.sh launch_meno1.sh launch_qwen35_9b.sh launch_reranker_8192.sh launch_embedder_8192.sh; do
  curl -4 --http1.1 -fLsS --retry 4 --retry-all-errors \
    -H 'Accept: application/vnd.github.raw+json' \
    "$LAUNCHER_API/$script?ref=$LAUNCHER_REF" -o "$script.new"
  case "$script" in
    *.sh) bash -n "$script.new" ;;
    *.py) python -m py_compile "$script.new" ;;
  esac
  command mv -f "$script.new" "$script"
done
CUDA_VISIBLE_DEVICES=1 bash launch_embedder_8192.sh
CUDA_VISIBLE_DEVICES=1 bash launch_reranker_8192.sh
CUDA_VISIBLE_DEVICES=1 bash launch_meno2.sh
)
```

По умолчанию веса находятся в `../models/meno-lite-0.2`, `../models/bge-reranker-v2-m3` и `../models/bge-m3` относительно серверного каталога запуска. Другие пути задаются через `MENO2_MODEL_PATH`, `RERANKER_MODEL_PATH`, `EMBEDDER_MODEL_PATH`, `VLLM_ENV_DIR`, `VLLM_LAUNCH_DIR`. Обёртки находят остальные скрипты рядом с собой. Для реранкера используется `vllm_process_reranker.txt`, для эмбеддера — `vllm_process_embedder.txt`; если прежний PID записан в другом файле, задайте `RERANKER_PID_FILE` или `EMBEDDER_PID_FILE`. Неучтённые слушатели скрипты не завершают, а выдают явную инструкцию.

Для переключения генератора выполните **одну** из команд: `CUDA_VISIBLE_DEVICES=1 bash launch_meno1.sh` или `CUDA_VISIBLE_DEVICES=1 bash launch_qwen35_9b.sh`. Пути можно переопределить через `MENO1_MODEL_PATH` и `QWEN35_MODEL_PATH`. После проверки локального checkpoint общий launcher останавливает только записанные PID трёх известных генераторов на 9022/9025; работающие эмбеддер и реранкер сохраняются. Общие sampling defaults заданы серверу явно: greedy, seed 42, repetition_penalty 1.03. Предел CUDA graph capture совпадает с max-num-seqs (16/4).

Qwen3.5 запускается в текстовом режиме (`--language-model-only`) с `--reasoning-parser qwen3` и `--default-chat-template-kwargs '{"enable_thinking":false}'`, чтобы JSON-извлечение и короткие ответы работали без thinking. Для его длинного endpoint в конфиге укажите `rope_profile: "native-65536-no-scaling"`. Источники: [конфигурация Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json), [рекомендации vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html#configuration-tips).

Лончер различает работающий процесс, zombie и устаревший PID. Для новых запусков рядом с PID-файлом сохраняется `.identity.json` с временем рождения процесса: переиспользованный PID не считается прежним сервисом. Старые записи очищаются автоматически; неизвестный живой процесс не завершается. Перед запуском проверяется, что нужные порты свободны.

Логи дописываются при повторных попытках:

- `vllm_meno2_launcher.log` — этап, время UTC, GPU 1, команда и причина ошибки запуска Meno;
- `vllm_meno2.log`, `vllm_meno2_yarn.log` — вывод соответствующего процесса vLLM;
- `vllm_meno1_launcher.log`, `vllm_meno1.log`, `vllm_meno1_long.log` — Meno-Lite-0.1;
- `vllm_qwen35_launcher.log`, `vllm_qwen35.log`, `vllm_qwen35_long.log` — Qwen3.5-9B;
- `vllm_reranker_launcher.log`, `vllm_reranker.log` — запуск и вывод реранкера.
- `vllm_embedder_launcher.log`, `vllm_embedder.log` — запуск и вывод эмбеддера.

При ошибке последние 60 строк модельных логов включаются в лог лончера. Если vLLM завершился во время запуска, сохраняется его код завершения. Вывод также идёт в терминал. Для диагностики Meno:

```sh
tail -n 200 vllm_meno2_launcher.log
```

Для Qwen2/Qwen2.5 с исходным контекстом 32 768 общий launcher проверяет исходный конфиг, сохраняет checkpoint `rope_theta` и запускает обычный сервер на 9022 и YaRN на 9025 последовательно. В vLLM 0.29 используется `--hf-overrides` с `rope_parameters`; старый `--rope-scaling` больше не поддерживается. Исходный config.json не изменяется. Пример вызова с собственными путями:

```sh
VLLM_BIN=/path/to/venv/bin/vllm \
NATIVE_PID_FILE=/path/to/native.pid YARN_PID_FILE=/path/to/yarn.pid \
bash launch_split_context.sh /path/to/checkpoint meno-lite-0.2
```

Стартовый бюджет для H100 80GB: 0.40 GPU памяти обычному экземпляру, 0.45 — YaRN, 0.04 — эмбеддеру и 0.06 — реранкеру; сумма долей — 0.95 всей карты. Клиентский параллелизм — 8; серверные `max-num-seqs` — 16 у обычного Meno и 4 у остальных сервисов. Это стартовые параметры для измерения, не обещание линейного ускорения; фактическая память и возможные ошибки видны в логах лончеров.

Реранкеру нужен `--max-model-len 8192`: развёртывание с лимитом 512 принимало короткий smoke, но отклоняло реальные source chunks. Эмбеддеру также нужен контекст 8192. Теперь `preflight` проверяет лимиты всех четырёх сервисов и автоматически выполняется до `smoke`, `index` и `run`, чтобы выявлять такие проблемы до извлечения историй.

Для эмбеддера используется served name `bge-m3`, `/v1/embeddings`, режим `pooling`, задача `embed`, CLS pooling и L2-нормализация. BGE-M3 имеет штатное окно 8192 и размерность dense-вектора 1024; YaRN для него не нужен. Параметры: [карточка BGE-M3](https://huggingface.co/BAAI/bge-m3), [pooling в vLLM 0.29](https://docs.vllm.ai/en/v0.29.0/models/pooling_models/).

Источники: [Qwen2.5: расширение контекста и статический YaRN](https://qwen.readthedocs.io/en/v2.5/deployment/vllm.html#extended-context-support), [vLLM 0.29: актуальный синтаксис](https://docs.vllm.ai/en/v0.29.0/features/context_extension/).

Для основного `bounded`-прогона добавьте при запуске каждого оцениваемого vLLM один и тот же серверный флаг:

```sh
--structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
```

Это необходимо из-за зацикливания на whitespace в структурированных ответах. В [vLLM 0.29.0](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/structured_output/backend_xgrammar.py) этот параметр читается из серверной конфигурации, а не из запроса. `protocol.server_structured_outputs` записывает фактически применённый профиль сервера в протокол и ключ кэша, но сам сервер не перенастраивает. При изменении серверного декодера обновите профиль: старые обрезанные ответы нельзя переиспользовать. `/server_info` на проверенном сервере недоступен, поэтому установку флага подтверждает оператор; затем нужны smoke и регрессии. Эмбеддер и реранкер перезапускать не требуется.

`preflight` проверяет имена трёх сервисов, размерность эмбеддингов и порядок результатов реранкера. Штатный `ScorerOpenAI` использует `/v1/score` (`text_1`, `text_2`); у проверенного vLLM этот интерфейс совместим с `/v1/rerank`. Без реранкера нет скрытого перехода к поиску без него. `smoke` проверяет две синтетические сессии: состоявшуюся покупку пастели и ещё не состоявшуюся покупку акварели.

Пилот зафиксирован в `data/pilot50.json`: 50 вопросов, seed 42, пользователи `user_44`, `user_61`, `user_65`, 155 сессий, по 12–13 вопросов в четырёх категориях temporal × session-scope. Это отладочный пилот, не репрезентативная оценка test. Данные закреплены на `d0480d86d4ef20701d8538dd3d50528ee065ab63`, официальные функции RUMBA — на `a20471ab5bb9d6d0b3bc8a263495a799868e1a3b`.

Для остальных колонок задайте endpoint и повторите команду с `--model meno-lite-0.1` или `--model qwen3.5-9b` (served name `Qwen3.5-9B`). Модели можно разворачивать последовательно на одном GPU. Память одной модели не используется для другой. Эталоны, evidence и категории не передаются в извлечение или запрос модели; evidence используется только для диагностики.

## Протокол RAGU

- Greedy: `temperature=0`, `top_p=1`, `top_k=-1`, `min_p=0`, seed 42, `repetition_penalty=1.03`, остальные penalties — 0. Все флаги явно передаются vLLM во всех вызовах извлечения, суммаризации и ответа; для трёх моделей протокол одинаковый.
- Настоящий `TwoStageArtifactsExtractorLLM`: извлечение и LLM-проверка сущностей, извлечение и LLM-проверка связей. Программная проверка NEREL также включена; сигнатуры, отсечение неприменимых типов и диагностика онтологии сохранены. Промпты проверки связей, превышавшие 30 000 токенов, направляются в YaRN по описанному выше правилу. Этот выбор одинаков для трёх моделей.
- Основной вариант `extraction_decoding=bounded`: JSON-схема upstream с максимумом 16 сущностей/связей, именами до 80 и описаниями до 200 символов; все объявленные ключи обязательны, дополнительные ключи запрещены. Это закрывает случай `{}`, который исходная схема с default-полями молча превращает в пустые списки. Пустой список по-прежнему допустим. Серверный профиль выше ограничивает whitespace. Пределы могут снижать полноту памяти, поэтому они явно входят в протокол.
- Для сравнения доступны `native` — исходная JSON-схема без дополнительных лимитов, и `free` — свободный JSON с последующим разбором Pydantic. В начальных реальных пробах `native` повторял сущности, а `bounded` без серверного флага иногда зацикливался на пробелах между полями. Режимы и лимиты входят в хеш протокола и не смешиваются.
- Экспериментальный `grammar`: клиентская EBNF по полям и типам upstream Pydantic, максимум 16 сущностей/связей, конечный whitespace (`grammar_whitespace_chars`), фиксированный порядок полей, без дополнительных ключей. Строки не ограничены длиной. Он прошёл полный smoke без JSON-ошибок, но на проверенном сервере занял около 245 секунд на две сессии, а отдельные проблемные запросы — около 100 секунд; для основного пилота не выбран. Пределы строк используются только в `bounded`; пределы списков — в `bounded` и `grammar`.
- ICL: один штатный пример RAGU, стратегия `hybrid`. Начальный smoke с двумя примерами переполнил контекст 30 000 токенов при проверке связей; он не используется в пилоте.
- Входной адаптер сохраняет весь исходный диалог с ролями: фрагменты до 3 500 символов, перекрытие 200. Каждый содержит user ID, session ID, дату, номер и роль, внутри которой он начинается. В RAGU они передаются как готовые chunks (`chunker=None`), чтобы метаданные не терялись при последующем разбиении.
- Суммаризация описаний включена (порог 7), clustering и community summaries отключены. Изолированные сущности сохраняются, чтобы отсутствие допустимой связи не удаляло сущность с личным фактом. Исходные chunks хранятся независимо от успеха извлечения.
- NetworkX + JSON KV + штатный Qdrant adapter в локальном дисковом режиме. Гибрид: лексический BM25 и удалённый BGE-M3, 1024 измерения. Каждая Qdrant-коллекция имеет отдельный каталог для исключения конфликтов блокировок.
- `LocalSearchEngine`: `top_k=20`, после rerank остаются 5 seed entities; source chunks включены, summaries отключены. Штатный scorer реранжирует entities, relations и chunks. Бюджет контекста — 12 000 токенов `tiktoken/gpt-4o` (прокси, не токенизатор оцениваемой модели). Фактический промпт сохраняется, чтобы выявлять выпадение chunks при усечении.
- Официальный `ANSWER_PROMPT_RU` задаётся через `engine.update_prompt`; дата вопроса передаётся явно. Лимиты: извлечение/суммаризация 4 096 токенов, ответ 256, stop `\n`. Обрезанный ответ сохраняется без исправления и помечается `truncated`; неполный структурированный результат считается ошибкой.

Транспортный адаптер интерфейса RAGU `LLM` нужен для явной передачи `seed`, `repetition_penalty` и ограничений декодера: штатный `CachedAsyncOpenAI.chat_completion` закреплённой ревизии не принимает эти параметры. Сам extractor, граф, поиск и реранжирование остаются upstream; файлы субмодуля не изменяются. EBNF строится для небольшой части JSON Schema, используемой закреплёнными extractor и summarizers; неподдерживаемые конструкции вызывают явную ошибку. Грамматика не исправляет и не дописывает ответы модели постфактум. Предыдущий custom harness, оба режима `free`/`structured` и эксперимент с личными фактами сохранены отдельно: [LEGACY_HARNESS.md](LEGACY_HARNESS.md). Их результаты не смешиваются с RAGU.

## Результаты и судья

Результаты находятся в `runs/ragu-<hash>/<model>/`. Хеш включает ревизию и исходники RAGU, код адаптера, протокол, данные, retrieval services и версии ключевых пакетов; endpoint модели проверяется отдельно. Повторная команда продолжает по графам, ответам и точному кэшу запросов. Изменение протокола создаёт отдельный запуск.

- `graphs/<user>/build_report.json`: исходные chunks, артефакты, ошибки LLM, решения онтологии, согласованность хранилищ.
- `retrieval/pilot/*.json`: найденный контекст, chunk IDs и фактически отправленные сообщения.
- `answers/pilot/*.json`: ответы, finish reason, категории, полнота evidence.
- `pilot/report.json`, `pilot/predictions.json`, `pilot/qc.json`: F1, temporal × session-scope, полнота ответов, обрезания, ошибки извлечения и видимые в промпте evidence.
- `comparison-pilot.md` и CSV в родительской папке: все три колонки; неизмеренные результаты помечены как ожидающиеся.
- `runs/cache/ragu/`: исходные генерации и кэш эмбеддингов/score. Данные, рабочие адреса и runtime-артефакты не публикуются в Git.

`retrieval_recall` считает найденные сессии evidence; `visible_evidence_recall` — сессии с хотя бы одним полностью вошедшим в промпт chunk. Это диагностика источников, не семантическая оценка ответа. Согласованность графа не означает полноту извлечённых фактов: ошибки LLM и отбрасывания онтологии учитываются отдельно.

F1 повторяет LightEval 0.13.0 из официального окружения: множества слов после `split()`, с сохранением регистра и пунктуации. Поэтому дата в ISO и та же дата словами могут получить 0. `rumba_metric` считается независимым `ai-forever/pollux-judge-32b`: точные `judge_template`, `process_judge_response`, `extract_judge_feedback` загружаются из закреплённого RU-модуля без локальных весов. Вердикты 0/1 → 0, 2/3 → 1; ошибка разбора → 0 с отдельным флагом, как у авторов.

Pollux поднимается отдельным этапом: сначала сохраняются ответы всех трёх моделей, затем обычный/YaRN генераторы, эмбеддер и реранкер освобождают GPU, и на их месте запускается судья. Для `score` и `report` эти inference services уже не требуются; сохранённые извлечения и ответы повторять не нужно. Заполните `judge` в локальном конфиге (`base_url`, `model`, `source_model: ai-forever/pollux-judge-32b`) и передайте фактический каталог сохранённого запуска:

```sh
.venv-ragu/bin/python -m rumba_bench.ragu_harness score --suite pilot \
  --run-folder runs/ragu-HASH/meno-lite-0.2
# Пересобрать отчёт без обращений к сервисам:
.venv-ragu/bin/python -m rumba_bench.ragu_harness report --suite pilot \
  --run-folder runs/ragu-HASH/meno-lite-0.2
```

Без Pollux статус — `judge_pending`; F1 не подменяет официальный rumba_metric.

## Локальные расходы

Окружение RAGU занимает около 915 МБ, данные — 53 МБ. PyTorch, CUDA и веса генератора/эмбеддера/реранкера локально не устанавливаются. Здесь выполняются HTTP, BM25, операции над графом и локальный Qdrant. Кэш и графы увеличивают расход диска по мере обработки. Текущий протокол требует четырёх LLM-вызовов на chunk плюс суммаризация: для 411 chunks пилота это 1 644 базовых вызова, затем 50 ответов и 50 вызовов судьи. Параллелизм задаётся `runtime.concurrency`; в примере — 8 запросов. Время полного прогона следует оценивать по фактическому времени пилота.

Для проверки декодера на ранее сохранённых проблемных запросах:

```sh
.venv-ragu/bin/python scripts/probe_ragu_decoding.py runs/cache/ragu/generation/RESPONSE.json \
  --output runs/decoder-regression.json
```

Скрипт повторяет исходные сообщения с текущими флагами и схемой, сохраняет число артефактов и исходы. Ошибка или пустой результат явно проваливает такую регрессию; эти диагностические ответы не подменяют ответы пилота.

Источники и лицензии: [RAGU](https://github.com/RaguTeam/RAGU), [RUMBA](https://github.com/ai-forever/RUMBA), [данные](https://huggingface.co/datasets/ai-forever/RUMBA), [LightEval](https://github.com/huggingface/lighteval/blob/v0.13.0/src/lighteval/metrics/metrics_sample.py). Лицензии снимков — в `vendor/`, лицензия RAGU — внутри субмодуля.
