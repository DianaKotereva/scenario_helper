# Prompt Debug Module

Этот модуль нужен для поэтапной диагностики качества ответа агента:
- логирует выходы этапов (`reasoning`, `question_generation`, `retrieval`, `final_aggregation`);
- сохраняет сырые артефакты по каждому вопросу;
- классифицирует, на каком этапе вероятнее всего ошибка, включая кейсы нерелевантного retrieval/graph контента.

## Запуск

Из `scenatio_helper`:

```bash
python tools/prompt_debug/run_full_diagnostics.py --only-with-gold
```

Опции:
- `--limit N` — ограничить количество вопросов;
- `--xlsx test_data/test_questions.xlsx` — путь к корзине;
- `--output-dir test_data/results` — корневая папка результатов.
- `--no-llm-judge` — отключить LLM-as-a-judge (по умолчанию включен).

## Результаты

Скрипт создает папку вида `test_data/results/prompt_debug_YYYYMMDD_HHMMSS`:
- `events.jsonl` — все stage-события;
- `cases/row_*.json` — детально по каждому вопросу (вопрос, ответ, контекст этапов, анализ);
- `summary.csv` — компактная таблица;
- `analysis.json` — агрегация классификации ошибок;
- `SUMMARY.md` — читаемый отчет с проблемными этапами.

По умолчанию итоговая стадия ошибки выбирается с учетом `LLM-as-a-judge` (если confidence >= 0.7).
