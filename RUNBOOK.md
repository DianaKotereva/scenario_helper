# Runbook

## Быстрый старт

1. Поднять сервисы:
```powershell
docker compose up -d --build
```

2. Инициализировать индекс OpenSearch (идемпотентно):
```powershell
docker exec -e PYTHONPATH=/app scenario_helper-frontend-1 python tools/init_db.py
```

3. Проверить, что сервисы живы:
```powershell
docker compose ps
```

4. Проверить smoke-test агента:
```powershell
docker exec -e PYTHONPATH=/app scenario_helper-frontend-1 python -c "from src.agent.agent_graph.agent_graph import app; s=app.invoke({'user_question':'Кто такой Малекит?'}); print(len(str(s.get('final_answer',''))))"
```

## Повторная инициализация индекса

- Если индекс уже заполнен, `tools/init_db.py` ничего не делает.
- Принудительная пересборка:
```powershell
docker exec -e PYTHONPATH=/app scenario_helper-frontend-1 python tools/init_db.py --force-reload
```

## Остановка

```powershell
docker compose down
```

## Debug Preprocess (до 5 глав)

```powershell
python -m tools.preprocess_book.main --book-path data/Ten-i-Plama.txt --debug
```

Полезные флаги:
- `--debug` — включает быстрый режим и ограничивает обработку максимум 5 глав.
- `--max-chapters N` — явный лимит глав; в debug режиме всё равно не больше 5.
