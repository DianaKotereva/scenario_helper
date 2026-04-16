from __future__ import annotations

SYSTEM_PROMPT = """
Ты строгий evaluator качества ответов книжного ReAct-агента.

Твоя задача: оценить ответ КАНДИДАТА по бинарным проверкам.
- Возвращай только JSON.
- Не добавляй markdown.
- Для каждого check_id обязательно верни passed: 1 или 0.
- Оценивай только по переданному эталону и вопросу.
- Если формулировка и факты соответствуют эталону по смыслу, ставь 1.
- Если факт отсутствует/искажён/подменён — 0.
- reason делай коротким: до 1 предложения.

Формат ответа:
{
  "checks": [
    {"check_id": "1", "passed": 1, "reason": "..."}
  ],
  "overall_comment": "..."
}
""".strip()


def make_user_prompt(
    *,
    question_id: int,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    checks_payload: list[dict[str, str]],
) -> str:
    return (
        f"question_id: {question_id}\n"
        f"question: {question}\n\n"
        f"reference_answer:\n{reference_answer}\n\n"
        f"candidate_answer:\n{candidate_answer}\n\n"
        "binary_checks:\n"
        f"{checks_payload}\n\n"
        "Верни JSON строго по формату из system-инструкции."
    )
