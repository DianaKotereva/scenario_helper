from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import json_repair

from src.llm_core.llm_core import llm


JUDGE_PROMPT = """
Ты — строгий аудитор RAG-агента.
Твоя задача: определить, на каком этапе возникла главная ошибка ответа.

Доступные классы stage:
- retrieval: поиск не нашел нужную информацию
- retrieval_irrelevant: поиск нашел, но контент нерелевантен вопросу
- graph_irrelevant: из графа знаний пришел нерелевантный контент
- subquestions: неверная декомпозиция/подвопросы
- retrieve_generation: плохо сгенерирован промежуточный ответ по контексту
- final_aggregation: финальный промпт/агрегация испортили ответ
- runtime: техническая ошибка выполнения
- mixed: есть несколько причин, одна не доминирует
- ok: существенных проблем нет

Верни только JSON:
{
  "stage": str,
  "confidence": float,   // 0..1
  "reason": str,         // 1-2 предложения
  "retrieval_relevance": float, // 0..1, оценка релевантности retrieval-контента
  "graph_relevance": float       // 0..1, оценка релевантности graph-контента
}
"""


@dataclass
class LLMJudgeResult:
    stage: str
    confidence: float
    reason: str
    retrieval_relevance: float
    graph_relevance: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "confidence": self.confidence,
            "reason": self.reason,
            "retrieval_relevance": self.retrieval_relevance,
            "graph_relevance": self.graph_relevance,
        }


def _compact_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    retrieval = []
    graph = []
    subquestions = []
    retrieve_generation = []
    final_agg = []

    for event in events:
        stage = event.get("stage")
        payload = event.get("payload", {})
        if stage == "retrieval_search_raw":
            retrieval.append(
                {
                    "query": payload.get("query", ""),
                    "total_docs": payload.get("total_docs", 0),
                    "by_source": payload.get("by_source", {}),
                    "top_docs": payload.get("top_docs", [])[:3],
                }
            )
        elif stage == "retrieval_graph_processed":
            graph.append(
                {
                    "nodes_text_len": payload.get("nodes_text_len", 0),
                    "rels_text_len": payload.get("rels_text_len", 0),
                    "nodes_preview": payload.get("nodes_preview", ""),
                    "rels_preview": payload.get("rels_preview", ""),
                }
            )
        elif stage == "question_generation":
            subquestions.append({"questions": payload.get("questions", [])})
        elif stage == "retrieve_generation":
            retrieve_generation.append(
                {
                    "answer_len": payload.get("answer_len", 0),
                    "answer_preview": payload.get("answer_preview", ""),
                }
            )
        elif stage == "final_aggregation":
            final_agg.append(
                {
                    "context_items": payload.get("context_items", 0),
                    "thoughts_items": payload.get("thoughts_items", 0),
                    "final_answer_len": payload.get("final_answer_len", 0),
                    "final_answer_preview": payload.get("final_answer_preview", ""),
                }
            )
        elif stage.endswith("_error"):
            return {"runtime_error": {"stage": stage, "error": payload.get("error", "")}}

    return {
        "retrieval": retrieval,
        "graph": graph,
        "subquestions": subquestions,
        "retrieve_generation": retrieve_generation,
        "final_aggregation": final_agg,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except Exception:  # noqa: BLE001
        return default
    return max(0.0, min(1.0, num))


def _parse_judge_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except Exception:  # noqa: BLE001
        pass

    try:
        return json_repair.loads(content)
    except Exception:  # noqa: BLE001
        return {}


def judge_case_with_llm(
    question: str,
    expected: str,
    actual: str,
    events: List[Dict[str, Any]],
) -> LLMJudgeResult:
    compact = _compact_events(events)
    user_payload = {
        "question": question,
        "expected": expected,
        "actual": actual,
        "events": compact,
    }
    msg = (
        "Проанализируй кейс и верни JSON по формату.\n"
        + json.dumps(user_payload, ensure_ascii=False)
    )
    response = llm.invoke(
        [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": msg},
        ]
    )
    content = response.content if hasattr(response, "content") else str(response)
    parsed = _parse_judge_json(content)

    return LLMJudgeResult(
        stage=str(parsed.get("stage", "mixed")),
        confidence=_safe_float(parsed.get("confidence", 0.5), 0.5),
        reason=str(parsed.get("reason", "LLM judge did not provide a clear reason.")),
        retrieval_relevance=_safe_float(parsed.get("retrieval_relevance", 0.0), 0.0),
        graph_relevance=_safe_float(parsed.get("graph_relevance", 0.0), 0.0),
    )

