from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


RUS_STOPWORDS = {
    "это",
    "как",
    "что",
    "или",
    "для",
    "его",
    "ее",
    "они",
    "она",
    "оно",
    "при",
    "над",
    "под",
    "без",
    "про",
    "кто",
    "где",
    "когда",
    "если",
    "только",
    "очень",
    "также",
}


FAILURE_MARKERS = (
    "не удалось",
    "ошибка",
    "недостаточно информации",
    "произошла ошибка",
    "попробуйте переформулировать",
)


@dataclass
class CaseAnalysis:
    case_id: str
    stage: str
    severity: str
    confidence: float
    reasons: List[str]
    metrics: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "stage": self.stage,
            "severity": self.severity,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


def _tokenize(text: str) -> set[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[a-zа-яё0-9]{3,}", text)
    return {t for t in tokens if t not in RUS_STOPWORDS}


def _coverage(expected: str, actual: str) -> float:
    expected_t = _tokenize(expected)
    actual_t = _tokenize(actual)
    if not expected_t:
        return 0.0
    return len(expected_t.intersection(actual_t)) / max(len(expected_t), 1)


def _extract_stage_payloads(events: Iterable[Dict[str, Any]], stage: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for event in events:
        if event.get("stage") == stage:
            out.append(event.get("payload", {}))
    return out


def analyze_case(
    case_id: str,
    expected: str,
    actual: str,
    events: List[Dict[str, Any]],
    runtime_error: str = "",
) -> CaseAnalysis:
    coverage = round(_coverage(expected, actual), 4)
    has_failure_markers = any(m in (actual or "").lower() for m in FAILURE_MARKERS)

    if runtime_error:
        return CaseAnalysis(
            case_id=case_id,
            stage="runtime",
            severity="critical",
            confidence=0.99,
            reasons=[f"Runtime error: {runtime_error}"],
            metrics={"coverage": coverage},
        )

    retrieval_payloads = _extract_stage_payloads(events, "retrieval_search_raw")
    retrieval_queries_text = " ".join(str(p.get("query", "")) for p in retrieval_payloads)
    retrieval_docs_total = sum(int(p.get("total_docs", 0)) for p in retrieval_payloads)
    retrieval_queries = max(len(retrieval_payloads), 1)
    retrieval_docs_avg = retrieval_docs_total / retrieval_queries

    question_payloads = _extract_stage_payloads(events, "question_generation")
    generated_questions = sum(len(p.get("questions", [])) for p in question_payloads)

    retrieve_generation_payloads = _extract_stage_payloads(events, "retrieve_generation")
    retrieve_answers_short = 0
    for payload in retrieve_generation_payloads:
        if int(payload.get("answer_len", 0)) < 80:
            retrieve_answers_short += 1

    final_payloads = _extract_stage_payloads(events, "final_aggregation")
    final_context_items = 0
    final_answer_len = len(actual or "")
    if final_payloads:
        final_context_items = int(final_payloads[-1].get("context_items", 0))
        final_answer_len = int(final_payloads[-1].get("final_answer_len", final_answer_len))

    intent_text = f"{expected} {retrieval_queries_text}".strip()
    intent_tokens = _tokenize(intent_text)
    if not intent_tokens:
        intent_tokens = _tokenize(retrieval_queries_text)

    # Relevance of retrieved DB/vector docs to query intent.
    retrieved_snippets = []
    for payload in retrieval_payloads:
        for doc in payload.get("top_docs", []):
            retrieved_snippets.append(str(doc.get("snippet", "")))
    retrieval_text = " ".join(retrieved_snippets)
    retrieval_tokens = _tokenize(retrieval_text)
    retrieval_relevance = 0.0
    if intent_tokens:
        retrieval_relevance = len(intent_tokens.intersection(retrieval_tokens)) / len(intent_tokens)

    # Relevance of graph-derived text to query intent.
    graph_payloads = _extract_stage_payloads(events, "retrieval_graph_processed")
    graph_text = " ".join(
        f"{p.get('nodes_preview', '')} {p.get('rels_preview', '')}" for p in graph_payloads
    )
    graph_tokens = _tokenize(graph_text)
    graph_relevance = 0.0
    graph_signal_present = any(
        int(p.get("nodes_text_len", 0)) > 0 or int(p.get("rels_text_len", 0)) > 0
        for p in graph_payloads
    )
    if intent_tokens and graph_tokens:
        graph_relevance = len(intent_tokens.intersection(graph_tokens)) / len(intent_tokens)

    reasons: List[str] = []
    stage = "ok"
    severity = "none"
    confidence = 0.5

    if retrieval_docs_total == 0:
        stage = "retrieval"
        severity = "high"
        confidence = 0.9
        reasons.append("Поиск не вернул документов ни по одному запросу.")
    elif retrieval_docs_total > 0 and retrieval_relevance < 0.12 and coverage < 0.3:
        stage = "retrieval_irrelevant"
        severity = "high"
        confidence = 0.86
        reasons.append("Ретривер вернул в основном нерелевантные документы относительно запроса.")
    elif graph_signal_present and graph_relevance < 0.1 and coverage < 0.3:
        stage = "graph_irrelevant"
        severity = "high"
        confidence = 0.82
        reasons.append("Из графа знаний извлечен в основном нерелевантный контент.")
    elif generated_questions == 0 and len(question_payloads) > 0 and retrieval_docs_avg < 1:
        stage = "subquestions"
        severity = "high"
        confidence = 0.8
        reasons.append("Генератор подвопросов не выдал полезных запросов.")
    elif retrieve_answers_short > 0 and coverage < 0.25:
        stage = "retrieve_generation"
        severity = "medium"
        confidence = 0.75
        reasons.append("Ответы retrieve-этапа короткие/пустые при наличии результатов поиска.")
    elif final_context_items > 0 and (coverage < 0.25 or has_failure_markers or final_answer_len < 120):
        stage = "final_aggregation"
        severity = "medium"
        confidence = 0.7
        reasons.append("Агрегация не использовала собранный контекст достаточно качественно.")
    elif coverage < 0.2 and expected.strip():
        stage = "mixed"
        severity = "medium"
        confidence = 0.55
        reasons.append("Низкое соответствие эталону без явного единственного узкого места.")
    else:
        stage = "ok"
        severity = "none"
        confidence = 0.9
        reasons.append("Критичных проблем не выявлено по эвристикам.")

    return CaseAnalysis(
        case_id=case_id,
        stage=stage,
        severity=severity,
        confidence=round(confidence, 2),
        reasons=reasons,
        metrics={
            "coverage": coverage,
            "retrieval_docs_total": retrieval_docs_total,
            "retrieval_docs_avg": round(retrieval_docs_avg, 3),
            "retrieval_relevance": round(retrieval_relevance, 4),
            "graph_relevance": round(graph_relevance, 4),
            "graph_signal_present": graph_signal_present,
            "generated_questions": generated_questions,
            "retrieve_answers_short": retrieve_answers_short,
            "final_context_items": final_context_items,
            "final_answer_len": final_answer_len,
            "has_failure_markers": has_failure_markers,
        },
    )


def summarize_analyses(analyses: Iterable[CaseAnalysis]) -> Dict[str, Any]:
    stage_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {}
    coverage_values: List[float] = []

    for item in analyses:
        stage_counts[item.stage] = stage_counts.get(item.stage, 0) + 1
        severity_counts[item.severity] = severity_counts.get(item.severity, 0) + 1
        coverage_values.append(float(item.metrics.get("coverage", 0.0)))

    worst_stage = "ok"
    worst_stage_count = 0
    for stage, cnt in stage_counts.items():
        if stage == "ok":
            continue
        if cnt > worst_stage_count:
            worst_stage = stage
            worst_stage_count = cnt

    avg_coverage = sum(coverage_values) / max(len(coverage_values), 1)
    return {
        "stage_counts": stage_counts,
        "severity_counts": severity_counts,
        "worst_stage": worst_stage,
        "worst_stage_count": worst_stage_count,
        "avg_coverage": round(avg_coverage, 4),
    }


def save_analysis(output_path: Path, analyses: Iterable[CaseAnalysis], summary: Dict[str, Any]) -> None:
    payload = {
        "summary": summary,
        "cases": [a.as_dict() for a in analyses],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
