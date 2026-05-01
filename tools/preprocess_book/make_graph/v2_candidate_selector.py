from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List

from tools.preprocess_book.make_graph.v2_models import CandidateScore, EntityCluster, NodeEvent


def normalize_text(value: str) -> str:
    value = (value or "").lower().strip()
    value = re.sub(r"[^\w\s-]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)
    return value


def tokenize(value: str) -> set[str]:
    return {t for t in normalize_text(value).split(" ") if t}


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a_set, b_set = set(a), set(b)
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return inter / union if union else 0.0


class CandidateSelector:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def _name_score(self, event: NodeEvent, cluster: EntityCluster) -> float:
        event_names = [normalize_text(n) for n in event.all_names if n]
        cluster_names = [normalize_text(cluster.canonical_name)] + [
            normalize_text(n) for n in cluster.alt_names if n
        ]
        if not event_names or not cluster_names:
            return 0.0
        max_ratio = 0.0
        for en in event_names:
            for cn in cluster_names:
                if en == cn:
                    return 1.0
                max_ratio = max(max_ratio, SequenceMatcher(None, en, cn).ratio())
        return max_ratio

    def _alias_score(self, event: NodeEvent, cluster: EntityCluster) -> float:
        event_tokens = set()
        cluster_tokens = set()
        for n in event.all_names:
            event_tokens |= tokenize(n)
        for n in [cluster.canonical_name] + cluster.alt_names:
            cluster_tokens |= tokenize(n)
        return jaccard(event_tokens, cluster_tokens)

    def _context_score(self, event: NodeEvent, cluster: EntityCluster) -> float:
        event_tokens = tokenize(event.actions)
        cluster_tokens = tokenize(cluster.profile_text(last_n=8))
        return jaccard(event_tokens, cluster_tokens)

    def select(self, event: NodeEvent, clusters: Dict[str, EntityCluster]) -> List[CandidateScore]:
        scored: List[CandidateScore] = []
        for cluster in clusters.values():
            if not cluster.is_active:
                continue
            if cluster.classification != event.classification:
                continue

            name_score = self._name_score(event, cluster)
            alias_score = self._alias_score(event, cluster)
            context_score = self._context_score(event, cluster)
            total = 0.55 * name_score + 0.25 * alias_score + 0.20 * context_score

            # Fast reject: no lexical overlap at all.
            if name_score < 0.40 and alias_score < 0.20:
                continue

            scored.append(
                CandidateScore(
                    cluster_id=cluster.cluster_id,
                    total_score=total,
                    name_score=name_score,
                    alias_score=alias_score,
                    context_score=context_score,
                )
            )

        scored.sort(key=lambda x: x.total_score, reverse=True)
        return scored[: self.top_k]
