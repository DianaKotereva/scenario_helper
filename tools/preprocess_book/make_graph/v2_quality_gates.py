from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.utils.graph_search import BookGraph
from tools.preprocess_book.make_graph.v2_candidate_selector import normalize_text


CONTROL_IDENTITY_GROUP = ["Малекит", "Феанаро", "Куруфинвэ Феанаро"]
AENARION_DISAMBIGUATION_PAIRS = [
    ("Аэнарион Защитник", "Аэнарион"),
]
RELATION_DROP_CONTROL_PAIRS = [
    ("Малекит", "Нерданэль"),
    ("Феанаро", "Нерданэль"),
    ("Малекит", "Финвэ"),
    ("Феанаро", "Финвэ"),
]


def _normalized_name_map(graph: BookGraph) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for alias, canonical in graph.nodes.names_list.items():
        key = normalize_text(alias)
        if key and key not in data:
            data[key] = canonical
    for canonical in graph.nodes.nodes.keys():
        key = normalize_text(canonical)
        if key and key not in data:
            data[key] = canonical
    return data


def canonical_for_name(graph: BookGraph, name: str) -> Optional[str]:
    if name in graph.nodes.names_list:
        return graph.nodes.names_list[name]
    norm_map = _normalized_name_map(graph)
    return norm_map.get(normalize_text(name))


def _relation_description_count(graph: BookGraph, left_name: str, right_name: str) -> Optional[int]:
    left = canonical_for_name(graph, left_name)
    right = canonical_for_name(graph, right_name)
    if not left or not right:
        return None
    pair = tuple(sorted([left, right]))
    edge = graph.relationships.relationships.get(pair)
    if not edge:
        return 0
    return len(edge.description)


def evaluate_merge_quality(
    graph: BookGraph,
    baseline_graph: Optional[BookGraph] = None,
    max_relation_drop_ratio: float = 0.45,
) -> Dict:
    violations: List[str] = []
    warnings: List[str] = []
    checks: List[Dict] = []

    # Control case 1: Malekit / Feanaro / Kurufinwe Feanaro are one entity.
    identity_targets: List[Tuple[str, Optional[str]]] = [
        (name, canonical_for_name(graph, name)) for name in CONTROL_IDENTITY_GROUP
    ]
    missing_identity = [name for name, canonical in identity_targets if canonical is None]
    if missing_identity:
        violations.append(
            f"identity_group_missing:{', '.join(missing_identity)}"
        )
    else:
        unique_canonicals = {canonical for _, canonical in identity_targets}
        if len(unique_canonicals) != 1:
            violations.append(
                "identity_group_split:" + ", ".join(
                    f"{name}->{canonical}" for name, canonical in identity_targets
                )
            )

        # If merged, require explicit identity evidence in self-edge.
        merged_canonical = next(iter(unique_canonicals))
        pair = (merged_canonical, merged_canonical)
        self_edge = graph.relationships.relationships.get(pair)
        has_identity_evidence = False
        if self_edge:
            has_identity_evidence = any(
                d.type == "IDENTITY_MERGED_CONTEXT" for d in self_edge.description
            )
        if not has_identity_evidence:
            violations.append("identity_evidence_missing:self_edge_identity_merged_context")

    checks.append(
        {
            "check": "identity_group",
            "details": [{"name": name, "canonical": canonical} for name, canonical in identity_targets],
        }
    )

    # Control case 2: Aenarion father/son disambiguation (if names present).
    disambiguation_checked = False
    for father_name, son_name in AENARION_DISAMBIGUATION_PAIRS:
        father_cur = canonical_for_name(graph, father_name)
        son_cur = canonical_for_name(graph, son_name)
        father_base = canonical_for_name(baseline_graph, father_name) if baseline_graph else None
        son_base = canonical_for_name(baseline_graph, son_name) if baseline_graph else None

        if not any([father_cur, son_cur, father_base, son_base]):
            continue

        disambiguation_checked = True
        checks.append(
            {
                "check": "aenarion_disambiguation",
                "pair": [father_name, son_name],
                "current": [father_cur, son_cur],
                "baseline": [father_base, son_base],
            }
        )

        if baseline_graph and father_base and son_base:
            if father_base != son_base:
                if not father_cur or not son_cur:
                    violations.append(
                        f"aenarion_degradation_missing:{father_name}/{son_name}"
                    )
                elif father_cur == son_cur:
                    violations.append(
                        f"aenarion_degradation_merge:{father_name}/{son_name}"
                    )
            elif father_cur and son_cur and father_cur == son_cur:
                warnings.append(
                    f"aenarion_not_disambiguated_in_baseline:{father_name}/{son_name}"
                )
        elif father_cur and son_cur and father_cur == son_cur:
            warnings.append(f"aenarion_collision_unverified:{father_name}/{son_name}")

    if not disambiguation_checked:
        warnings.append("aenarion_disambiguation_not_checked:no_known_pair_present")

    # Baseline relation degradation checks.
    if baseline_graph:
        for left_name, right_name in RELATION_DROP_CONTROL_PAIRS:
            base_cnt = _relation_description_count(baseline_graph, left_name, right_name)
            cur_cnt = _relation_description_count(graph, left_name, right_name)
            checks.append(
                {
                    "check": "relation_drop",
                    "pair": [left_name, right_name],
                    "baseline_count": base_cnt,
                    "current_count": cur_cnt,
                }
            )
            if base_cnt is None or cur_cnt is None or base_cnt <= 0:
                continue
            min_allowed = int(base_cnt * (1.0 - max_relation_drop_ratio))
            if cur_cnt < min_allowed:
                violations.append(
                    f"relation_drop:{left_name}-{right_name}: current={cur_cnt}, baseline={base_cnt}, min_allowed={min_allowed}"
                )

    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
        "checks": checks,
    }
