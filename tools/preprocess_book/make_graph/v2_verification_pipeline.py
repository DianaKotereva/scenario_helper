from __future__ import annotations

import asyncio
import logging
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.graph_search import Action, AllBookNodes, BookGraph, BookNode
from tools.preprocess_book.make_graph.v2_candidate_selector import (
    CandidateSelector,
    normalize_text,
)
from tools.preprocess_book.make_graph.v2_cluster_manager import ClusterManager
from tools.preprocess_book.make_graph.v2_edge_rewriter import EdgeRewriter
from tools.preprocess_book.make_graph.v2_io import append_jsonl, load_results
from tools.preprocess_book.make_graph.v2_llm_judge import LLMJudge
from tools.preprocess_book.make_graph.v2_models import (
    NodeEvent,
    RepairCandidate,
    VerificationDecision,
)
from tools.preprocess_book.storage.storage import FileManager

logger = logging.getLogger(__name__)

GENERIC_NAME_TOKENS = {
    "тень",
    "чародей",
    "колдун",
    "маг",
    "король",
    "королева",
    "бог",
    "боги",
    "эльф",
    "эльфы",
    "люди",
    "страж",
    "стражи",
}
GENERIC_NAME_PREFIXES = (
    "мать ",
    "отец ",
    "сын ",
    "дочь ",
    "дом ",
)
KINSHIP_PATTERNS = {
    "child_of": [
        re.compile(r"\bсын\s+([а-яa-zё][а-яa-zё-]{2,})", re.IGNORECASE),
        re.compile(r"\bдочь\s+([а-яa-zё][а-яa-zё-]{2,})", re.IGNORECASE),
    ],
    "parent_of": [
        re.compile(r"\bотец\s+([а-яa-zё][а-яa-zё-]{2,})", re.IGNORECASE),
        re.compile(r"\bмать\s+([а-яa-zё][а-яa-zё-]{2,})", re.IGNORECASE),
    ],
    "grandchild_of": [
        re.compile(r"\bвнук\s+([а-яa-zё][а-яa-zё-]{2,})", re.IGNORECASE),
        re.compile(r"\bвнучка\s+([а-яa-zё][а-яa-zё-]{2,})", re.IGNORECASE),
    ],
}
KINSHIP_ALIAS_PREFIXES = (
    "сын ",
    "дочь ",
    "отец ",
    "мать ",
    "внук ",
    "внучка ",
)
ROLE_LIKE_TOKENS = {
    "отец",
    "мать",
    "сын",
    "дочь",
    "внук",
    "внучка",
    "мальчик",
    "девочка",
    "ребёнок",
    "ребенок",
    "жена",
    "муж",
    "невеста",
    "жених",
    "король",
    "королева",
    "принц",
    "принцесса",
    "мачеха",
    "будущий",
    "будущая",
    "будущее",
}
ROLE_LIKE_PREFIXES = (
    "сын ",
    "дочь ",
    "отец ",
    "мать ",
    "внук ",
    "внучка ",
    "будущий ",
    "будущая ",
    "будущее ",
    "моя ",
    "мой ",
    "мальчик",
    "девочка",
)


class VerificationPipelineV2:
    def __init__(
        self,
        results_dir: Path,
        output_path: Path,
        logs_path: Path,
        top_k: int = 10,
        llm_type: str | None = None,
        llm_enabled: bool = True,
        max_files: int | None = None,
        enable_bridge_merge: bool = True,
        judge_concurrency: int = 6,
        strict_source_coverage: bool | None = None,
        strict_extraction_coverage: bool | None = None,
        repair_exact_name_only: bool = True,
    ):
        self.results_dir = results_dir
        self.output_path = output_path
        self.logs_path = logs_path
        self.top_k = top_k
        self.max_files = max_files
        self.enable_bridge_merge = enable_bridge_merge
        self.judge_concurrency = max(1, judge_concurrency)
        self.strict_source_coverage = (
            bool(strict_source_coverage)
            if strict_source_coverage is not None
            else os.getenv("V2_STRICT_SOURCE_COVERAGE", "false").lower()
            in {"1", "true", "yes", "on"}
        )
        self.strict_extraction_coverage = (
            bool(strict_extraction_coverage)
            if strict_extraction_coverage is not None
            else os.getenv("V2_STRICT_EXTRACTION_COVERAGE", "false").lower()
            in {"1", "true", "yes", "on"}
        )
        self.repair_exact_name_only = bool(repair_exact_name_only)

        self.selector = CandidateSelector(top_k=top_k)
        self.cluster_manager = ClusterManager()
        self.judge = LLMJudge(llm_type=llm_type, enabled=llm_enabled)
        self._judge_semaphore = asyncio.Semaphore(self.judge_concurrency)

    @staticmethod
    def _extract_source_ids_from_payload(payload: dict) -> set[int]:
        source_ids: set[int] = set()
        nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
        relations = (
            payload.get("relations")
            if isinstance(payload.get("relations"), list)
            else []
        )

        for node in nodes:
            if not isinstance(node, dict):
                continue
            for action in (
                node.get("actions") if isinstance(node.get("actions"), list) else []
            ):
                if isinstance(action, dict) and isinstance(
                    action.get("source_id"), int
                ):
                    source_ids.add(int(action["source_id"]))

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            for desc in (
                rel.get("descriptions")
                if isinstance(rel.get("descriptions"), list)
                else []
            ):
                if isinstance(desc, dict) and isinstance(desc.get("source_id"), int):
                    source_ids.add(int(desc["source_id"]))

        return source_ids

    def _extract_source_ids_from_clusters(self) -> set[int]:
        source_ids: set[int] = set()
        for cluster in self.cluster_manager.get_active_clusters().values():
            for _action, sid in cluster.actions:
                if isinstance(sid, tuple):
                    source_ids.update(int(v) for v in sid if isinstance(v, int))
                elif isinstance(sid, int):
                    source_ids.add(int(sid))
        return source_ids

    @staticmethod
    def _extract_source_ids_from_relations_by_source(
        relations_by_source: Dict[Tuple[int, ...], List[dict]],
    ) -> set[int]:
        source_ids: set[int] = set()
        for rel_list in relations_by_source.values():
            for rel in rel_list:
                if not isinstance(rel, dict):
                    continue
                for desc in (
                    rel.get("descriptions")
                    if isinstance(rel.get("descriptions"), list)
                    else []
                ):
                    if isinstance(desc, dict) and isinstance(
                        desc.get("source_id"), int
                    ):
                        source_ids.add(int(desc["source_id"]))
        return source_ids

    @staticmethod
    def _confidence_bonus(confidence: str) -> float:
        value = normalize_text(confidence)
        if value in {"высокий", "high"}:
            return 0.10
        if value in {"средний", "medium"}:
            return 0.04
        if value in {"низкий", "low"}:
            return -0.03
        return 0.0

    @staticmethod
    def _main_to_canonical_score(main_name: str, canonical_name: str) -> float:
        return SequenceMatcher(
            None,
            normalize_text(main_name),
            normalize_text(canonical_name),
        ).ratio()

    @staticmethod
    def _cluster_support(cluster) -> float:
        return ClusterManager.cluster_support_score(cluster)

    @staticmethod
    def _event_main_hits_cluster_alias(event: NodeEvent, cluster) -> bool:
        main_norm = normalize_text(event.main_name)
        if not main_norm:
            return False
        names = {normalize_text(cluster.canonical_name)} | {
            normalize_text(x) for x in getattr(cluster, "alt_names", [])
        }
        return main_norm in names

    @staticmethod
    def _passes_repair_acceptance_rules(cand: RepairCandidate) -> bool:
        if cand.name_score >= 0.90:
            return True
        if cand.mention_score >= 1.0:
            return True
        if cand.alias_score >= 0.45 and cand.name_score >= 0.70:
            return True
        return False

    @staticmethod
    def _extract_kinship_signature(text: str) -> Dict[str, set[str]]:
        norm = normalize_text(text)
        signature = {k: set() for k in KINSHIP_PATTERNS}
        if not norm:
            return signature
        for kind, patterns in KINSHIP_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(norm):
                    anchor = normalize_text(match.group(1))
                    if anchor:
                        signature[kind].add(anchor)
        return signature

    @staticmethod
    def _cluster_kinship_signature(cluster) -> Dict[str, set[str]]:
        payload = " ".join(
            [cluster.canonical_name]
            + list(cluster.alt_names)
            + list(getattr(cluster, "kinship_aliases", []))
            + [cluster.profile_text(last_n=24)]
        )
        return VerificationPipelineV2._extract_kinship_signature(payload)

    @staticmethod
    def _anchors_overlap(a: set[str], b: set[str], min_ratio: float = 0.84) -> bool:
        if not a or not b:
            return False
        for left in a:
            for right in b:
                if left == right:
                    return True
                if SequenceMatcher(None, left, right).ratio() >= min_ratio:
                    return True
        return False

    @staticmethod
    def _hard_negative_reason(cluster_a, cluster_b) -> str | None:
        sig_a = VerificationPipelineV2._cluster_kinship_signature(cluster_a)
        sig_b = VerificationPipelineV2._cluster_kinship_signature(cluster_b)
        name_a = normalize_text(cluster_a.canonical_name)
        name_b = normalize_text(cluster_b.canonical_name)

        # Different parent anchors for same kinship role.
        if (
            sig_a["child_of"]
            and sig_b["child_of"]
            and not VerificationPipelineV2._anchors_overlap(
                sig_a["child_of"], sig_b["child_of"]
            )
        ):
            return "hard_negative:child_of_conflict"

        if (
            sig_a["grandchild_of"]
            and sig_b["grandchild_of"]
            and not VerificationPipelineV2._anchors_overlap(
                sig_a["grandchild_of"], sig_b["grandchild_of"]
            )
        ):
            return "hard_negative:grandchild_of_conflict"

        # Impossible cycle: one cluster says child_of(X), another says parent_of(X).
        if VerificationPipelineV2._anchors_overlap(
            sig_a["child_of"], sig_b["parent_of"]
        ):
            return "hard_negative:child_parent_cycle"
        if VerificationPipelineV2._anchors_overlap(
            sig_b["child_of"], sig_a["parent_of"]
        ):
            return "hard_negative:child_parent_cycle"
        # Direct parent-child contradiction by canonical names.
        if name_a and name_a in sig_b["child_of"]:
            return "hard_negative:canonical_child_parent_cycle"
        if name_b and name_b in sig_a["child_of"]:
            return "hard_negative:canonical_child_parent_cycle"

        return None

    @staticmethod
    def _is_role_like_label(value: str) -> bool:
        norm = normalize_text(value)
        if not norm:
            return False
        if any(norm.startswith(prefix) for prefix in ROLE_LIKE_PREFIXES):
            return True
        tokens = [t for t in norm.split(" ") if t]
        if len(tokens) == 1 and tokens[0] in ROLE_LIKE_TOKENS:
            return True
        return False

    @staticmethod
    def _strong_alias_set(cluster) -> set[str]:
        values = {normalize_text(cluster.canonical_name)}
        values.update(normalize_text(v) for v in getattr(cluster, "alt_names", []))
        values.update(normalize_text(v) for v in getattr(cluster, "kinship_aliases", []))
        filtered: set[str] = set()
        for value in values:
            if not value:
                continue
            if VerificationPipelineV2._is_role_like_label(value):
                continue
            filtered.add(value)
        return filtered

    @staticmethod
    def _passes_non_identical_name_guard(cluster_a, cluster_b, cand: RepairCandidate) -> bool:
        name_a = normalize_text(cluster_a.canonical_name)
        name_b = normalize_text(cluster_b.canonical_name)
        if not name_a or not name_b or name_a == name_b:
            return True

        # Do not merge explicit role labels into canonical entities.
        if VerificationPipelineV2._is_role_like_label(cluster_a.canonical_name):
            return False
        if VerificationPipelineV2._is_role_like_label(cluster_b.canonical_name):
            return False

        aliases_a = VerificationPipelineV2._strong_alias_set(cluster_a)
        aliases_b = VerificationPipelineV2._strong_alias_set(cluster_b)
        explicit_bridge = (name_a in aliases_b) or (name_b in aliases_a)
        lexical_similarity = SequenceMatcher(None, name_a, name_b).ratio()

        # Accept non-identical names only with strong evidence.
        if explicit_bridge:
            return True
        if lexical_similarity >= 0.92:
            return True
        if cand.alias_score >= 0.60 and cand.mention_score >= 1.0:
            return True
        return False

    @staticmethod
    def _is_generic_main_name(main_name: str) -> bool:
        norm = normalize_text(main_name)
        if not norm:
            return False
        for prefix in GENERIC_NAME_PREFIXES:
            if norm.startswith(prefix):
                return True
        tokens = [t for t in norm.split(" ") if t]
        return len(tokens) == 1 and tokens[0] in GENERIC_NAME_TOKENS

    @staticmethod
    def _split_alt_names(alt_names: List[str]) -> tuple[List[str], List[str]]:
        clean: List[str] = []
        kinship: List[str] = []
        for raw in alt_names:
            value = (raw or "").strip()
            if not value:
                continue
            normalized = normalize_text(value)
            if any(normalized.startswith(prefix) for prefix in KINSHIP_ALIAS_PREFIXES):
                if value not in kinship:
                    kinship.append(value)
                continue
            if value not in clean:
                clean.append(value)
        return clean, kinship

    @staticmethod
    def _judge_has_hard_conflict(judge_payload: dict) -> bool:
        flags = judge_payload.get("hard_conflict_flags", [])
        return isinstance(flags, list) and len(flags) > 0

    @staticmethod
    def _normalize_classification(value: str) -> str:
        raw = str(value or "").strip()
        mapping = {
            "EntityClassification.PERSON": "персонаж",
            "EntityClassification.PLACE": "место",
            "EntityClassification.ORG": "организация",
            "EntityClassification.TERM": "термин",
            "EntityClassification.FORCE": "сила природы",
        }
        return mapping.get(raw, raw)

    @staticmethod
    def _flatten_actions(raw_actions) -> str:
        if isinstance(raw_actions, str):
            return raw_actions.strip()
        if not isinstance(raw_actions, list):
            return ""
        chunks: List[str] = []
        for item in raw_actions:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    chunks.append(text)
                continue
            if not isinstance(item, dict):
                continue
            desc = str(item.get("description", "")).strip()
            if not desc:
                continue
            sid = item.get("source_id")
            cid = item.get("chapter_id")
            chunks.append(f"[source_id={sid} chapter_id={cid}] {desc}")
        return ". ".join(chunks)

    @staticmethod
    def _extract_action_entries(
        raw_actions, fallback_source: Tuple[int, ...]
    ) -> List[Tuple[str, Tuple[int, ...]]]:
        entries: List[Tuple[str, Tuple[int, ...]]] = []
        if isinstance(raw_actions, list):
            for item in raw_actions:
                if isinstance(item, dict):
                    desc = str(item.get("description", "")).strip()
                    if not desc:
                        continue
                    sid = item.get("source_id")
                    if isinstance(sid, int):
                        entries.append((desc, (int(sid),)))
                    elif fallback_source:
                        entries.append((desc, fallback_source))
                    continue
                if isinstance(item, str):
                    desc = item.strip()
                    if desc:
                        entries.append((desc, fallback_source))
            return entries
        if isinstance(raw_actions, str):
            desc = raw_actions.strip()
            if desc:
                entries.append((desc, fallback_source))
        return entries

    @staticmethod
    def _event_chapter_id(raw: dict, fallback_source: Tuple[int, ...]) -> int:
        fallback = int(fallback_source[0]) if fallback_source else 0
        actions = raw.get("actions")
        if isinstance(actions, list):
            for item in actions:
                if isinstance(item, dict) and isinstance(item.get("chapter_id"), int):
                    return int(item["chapter_id"])
        return fallback

    def _passes_generic_guard(self, event: NodeEvent, cluster, cand) -> bool:
        if not self._is_generic_main_name(event.main_name):
            return True

        event_main = normalize_text(event.main_name)
        canonical = normalize_text(cluster.canonical_name)
        alt_names = {normalize_text(x) for x in getattr(cluster, "alt_names", [])}
        explicit_alias = event_main == canonical or event_main in alt_names
        # Generic names are allowed only with explicit alias and strong lexical match.
        return explicit_alias and cand.name_score >= 0.90

    def _build_event(
        self, source_id: Tuple[int, ...], idx: int, raw: dict
    ) -> NodeEvent:
        chapter_id = self._event_chapter_id(raw, source_id)
        alt_names, kinship_aliases = self._split_alt_names(
            list(raw.get("alt_names", []) or [])
        )
        raw_actions = raw.get("actions", "")
        return NodeEvent(
            source_id=source_id,
            chapter_id=chapter_id,
            main_name=raw.get("main_name", ""),
            alt_names=alt_names,
            kinship_aliases=kinship_aliases,
            classification=self._normalize_classification(
                raw.get("classification", "")
            ),
            actions=self._flatten_actions(raw_actions),
            event_idx=idx,
            action_entries=self._extract_action_entries(raw_actions, source_id),
        )

    async def _judge_candidate_async(self, event: NodeEvent, cand, cluster) -> dict:
        async with self._judge_semaphore:
            judge = await self.judge.verify_async(
                cluster=cluster,
                new_node={
                    "main_name": event.main_name,
                    "alt_names": event.alt_names,
                    "kinship_aliases": event.kinship_aliases,
                    "classification": event.classification,
                    "actions": event.actions,
                },
                source_id=event.source_id,
            )
        has_hard_conflict = self._judge_has_hard_conflict(judge)
        is_same = bool(judge.get("is_same_entity", False)) and not has_hard_conflict
        passed_generic_guard = self._passes_generic_guard(event, cluster, cand)
        return {
            "candidate_cluster_id": cand.cluster_id,
            "candidate_canonical_name": cluster.canonical_name,
            "candidate_score": {
                "total": cand.total_score,
                "name": cand.name_score,
                "alias": cand.alias_score,
                "context": cand.context_score,
            },
            "judge": judge,
            "judge_hard_conflict": has_hard_conflict,
            "passed_generic_guard": passed_generic_guard,
            "is_same": is_same,
        }

    async def _verify_event_async(self, event: NodeEvent) -> VerificationDecision:
        active_clusters = self.cluster_manager.get_active_clusters()
        candidates = self.selector.select(event, active_clusters)
        return await self._verify_event_with_candidates_async(
            event=event,
            candidates=candidates,
            active_clusters=active_clusters,
        )

    async def _verify_event_with_candidates_async(
        self,
        event: NodeEvent,
        candidates,
        active_clusters,
    ) -> VerificationDecision:
        if not candidates:
            return VerificationDecision(
                accepted_cluster_id=None,
                candidates_checked=[],
                judge_traces=[],
                reason="no_candidates",
            )

        tasks = []
        effective_candidates = []
        for cand in candidates:
            cluster = active_clusters.get(cand.cluster_id)
            if cluster is None:
                continue
            effective_candidates.append(cand)
            tasks.append(self._judge_candidate_async(event, cand, cluster))

        if not tasks:
            return VerificationDecision(
                accepted_cluster_id=None,
                candidates_checked=[],
                judge_traces=[],
                reason="no_candidates",
            )

        judge_traces = await asyncio.gather(*tasks)
        accepted_options: List[dict] = []

        alias_hit_candidates = 0
        for cand, trace in zip(effective_candidates, judge_traces, strict=True):
            if not trace["is_same"] or not trace["passed_generic_guard"]:
                continue
            cluster = active_clusters.get(cand.cluster_id)
            if cluster is None:
                continue

            confidence = str(trace["judge"].get("confidence", ""))
            confidence_bonus = self._confidence_bonus(confidence)
            canonical_match = self._main_to_canonical_score(
                event.main_name,
                trace["candidate_canonical_name"],
            )
            alias_hit = self._event_main_hits_cluster_alias(event, cluster)
            if alias_hit:
                alias_hit_candidates += 1

            accepted_options.append(
                {
                    "cluster_id": cand.cluster_id,
                    "candidate_score": cand.total_score,
                    "confidence_bonus": confidence_bonus,
                    "canonical_match": canonical_match,
                    "alias_hit": alias_hit,
                    "support": self._cluster_support(cluster),
                }
            )

        if accepted_options:
            for option in accepted_options:
                canonical_weight = 0.08
                # If multiple accepted clusters already mention this main_name as alias,
                # canonical lexical proximity becomes less reliable.
                if alias_hit_candidates >= 2 and option["alias_hit"]:
                    canonical_weight = 0.0

                option["rank"] = (
                    option["candidate_score"]
                    + option["confidence_bonus"]
                    + (canonical_weight * option["canonical_match"])
                    + (0.02 if option["alias_hit"] else 0.0)
                    + min(0.10, 0.01 * option["support"])
                )

            accepted_options.sort(
                key=lambda x: (-x["rank"], -x["support"], x["cluster_id"]),
            )
            best_cluster_id = accepted_options[0]["cluster_id"]
            return VerificationDecision(
                accepted_cluster_id=best_cluster_id,
                candidates_checked=effective_candidates,
                judge_traces=judge_traces,
                reason="judge_accept",
            )

        return VerificationDecision(
            accepted_cluster_id=None,
            candidates_checked=effective_candidates,
            judge_traces=judge_traces,
            reason="judge_reject_all",
        )

    async def _verify_events_batch_async(
        self, events: List[NodeEvent]
    ) -> List[VerificationDecision]:
        """
        Async batch verification for a set of events.

        Important: all events in the batch are verified against the same
        snapshot of active clusters, then decisions are applied afterwards.
        This matches the requested "compare async, then collect" scheme.
        """
        if not events:
            return []

        active_clusters_snapshot = self.cluster_manager.get_active_clusters()
        prepared = []
        for event in events:
            candidates = self.selector.select(event, active_clusters_snapshot)
            prepared.append((event, candidates))

        tasks = [
            self._verify_event_with_candidates_async(
                event=event,
                candidates=candidates,
                active_clusters=active_clusters_snapshot,
            )
            for event, candidates in prepared
        ]
        decisions = await asyncio.gather(*tasks)
        return decisions

    @staticmethod
    def _repair_new_node_payload(cluster) -> dict:
        return {
            "main_name": cluster.canonical_name,
            "alt_names": cluster.alt_names,
            "kinship_aliases": list(getattr(cluster, "kinship_aliases", [])),
            "classification": cluster.classification,
            "actions": cluster.profile_text(last_n=12),
        }

    def _pick_merge_direction(self, a_root: str, b_root: str) -> tuple[str, str]:
        a = self.cluster_manager.clusters[a_root]
        b = self.cluster_manager.clusters[b_root]
        a_support = self._cluster_support(a)
        b_support = self._cluster_support(b)
        if b_support > a_support:
            return b_root, a_root
        if b_support < a_support:
            return a_root, b_root
        if b.first_seen_chapter < a.first_seen_chapter:
            return b_root, a_root
        if b.first_seen_chapter > a.first_seen_chapter:
            return a_root, b_root
        return tuple(sorted([a_root, b_root]))  # deterministic fallback

    async def _judge_repair_candidate_async(
        self, iteration: int, cand: RepairCandidate
    ) -> dict:
        a_root = self.cluster_manager.resolve_cluster_id(cand.cluster_a_id)
        b_root = self.cluster_manager.resolve_cluster_id(cand.cluster_b_id)
        if a_root == b_root:
            return {"skip": True, "reason": "already_merged", "candidate": cand}
        ca = self.cluster_manager.clusters.get(a_root)
        cb = self.cluster_manager.clusters.get(b_root)
        if not ca or not cb or not ca.is_active or not cb.is_active:
            return {"skip": True, "reason": "inactive_cluster", "candidate": cand}
        if ca.classification != cb.classification:
            return {
                "skip": True,
                "reason": "classification_mismatch",
                "candidate": cand,
            }
        if self.repair_exact_name_only:
            name_a = normalize_text(ca.canonical_name)
            name_b = normalize_text(cb.canonical_name)
            if name_a != name_b:
                return {
                    "skip": True,
                    "reason": "hard_negative:repair_exact_name_only",
                    "candidate": cand,
                }
        hard_negative_reason = self._hard_negative_reason(ca, cb)
        if hard_negative_reason:
            return {"skip": True, "reason": hard_negative_reason, "candidate": cand}
        if not self._passes_non_identical_name_guard(ca, cb, cand):
            return {
                "skip": True,
                "reason": "hard_negative:non_identical_name_guard",
                "candidate": cand,
            }

        async with self._judge_semaphore:
            judge = await self.judge.verify_async(
                cluster=ca,
                new_node=self._repair_new_node_payload(cb),
                source_id=(cb.first_seen_chapter,),
            )

        return {
            "skip": False,
            "iteration": iteration,
            "a_root": a_root,
            "b_root": b_root,
            "candidate": cand,
            "judge": judge,
            "is_same": bool(judge.get("is_same_entity", False))
            and not self._judge_has_hard_conflict(judge),
        }

    async def _repair_iteration_async(self, iteration: int) -> int:
        candidates = self.cluster_manager.candidate_clusters_for_repair(
            top_k_per_cluster=max(6, self.top_k),
            min_total_score=0.34,
            max_token_bucket=70,
            max_pair_candidates=250,
        )
        if not candidates:
            append_jsonl(
                self.logs_path,
                {
                    "stage": "repair_iteration",
                    "iteration": iteration,
                    "candidates": 0,
                    "merges": 0,
                },
            )
            return 0

        tasks = [
            self._judge_repair_candidate_async(iteration, cand) for cand in candidates
        ]
        results = await asyncio.gather(*tasks)

        accepted = [
            r
            for r in results
            if not r["skip"]
            and r["is_same"]
            and self._passes_repair_acceptance_rules(r["candidate"])
        ]
        accepted.sort(
            key=lambda r: (
                r["candidate"].total_score,
                r["candidate"].mention_score,
                r["candidate"].name_score,
                r["candidate"].alias_score,
            ),
            reverse=True,
        )

        merges = 0
        for item in results:
            cand = item["candidate"]
            append_jsonl(
                self.logs_path,
                {
                    "stage": "repair_candidate",
                    "iteration": iteration,
                    "cluster_a": cand.cluster_a_id,
                    "cluster_b": cand.cluster_b_id,
                    "scores": {
                        "total": cand.total_score,
                        "name": cand.name_score,
                        "alias": cand.alias_score,
                        "mention": cand.mention_score,
                        "context": cand.context_score,
                    },
                    "skip": item.get("skip", False),
                    "skip_reason": item.get("reason"),
                    "judge": item.get("judge"),
                },
            )

        for item in accepted:
            a_root = self.cluster_manager.resolve_cluster_id(item["a_root"])
            b_root = self.cluster_manager.resolve_cluster_id(item["b_root"])
            if a_root == b_root:
                continue
            ca = self.cluster_manager.clusters.get(a_root)
            cb = self.cluster_manager.clusters.get(b_root)
            if not ca or not cb or not ca.is_active or not cb.is_active:
                continue
            if ca.classification != cb.classification:
                continue

            target_id, donor_id = self._pick_merge_direction(a_root, b_root)
            merged_into = self.cluster_manager.merge_clusters(target_id, donor_id)
            if merged_into == donor_id:
                continue
            merges += 1
            append_jsonl(
                self.logs_path,
                {
                    "stage": "repair_merge",
                    "iteration": iteration,
                    "target": target_id,
                    "donor": donor_id,
                    "merged_into": merged_into,
                    "candidate_scores": {
                        "total": item["candidate"].total_score,
                        "name": item["candidate"].name_score,
                        "alias": item["candidate"].alias_score,
                        "mention": item["candidate"].mention_score,
                        "context": item["candidate"].context_score,
                    },
                    "judge": item["judge"],
                },
            )

        append_jsonl(
            self.logs_path,
            {
                "stage": "repair_iteration",
                "iteration": iteration,
                "candidates": len(candidates),
                "accepted": len(accepted),
                "merges": merges,
            },
        )
        return merges

    async def _repair_to_convergence_async(
        self, max_iterations: int = 6
    ) -> tuple[int, int]:
        total_merges = 0
        iterations = 0
        for iteration in range(1, max_iterations + 1):
            iterations = iteration
            merges = await self._repair_iteration_async(iteration)
            total_merges += merges
            if merges == 0:
                break
        return total_merges, iterations

    def _to_book_graph(
        self, relations_by_source: Dict[Tuple[int, ...], List[dict]]
    ) -> BookGraph:
        all_nodes = AllBookNodes(nodes={}, names_list={})
        canonical_name_counter: Dict[str, int] = {}
        cluster_to_node_key: Dict[str, str] = {}

        def _unique_key_for_canonical(canonical_name: str, cluster_id: str) -> str:
            base = canonical_name or cluster_id
            if base not in all_nodes.nodes:
                return base
            canonical_name_counter[base] = canonical_name_counter.get(base, 1) + 1
            candidate = f"{base}__{cluster_id}"
            if candidate not in all_nodes.nodes:
                return candidate
            return f"{base}__{cluster_id}__{canonical_name_counter[base]}"

        def _safe_alias_map(alias: str, target_key: str) -> None:
            if not alias:
                return
            existing = all_nodes.names_list.get(alias)
            if existing is None or existing == target_key:
                all_nodes.names_list[alias] = target_key

        for cluster in self.cluster_manager.get_active_clusters().values():
            node = BookNode(
                main_name=cluster.canonical_name,
                classification=cluster.classification,
                alt_names=list(cluster.alt_names),
                actions=[Action(action=a, source_id=sid) for a, sid in cluster.actions],
            )
            node_key = _unique_key_for_canonical(node.main_name, cluster.cluster_id)
            cluster_to_node_key[cluster.cluster_id] = node_key
            all_nodes.nodes[node_key] = node
            _safe_alias_map(node.main_name, node_key)
            for alt in node.alt_names:
                _safe_alias_map(alt, node_key)
            if node_key != node.main_name:
                _safe_alias_map(f"{node.main_name}__{cluster.cluster_id}", node_key)
                _safe_alias_map(cluster.cluster_id, node_key)

        rel_graph = EdgeRewriter(self.cluster_manager).rewrite_relations(
            relations_by_source,
            cluster_to_node_key=cluster_to_node_key,
        )
        return BookGraph(nodes=all_nodes, relationships=rel_graph)

    async def run_async(self) -> BookGraph:
        results = load_results(self.results_dir, max_files=self.max_files)
        logger.info(
            "Loaded %s extraction result files from %s", len(results), self.results_dir
        )

        relations_by_source: Dict[Tuple[int, ...], List[dict]] = {}
        cumulative_expected_source_ids: set[int] = set()
        for payload in results:
            source_id = payload["_source_id"]
            nodes = payload.get("nodes", []) or []
            relations = payload.get("relations", []) or []
            relations_by_source[source_id] = relations
            expected_source_ids = {int(v) for v in source_id if isinstance(v, int)}
            extracted_source_ids = self._extract_source_ids_from_payload(payload)
            missing_in_extraction = sorted(expected_source_ids - extracted_source_ids)
            if self.strict_extraction_coverage and missing_in_extraction:
                raise RuntimeError(
                    "V2 extraction coverage gate failed: "
                    f"source_id={source_id}, missing={missing_in_extraction}, "
                    f"expected={sorted(expected_source_ids)}, extracted={sorted(extracted_source_ids)}"
                )

            events = [
                self._build_event(source_id, idx, raw_node)
                for idx, raw_node in enumerate(nodes)
            ]
            decisions = await self._verify_events_batch_async(events)

            for event, decision in zip(events, decisions, strict=True):
                if decision.accepted_cluster_id:
                    cid = self.cluster_manager.resolve_cluster_id(
                        decision.accepted_cluster_id
                    )
                    self.cluster_manager.attach_event(cid, event)
                    assigned = cid
                    action = "attach"
                else:
                    assigned = self.cluster_manager.create_cluster(event)
                    action = "create"

                append_jsonl(
                    self.logs_path,
                    {
                        "stage": "per_event",
                        "source_id": source_id,
                        "event_idx": event.event_idx,
                        "main_name": event.main_name,
                        "classification": event.classification,
                        "action": action,
                        "assigned_cluster_id": assigned,
                        "decision_reason": decision.reason,
                        "candidates_checked": [
                            {
                                "cluster_id": c.cluster_id,
                                "total": c.total_score,
                                "name": c.name_score,
                                "alias": c.alias_score,
                                "context": c.context_score,
                            }
                            for c in decision.candidates_checked
                        ],
                        "judge_traces": decision.judge_traces,
                    },
                )

            cumulative_expected_source_ids.update(expected_source_ids)
            graph_source_ids = self._extract_source_ids_from_clusters().union(
                self._extract_source_ids_from_relations_by_source(relations_by_source)
            )
            missing_current = sorted(expected_source_ids - graph_source_ids)
            missing_cumulative = sorted(
                cumulative_expected_source_ids - graph_source_ids
            )

            append_jsonl(
                self.logs_path,
                {
                    "stage": "batch_coverage",
                    "source_id": source_id,
                    "expected_source_ids_current": sorted(expected_source_ids),
                    "expected_source_ids_cumulative": sorted(
                        cumulative_expected_source_ids
                    ),
                    "extracted_source_ids_current": sorted(extracted_source_ids),
                    "missing_source_ids_in_extraction": missing_in_extraction,
                    "graph_source_ids_present": sorted(graph_source_ids),
                    "missing_source_ids_current": missing_current,
                    "missing_source_ids_cumulative": missing_cumulative,
                },
            )

            if self.strict_source_coverage and missing_current:
                raise RuntimeError(
                    "V2 source coverage gate failed: "
                    f"source_id={source_id}, missing_current={missing_current}"
                )

        repair_merges = 0
        repair_iterations = 0
        if self.enable_bridge_merge:
            repair_merges, repair_iterations = await self._repair_to_convergence_async(
                max_iterations=6
            )

        graph = self._to_book_graph(relations_by_source)
        FileManager.save_pickle(graph, self.output_path)
        logger.info(
            "v2 graph saved to %s (nodes=%s, relations=%s, repair_merges=%s, repair_iterations=%s)",
            self.output_path,
            len(graph.nodes.nodes),
            len(graph.relationships.relationships),
            repair_merges,
            repair_iterations,
        )
        return graph

    def run(self) -> BookGraph:
        return asyncio.run(self.run_async())
