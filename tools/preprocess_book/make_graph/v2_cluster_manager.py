from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple

from tools.preprocess_book.make_graph.v2_candidate_selector import jaccard, normalize_text, tokenize
from tools.preprocess_book.make_graph.v2_models import EntityCluster, NodeEvent, RepairCandidate


@dataclass
class MergeResult:
    created_cluster: str | None = None
    merged_into: str | None = None


class ClusterManager:
    def __init__(self):
        self.clusters: Dict[str, EntityCluster] = {}
        self.name_to_cluster: Dict[str, str] = {}
        # Ambiguous aliases map to multiple candidate clusters (homonyms/roles).
        self.ambiguous_name_to_clusters: Dict[str, Set[str]] = {}
        self._counter = 0

    def _next_cluster_id(self) -> str:
        self._counter += 1
        return f"c{self._counter:06d}"

    def resolve_cluster_id(self, cluster_id: str) -> str:
        current = cluster_id
        while current in self.clusters and self.clusters[current].merged_into:
            current = self.clusters[current].merged_into  # path traversal on merge chain
        return current

    def _register_name(self, name: str, cluster_id: str) -> None:
        if not name:
            return
        norm_name = normalize_text(name)
        if not norm_name:
            return
        resolved_new = self.resolve_cluster_id(cluster_id)
        existing = self.name_to_cluster.get(norm_name)
        if existing is None:
            self.name_to_cluster[norm_name] = resolved_new
            return

        resolved_existing = self.resolve_cluster_id(existing)
        if resolved_existing == resolved_new:
            self.name_to_cluster[norm_name] = resolved_existing
            return

        # Keep existing fast-path mapping stable and track ambiguity explicitly.
        bucket = self.ambiguous_name_to_clusters.setdefault(norm_name, set())
        bucket.add(resolved_existing)
        bucket.add(resolved_new)

    def candidate_cluster_ids_for_name(self, name: str) -> Set[str]:
        norm_name = normalize_text(name)
        if not norm_name:
            return set()

        candidates: Set[str] = set()
        mapped = self.name_to_cluster.get(norm_name)
        if mapped:
            candidates.add(self.resolve_cluster_id(mapped))

        bucket = self.ambiguous_name_to_clusters.get(norm_name, set())
        for cid in bucket:
            candidates.add(self.resolve_cluster_id(cid))

        return {
            cid
            for cid in candidates
            if cid in self.clusters and self.clusters[cid].is_active
        }

    def cluster_source_ids(self, cluster_id: str) -> Set[int]:
        root = self.resolve_cluster_id(cluster_id)
        cluster = self.clusters.get(root)
        if not cluster or not cluster.is_active:
            return set()
        out: Set[int] = set()
        for _action, sid in cluster.actions:
            if isinstance(sid, tuple):
                out.update(int(v) for v in sid if isinstance(v, int))
        return out

    def create_cluster(self, event: NodeEvent) -> str:
        cluster_id = self._next_cluster_id()
        action_items = event.action_entries or [(event.actions, event.source_id)]
        cluster = EntityCluster(
            cluster_id=cluster_id,
            canonical_name=event.main_name,
            classification=event.classification,
            alt_names=list(dict.fromkeys(event.alt_names)),
            kinship_aliases=list(dict.fromkeys(event.kinship_aliases)),
            actions=list(action_items),
            event_ids=[f"{event.chapter_id}:{event.event_idx}"],
            first_seen_chapter=event.chapter_id,
            last_seen_chapter=event.chapter_id,
        )
        self.clusters[cluster_id] = cluster
        for name in event.all_names:
            self._register_name(name, cluster_id)
        return cluster_id

    def attach_event(self, cluster_id: str, event: NodeEvent) -> None:
        root_id = self.resolve_cluster_id(cluster_id)
        cluster = self.clusters[root_id]
        cluster.last_seen_chapter = max(cluster.last_seen_chapter, event.chapter_id)
        action_items = event.action_entries or [(event.actions, event.source_id)]
        cluster.actions.extend(action_items)
        cluster.event_ids.append(f"{event.chapter_id}:{event.event_idx}")

        for name in event.alt_names:
            if name and name != cluster.canonical_name and name not in cluster.alt_names:
                cluster.alt_names.append(name)
        for marker in event.kinship_aliases:
            if marker and marker not in cluster.kinship_aliases:
                cluster.kinship_aliases.append(marker)
        if event.main_name != cluster.canonical_name and event.main_name not in cluster.alt_names:
            cluster.alt_names.append(event.main_name)

        for name in event.all_names:
            self._register_name(name, root_id)

    def merge_clusters(self, target_id: str, donor_id: str) -> str:
        t = self.resolve_cluster_id(target_id)
        d = self.resolve_cluster_id(donor_id)
        if t == d:
            return t
        target = self.clusters[t]
        donor = self.clusters[d]
        if not donor.is_active:
            return t

        for name in [donor.canonical_name] + donor.alt_names:
            if name and name != target.canonical_name and name not in target.alt_names:
                target.alt_names.append(name)
            self._register_name(name, t)
        for marker in donor.kinship_aliases:
            if marker and marker not in target.kinship_aliases:
                target.kinship_aliases.append(marker)
        target.actions.extend(donor.actions)
        target.event_ids.extend(donor.event_ids)
        target.first_seen_chapter = min(target.first_seen_chapter, donor.first_seen_chapter)
        target.last_seen_chapter = max(target.last_seen_chapter, donor.last_seen_chapter)

        donor.is_active = False
        donor.merged_into = t
        return t

    def get_active_clusters(self) -> Dict[str, EntityCluster]:
        return {cid: c for cid, c in self.clusters.items() if c.is_active}

    @staticmethod
    def _normalized_names(cluster: EntityCluster) -> List[str]:
        values = [cluster.canonical_name] + list(cluster.alt_names)
        unique = []
        seen = set()
        for value in values:
            norm = normalize_text(value)
            if not norm or norm in seen:
                continue
            unique.append(norm)
            seen.add(norm)
        return unique

    @staticmethod
    def _name_tokens(cluster: EntityCluster) -> set[str]:
        tokens = set()
        for name in [cluster.canonical_name] + list(cluster.alt_names):
            tokens |= {t for t in tokenize(name) if len(t) >= 3}
        return tokens

    @staticmethod
    def _context_tokens(cluster: EntityCluster) -> set[str]:
        return {t for t in tokenize(cluster.profile_text(last_n=12)) if len(t) >= 3}

    @staticmethod
    def _max_name_similarity(names_a: List[str], names_b: List[str]) -> float:
        best = 0.0
        for a in names_a:
            for b in names_b:
                if a == b:
                    return 1.0
                best = max(best, SequenceMatcher(None, a, b).ratio())
        return best

    @staticmethod
    def cluster_support_score(cluster: EntityCluster) -> float:
        chapter_ids = {
            sid_item
            for _action, sid in cluster.actions
            for sid_item in (sid or ())
            if isinstance(sid_item, int)
        }
        chapter_count = len(chapter_ids)
        return (1.0 * len(cluster.event_ids)) + (0.6 * chapter_count) + (0.2 * len(cluster.actions))

    def candidate_clusters_for_repair(
        self,
        top_k_per_cluster: int = 8,
        min_total_score: float = 0.34,
        max_token_bucket: int = 60,
        max_pair_candidates: int = 300,
    ) -> List[RepairCandidate]:
        active = self.get_active_clusters()
        if len(active) < 2:
            return []

        features = {}
        name_index: Dict[str, set[str]] = {}
        token_index: Dict[str, set[str]] = {}

        for cid, cluster in active.items():
            names = self._normalized_names(cluster)
            name_tokens = self._name_tokens(cluster)
            context_tokens = self._context_tokens(cluster)
            features[cid] = {
                "cluster": cluster,
                "names": names,
                "name_tokens": name_tokens,
                "context_tokens": context_tokens,
            }
            for name in names:
                name_index.setdefault(name, set()).add(cid)
            for token in name_tokens:
                token_index.setdefault(token, set()).add(cid)

        pair_best: Dict[Tuple[str, str], RepairCandidate] = {}

        for cid, payload in features.items():
            cluster = payload["cluster"]
            raw_candidates: set[str] = set()

            for name in payload["names"]:
                raw_candidates |= name_index.get(name, set())
            for token in payload["name_tokens"]:
                bucket = token_index.get(token, set())
                if len(bucket) <= max_token_bucket:
                    raw_candidates |= bucket
            raw_candidates.discard(cid)

            scored: List[RepairCandidate] = []
            for did in raw_candidates:
                if did not in features:
                    continue
                other = features[did]["cluster"]
                if other.classification != cluster.classification:
                    continue

                names_a = payload["names"]
                names_b = features[did]["names"]
                name_score = self._max_name_similarity(names_a, names_b)

                tokens_a = payload["name_tokens"]
                tokens_b = features[did]["name_tokens"]
                alias_score = jaccard(tokens_a, tokens_b)

                mention_score = 0.0
                if normalize_text(cluster.canonical_name) in names_b:
                    mention_score += 0.5
                if normalize_text(other.canonical_name) in names_a:
                    mention_score += 0.5

                context_score = jaccard(payload["context_tokens"], features[did]["context_tokens"])

                total = (
                    (0.45 * name_score)
                    + (0.30 * alias_score)
                    + (0.15 * mention_score)
                    + (0.10 * context_score)
                )
                if total < min_total_score and mention_score < 1.0 and name_score < 0.86:
                    continue

                a, b = sorted([cid, did])
                scored.append(
                    RepairCandidate(
                        cluster_a_id=a,
                        cluster_b_id=b,
                        total_score=total,
                        name_score=name_score,
                        alias_score=alias_score,
                        mention_score=mention_score,
                        context_score=context_score,
                    )
                )

            scored.sort(
                key=lambda x: (x.total_score, x.mention_score, x.name_score, x.alias_score),
                reverse=True,
            )
            for cand in scored[:top_k_per_cluster]:
                key = (cand.cluster_a_id, cand.cluster_b_id)
                prev = pair_best.get(key)
                if prev is None or cand.total_score > prev.total_score:
                    pair_best[key] = cand

        ranked = sorted(
            pair_best.values(),
            key=lambda x: (
                x.total_score,
                x.mention_score,
                x.name_score,
                x.alias_score,
            ),
            reverse=True,
        )
        return ranked[:max_pair_candidates]

    def candidate_clusters_for_gap_bridge(self, max_gap: int = 5) -> List[Tuple[str, str]]:
        buckets: Dict[str, List[EntityCluster]] = {}
        for cluster in self.get_active_clusters().values():
            key = normalize_text(cluster.canonical_name)
            buckets.setdefault(key, []).append(cluster)

        pairs: List[Tuple[str, str]] = []
        for members in buckets.values():
            if len(members) < 2:
                continue
            members = sorted(members, key=lambda c: c.first_seen_chapter)
            for i in range(len(members) - 1):
                a, b = members[i], members[i + 1]
                gap = b.first_seen_chapter - a.last_seen_chapter
                if gap <= max_gap:
                    pairs.append((a.cluster_id, b.cluster_id))
        return pairs
