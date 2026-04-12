from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class NodeEvent:
    source_id: Tuple[int, ...]
    chapter_id: int
    main_name: str
    alt_names: List[str]
    kinship_aliases: List[str]
    classification: str
    actions: str
    event_idx: int
    action_entries: List[Tuple[str, Tuple[int, ...]]] = field(default_factory=list)

    @property
    def all_names(self) -> List[str]:
        return [self.main_name] + self.alt_names


@dataclass
class CandidateScore:
    cluster_id: str
    total_score: float
    name_score: float
    alias_score: float
    context_score: float


@dataclass
class RepairCandidate:
    cluster_a_id: str
    cluster_b_id: str
    total_score: float
    name_score: float
    alias_score: float
    mention_score: float
    context_score: float


@dataclass
class EntityCluster:
    cluster_id: str
    canonical_name: str
    classification: str
    alt_names: List[str] = field(default_factory=list)
    kinship_aliases: List[str] = field(default_factory=list)
    actions: List[Tuple[str, Tuple[int, ...]]] = field(default_factory=list)
    event_ids: List[str] = field(default_factory=list)
    first_seen_chapter: int = 0
    last_seen_chapter: int = 0
    is_active: bool = True
    merged_into: str | None = None

    def profile_text(self, last_n: int = 10) -> str:
        payload = [a for a, _sid in self.actions[-last_n:]]
        return ". ".join(payload)


@dataclass
class VerificationDecision:
    accepted_cluster_id: str | None
    candidates_checked: List[CandidateScore]
    judge_traces: List[Dict]
    reason: str
