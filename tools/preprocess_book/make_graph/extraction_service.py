import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from tools.preprocess_book.config.preprocess_settings import (
    BATCH_STRICT_MERGE,
    BATCH_SIZE,
    ENABLE_PRE_NER_HELPER,
    EXTRACTION_CHAPTER_BATCH_SIZE,
    EXTRACTION_SMALL_CHUNK_OVERLAP,
    EXTRACTION_SMALL_CHUNK_SIZE,
    EXTRACTION_USE_SMALL_CHUNKS,
    PARALLEL_CONCURRENCY,
    PRE_NER_ENABLED,
    PRE_NER_COVERAGE_RETRY,
    PRE_NER_MAX_ENTITIES_PER_BATCH,
    RESULTS_DIR,
)
from tools.preprocess_book.make_graph.pre_ner_helper import (
    build_required_entities_from_chapter_blocks,
)
from tools.preprocess_book.prompts.extract_names import (
    ExtractedNode,
    ExtractedRelation,
    ExtractionPayload,
)

logger = logging.getLogger(__name__)


class ExtractionService:
    """Service for extracting entities and relations from chapter texts."""

    def __init__(self, extractor, extractor_relations=None, extractor_missing_nodes=None):
        if extractor_relations is None:
            raise ValueError("extractor_relations is required for two-stage extraction")
        self.extractor = extractor
        self.extractor_relations = extractor_relations
        self.extractor_missing_nodes = extractor_missing_nodes

    @staticmethod
    def _first_source_id(source_id: Any) -> Optional[int]:
        if isinstance(source_id, int):
            return source_id
        if isinstance(source_id, tuple) and source_id and isinstance(source_id[0], int):
            return source_id[0]
        if isinstance(source_id, list) and source_id and isinstance(source_id[0], int):
            return source_id[0]
        return None

    def _as_dict(self, result: Any) -> Optional[Dict[str, Any]]:
        return self._as_dict_with(self.extractor, result)

    @staticmethod
    def _as_dict_with(extractor: Any, result: Any) -> Optional[Dict[str, Any]]:
        if isinstance(result, dict):
            return dict(result)

        raw_text = None
        if isinstance(result, str):
            raw_text = result
        elif hasattr(result, "content"):
            content = result.content
            if isinstance(content, str):
                raw_text = content
            elif isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict):
                        t = item.get("text")
                        if isinstance(t, str):
                            text_parts.append(t)
                raw_text = "\n".join(text_parts).strip()

        if raw_text:
            try:
                parsed = extractor.parse_json_content(raw_text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                # For message-like outputs we expect JSON in content.
                # If parsing fails, force retry path instead of silent empty payload.
                return None

        if isinstance(result, BaseModel):
            if hasattr(result, "model_dump"):
                return result.model_dump(mode="json")
            if hasattr(result, "dict"):
                return result.dict()
        return None

    @staticmethod
    def _coerce_evidence_items(items: Any, fallback_source_id: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        if isinstance(items, str):
            text = items.strip()
            if text:
                out.append(
                    {
                        "source_id": fallback_source_id,
                        "chapter_id": fallback_source_id,
                        "description": text,
                    }
                )
            return out

        if not isinstance(items, list):
            return out

        for item in items:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    out.append(
                        {
                            "source_id": fallback_source_id,
                            "chapter_id": fallback_source_id,
                            "description": text,
                        }
                    )
                continue

            if not isinstance(item, dict):
                continue

            sid = item.get("source_id", fallback_source_id)
            cid = item.get("chapter_id", sid)
            desc = (item.get("description") or "").strip()

            if not isinstance(sid, int):
                sid = fallback_source_id
            if not isinstance(cid, int):
                cid = sid
            if not desc:
                continue

            out.append(
                {
                    "source_id": sid,
                    "chapter_id": cid,
                    "description": desc,
                }
            )

        return out

    def _coerce_payload_schema(self, payload: Dict[str, Any], source_id: Any) -> Dict[str, Any]:
        fallback_source_id = self._first_source_id(source_id) or 0

        nodes_in = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
        relations_in = payload.get("relations") if isinstance(payload.get("relations"), list) else []

        nodes: List[Dict[str, Any]] = []
        for node in nodes_in:
            if not isinstance(node, dict):
                continue
            raw_classification = node.get("classification")
            raw_classification = self._normalize_classification(raw_classification)
            nodes.append(
                {
                    "main_name": str(node.get("main_name", "")).strip(),
                    "alt_names": node.get("alt_names") if isinstance(node.get("alt_names"), list) else [],
                    "actions": self._coerce_evidence_items(node.get("actions"), fallback_source_id),
                    "classification": raw_classification,
                }
            )

        relations: List[Dict[str, Any]] = []
        for rel in relations_in:
            if not isinstance(rel, dict):
                continue
            descriptions = rel.get("descriptions")
            if descriptions is None and rel.get("description") is not None:
                descriptions = rel.get("description")
            source_node_type = str(rel.get("source_node_type", "")).strip() or "термин"
            target_node_type = str(rel.get("target_node_type", "")).strip() or "термин"
            rel_type = str(rel.get("type", "")).strip() or "СВЯЗАНО_С"

            relations.append(
                {
                    "source_node_id": str(rel.get("source_node_id", "")).strip(),
                    "source_node_type": source_node_type,
                    "target_node_id": str(rel.get("target_node_id", "")).strip(),
                    "target_node_type": target_node_type,
                    "type": rel_type,
                    "descriptions": self._coerce_evidence_items(descriptions, fallback_source_id),
                }
            )

        return {
            "nodes": nodes,
            "relations": relations,
            "summarization": str(payload.get("summarization", "") or ""),
        }

    @staticmethod
    def _normalize_classification(value: Any) -> str:
        raw = str(value or "").strip().lower().replace("ё", "е")
        if not raw:
            return "термин"
        mapping = {
            "персонаж": "персонаж",
            "персонажи": "персонаж",
            "person": "персонаж",
            "character": "персонаж",
            "место": "место",
            "location": "место",
            "place": "место",
            "организация": "организация",
            "org": "организация",
            "organization": "организация",
            "термин": "термин",
            "term": "термин",
            "сила природы": "сила природы",
            "force": "сила природы",
            "животное": "термин",
            "зверь": "термин",
        }
        return mapping.get(raw, "термин")

    @staticmethod
    def _norm_name(name: Any) -> str:
        if not isinstance(name, str):
            return ""
        return " ".join(name.strip().lower().replace("ё", "е").split())

    @staticmethod
    def _is_capitalized_name(name: Any) -> bool:
        """
        For fast in-batch merge we trust alias-overlap only for
        aliases that look like proper names (start with uppercase).
        """
        if not isinstance(name, str):
            return False
        s = name.strip()
        if not s:
            return False
        for ch in s:
            if ch.isalpha():
                return ch.isupper()
        return False

    @classmethod
    def _entity_match_key(cls, name: Any) -> str:
        """
        Match key with lightweight case normalization:
        supports checks like "?????????" -> "????????".
        """
        n = cls._norm_name(name)
        if not n:
            return ""
        suffixes = (
            "????",
            "???",
            "???",
            "???",
            "???",
            "???",
            "???",
            "??",
            "??",
            "??",
            "??",
            "??",
            "??",
            "??",
            "??",
            "??",
            "??",
            "??",
            "??",
            "??",
            "?",
            "?",
            "?",
            "?",
            "?",
            "?",
            "?",
            "?",
        )
        for suffix in suffixes:
            if len(n) > len(suffix) + 2 and n.endswith(suffix):
                return n[: -len(suffix)]
        return n

    @classmethod
    def _payload_entity_keys(cls, payload: Dict[str, Any]) -> set[str]:
        keys: set[str] = set()
        nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            main = node.get("main_name")
            k_main = cls._entity_match_key(main)
            if k_main:
                keys.add(k_main)
            for alias in node.get("alt_names") if isinstance(node.get("alt_names"), list) else []:
                k_alias = cls._entity_match_key(alias)
                if k_alias:
                    keys.add(k_alias)
        return keys

    @staticmethod
    def _dedupe_actions(actions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[tuple] = set()
        for action in actions:
            if not isinstance(action, dict):
                continue
            sid = action.get("source_id")
            cid = action.get("chapter_id")
            desc = str(action.get("description") or "").strip()
            if not isinstance(sid, int) or not isinstance(cid, int) or not desc:
                continue
            key = (sid, cid, desc)
            if key in seen:
                continue
            seen.add(key)
            out.append(action)
        return out

    @staticmethod
    def _normalize_required_class(value: Any) -> str:
        raw = str(value or "").strip().lower().replace("ё", "е")
        if not raw:
            return "термин"
        if raw in {"персонаж", "person", "character"}:
            return "персонаж"
        if raw in {"место", "place", "location"}:
            return "место"
        if raw in {"организация", "org", "organization"}:
            return "организация"
        if raw in {"термин", "term"}:
            return "термин"
        return "термин"

    def _group_missing_mentions_for_retry(
        self,
        missing_required: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """
        Group missing mentions by semantic class for targeted retry.
        Priority when one mention appears in multiple classes:
        персонаж > место > организация > термин.
        """
        class_priority = {
            "персонаж": 0,
            "место": 1,
            "организация": 2,
            "термин": 3,
        }

        mention_to_class: Dict[str, str] = {}
        mention_to_rank: Dict[str, int] = {}
        mention_to_original: Dict[str, str] = {}

        for item in missing_required:
            if not isinstance(item, dict):
                continue
            mention = str(item.get("main_name", "")).strip()
            if not mention:
                continue
            cls = self._normalize_required_class(item.get("classification"))
            rank = class_priority.get(cls, 3)
            key = self._norm_name(mention)
            if not key:
                continue
            if key not in mention_to_class or rank < mention_to_rank.get(key, 99):
                mention_to_class[key] = cls
                mention_to_rank[key] = rank
                mention_to_original[key] = mention

        buckets: Dict[str, List[str]] = {
            "персонаж": [],
            "место": [],
            "организация": [],
            "термин": [],
        }
        for key, cls in mention_to_class.items():
            buckets.setdefault(cls, []).append(mention_to_original[key])
        return buckets

    def _merge_retry_payload(
        self,
        base_payload: Dict[str, Any],
        patch_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_nodes = (
            [dict(x) for x in base_payload.get("nodes", [])]
            if isinstance(base_payload.get("nodes"), list)
            else []
        )
        patch_nodes = (
            patch_payload.get("nodes", [])
            if isinstance(patch_payload.get("nodes"), list)
            else []
        )

        def _node_keys(node: Dict[str, Any]) -> set[str]:
            keys: set[str] = set()
            k_main = self._entity_match_key(node.get("main_name"))
            if k_main:
                keys.add(k_main)
            for alt in node.get("alt_names") if isinstance(node.get("alt_names"), list) else []:
                k_alt = self._entity_match_key(alt)
                if k_alt:
                    keys.add(k_alt)
            return keys

        for patch_node in patch_nodes:
            if not isinstance(patch_node, dict):
                continue
            pkeys = _node_keys(patch_node)
            if not pkeys:
                continue
            target_idx: Optional[int] = None
            for idx, base_node in enumerate(base_nodes):
                if pkeys.intersection(_node_keys(base_node)):
                    target_idx = idx
                    break

            if target_idx is None:
                base_nodes.append(dict(patch_node))
                continue

            base_node = base_nodes[target_idx]
            merged_alt = list(base_node.get("alt_names") or [])
            for alt in patch_node.get("alt_names") if isinstance(patch_node.get("alt_names"), list) else []:
                if isinstance(alt, str) and alt.strip() and alt not in merged_alt:
                    merged_alt.append(alt)
            base_node["alt_names"] = merged_alt

            merged_actions = list(base_node.get("actions") or [])
            merged_actions.extend(patch_node.get("actions") if isinstance(patch_node.get("actions"), list) else [])
            base_node["actions"] = self._dedupe_actions(merged_actions)

            if (not str(base_node.get("classification") or "").strip()) and str(
                patch_node.get("classification") or ""
            ).strip():
                base_node["classification"] = patch_node.get("classification")

            base_nodes[target_idx] = base_node

        return {
            "nodes": base_nodes,
            "relations": (
                base_payload.get("relations", [])
                if isinstance(base_payload.get("relations"), list)
                else []
            ),
            "summarization": str(base_payload.get("summarization", "") or ""),
        }

    def _append_missing_nodes_payload(
        self,
        base_payload: Dict[str, Any],
        patch_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Lightweight merge for missing-node retry:
        - append new nodes from patch;
        - if main_name already exists, only append actions (deduped);
        - no fuzzy entity merge by alt_names.
        """
        base_nodes = (
            [dict(x) for x in base_payload.get("nodes", [])]
            if isinstance(base_payload.get("nodes"), list)
            else []
        )
        patch_nodes = (
            patch_payload.get("nodes", [])
            if isinstance(patch_payload.get("nodes"), list)
            else []
        )

        index_by_main: Dict[str, int] = {}
        for idx, node in enumerate(base_nodes):
            if not isinstance(node, dict):
                continue
            key = self._norm_name(node.get("main_name"))
            if key:
                index_by_main[key] = idx

        for node in patch_nodes:
            if not isinstance(node, dict):
                continue
            main_name = str(node.get("main_name", "")).strip()
            if not main_name:
                continue
            key = self._norm_name(main_name)
            if key and key in index_by_main:
                idx = index_by_main[key]
                base_node = base_nodes[idx]
                merged_actions = list(base_node.get("actions") or [])
                merged_actions.extend(
                    node.get("actions")
                    if isinstance(node.get("actions"), list)
                    else []
                )
                base_node["actions"] = self._dedupe_actions(merged_actions)
                base_nodes[idx] = base_node
            else:
                base_nodes.append(dict(node))
                if key:
                    index_by_main[key] = len(base_nodes) - 1

        return {
            "nodes": base_nodes,
            "relations": (
                base_payload.get("relations", [])
                if isinstance(base_payload.get("relations"), list)
                else []
            ),
            "summarization": str(base_payload.get("summarization", "") or ""),
        }

    def _append_relations_payload(
        self,
        base_payload: Dict[str, Any],
        patch_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        base_relations = (
            [dict(x) for x in base_payload.get("relations", [])]
            if isinstance(base_payload.get("relations"), list)
            else []
        )
        patch_relations = (
            patch_payload.get("relations", [])
            if isinstance(patch_payload.get("relations"), list)
            else []
        )

        def _rel_key(rel: Dict[str, Any]) -> tuple[str, str, str]:
            return (
                self._norm_name(rel.get("source_node_id")),
                self._norm_name(rel.get("target_node_id")),
                str(rel.get("type", "")).strip().lower(),
            )

        index_by_key: Dict[tuple[str, str, str], int] = {}
        for idx, rel in enumerate(base_relations):
            if not isinstance(rel, dict):
                continue
            key = _rel_key(rel)
            if key[0] and key[1] and key[2]:
                index_by_key[key] = idx

        for rel in patch_relations:
            if not isinstance(rel, dict):
                continue
            key = _rel_key(rel)
            if not (key[0] and key[1] and key[2]):
                continue

            if key in index_by_key:
                idx = index_by_key[key]
                base_rel = base_relations[idx]
                merged_desc = list(base_rel.get("descriptions") or [])
                merged_desc.extend(
                    rel.get("descriptions")
                    if isinstance(rel.get("descriptions"), list)
                    else []
                )
                base_rel["descriptions"] = self._dedupe_actions(merged_desc)
                if not str(base_rel.get("source_node_type", "")).strip():
                    base_rel["source_node_type"] = rel.get("source_node_type", "")
                if not str(base_rel.get("target_node_type", "")).strip():
                    base_rel["target_node_type"] = rel.get("target_node_type", "")
                base_relations[idx] = base_rel
            else:
                rel_copy = dict(rel)
                rel_copy["descriptions"] = self._dedupe_actions(
                    rel_copy.get("descriptions")
                    if isinstance(rel_copy.get("descriptions"), list)
                    else []
                )
                base_relations.append(rel_copy)
                index_by_key[key] = len(base_relations) - 1

        return {
            "nodes": (
                base_payload.get("nodes", [])
                if isinstance(base_payload.get("nodes"), list)
                else []
            ),
            "relations": base_relations,
            "summarization": str(base_payload.get("summarization", "") or ""),
        }

    def _append_extraction_payloads(
        self,
        base_payload: Dict[str, Any],
        patch_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = self._append_chunk_nodes_payload(base_payload, patch_payload)
        merged = self._append_relations_payload(merged, patch_payload)
        if not str(merged.get("summarization", "")).strip():
            merged["summarization"] = str(patch_payload.get("summarization", "") or "")
        return merged

    @staticmethod
    def _action_keys(node: Dict[str, Any]) -> set[tuple]:
        out: set[tuple] = set()
        for act in node.get("actions") if isinstance(node.get("actions"), list) else []:
            if not isinstance(act, dict):
                continue
            sid = act.get("source_id")
            cid = act.get("chapter_id")
            desc = str(act.get("description", "")).strip().lower()
            if isinstance(sid, int) and isinstance(cid, int) and desc:
                out.add((sid, cid, desc))
        return out

    @staticmethod
    def _action_scopes(node: Dict[str, Any]) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for act in node.get("actions") if isinstance(node.get("actions"), list) else []:
            if not isinstance(act, dict):
                continue
            sid = act.get("source_id")
            cid = act.get("chapter_id")
            if isinstance(sid, int) and isinstance(cid, int):
                out.add((sid, cid))
        return out

    @staticmethod
    def _action_token_set(node: Dict[str, Any]) -> set[str]:
        tokens: set[str] = set()
        for act in node.get("actions") if isinstance(node.get("actions"), list) else []:
            if not isinstance(act, dict):
                continue
            desc = str(act.get("description", "")).strip().lower().replace("ё", "е")
            if not desc:
                continue
            for token in re.findall(r"\w+", desc, flags=re.UNICODE):
                token = token.strip()
                if len(token) >= 3:
                    tokens.add(token)
        return tokens

    def _build_name_set(self, node: Dict[str, Any]) -> set[str]:
        names: set[str] = set()
        main = self._norm_name(node.get("main_name"))
        if main:
            names.add(main)
        for alias in (
            node.get("alt_names") if isinstance(node.get("alt_names"), list) else []
        ):
            alias_norm = self._norm_name(alias)
            if alias_norm and self._is_capitalized_name(alias):
                names.add(alias_norm)
        return names

    @staticmethod
    def _class_bucket(value: Any) -> str:
        raw = str(value or "").strip().lower().replace("ё", "е")
        if not raw:
            return "термин"
        if "персона" in raw or raw in {"person", "character"}:
            return "персонаж"
        if "мест" in raw or raw in {"place", "location"}:
            return "место"
        if "орган" in raw or raw in {"org", "organization"}:
            return "организация"
        if "сила" in raw or raw in {"force"}:
            return "сила природы"
        return "термин"

    def _has_hard_class_conflict(self, base_node: Dict[str, Any], patch_node: Dict[str, Any]) -> bool:
        base_cls = self._class_bucket(base_node.get("classification"))
        patch_cls = self._class_bucket(patch_node.get("classification"))
        if base_cls == patch_cls:
            return False
        # TERM is weak and can be merged into stronger class if other evidence matches.
        if base_cls == "термин" or patch_cls == "термин":
            return False
        return True

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a.intersection(b))
        union = len(a.union(b))
        return (inter / union) if union else 0.0

    def _should_merge_nodes_fast(
        self,
        base_node: Dict[str, Any],
        patch_node: Dict[str, Any],
    ) -> bool:
        """
        Deterministic cheap merge for batch chunks:
        - requires alias overlap or exact evidence overlap;
        - rejects hard class conflicts;
        - protects homonyms where only main_name matches.
        """
        base_names = self._build_name_set(base_node)
        patch_names = self._build_name_set(patch_node)
        if not base_names or not patch_names:
            return False

        if self._has_hard_class_conflict(base_node, patch_node):
            return False

        name_overlap = base_names.intersection(patch_names)
        action_overlap = bool(self._action_keys(base_node).intersection(self._action_keys(patch_node)))
        if not name_overlap and not action_overlap:
            return False

        base_main = self._norm_name(base_node.get("main_name"))
        patch_main = self._norm_name(patch_node.get("main_name"))
        # If overlap is only on identical main_name and no other signal -> do not merge.
        same_main = bool(base_main and patch_main and base_main == patch_main)
        base_cls = self._class_bucket(base_node.get("classification"))
        patch_cls = self._class_bucket(patch_node.get("classification"))
        # Safe shortcut for in-batch duplicates: same main + same class + same chapter/source scope.
        if same_main and base_cls == patch_cls:
            if self._action_scopes(base_node).intersection(self._action_scopes(patch_node)):
                return True

        if (
            name_overlap
            and not action_overlap
            and len(name_overlap) == 1
            and same_main
            and list(name_overlap)[0] == base_main
        ):
            return False

        # Context guard for strict mode.
        if BATCH_STRICT_MERGE:
            ctx_a = self._action_token_set(base_node)
            ctx_b = self._action_token_set(patch_node)
            ctx_sim = self._jaccard(ctx_a, ctx_b)
            if not action_overlap and ctx_sim < 0.08 and len(name_overlap) < 2:
                return False

        return True

    def _append_chunk_nodes_payload(
        self,
        base_payload: Dict[str, Any],
        patch_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Append nodes from chunk-level extraction without forced merge by main_name.
        """
        base_nodes = (
            [dict(x) for x in base_payload.get("nodes", [])]
            if isinstance(base_payload.get("nodes"), list)
            else []
        )
        patch_nodes = (
            patch_payload.get("nodes", [])
            if isinstance(patch_payload.get("nodes"), list)
            else []
        )

        for patch_node in patch_nodes:
            if not isinstance(patch_node, dict):
                continue
            main_name = str(patch_node.get("main_name", "")).strip()
            if not main_name:
                continue

            target_idx: Optional[int] = None
            best_score = -1.0
            for idx, base_node in enumerate(base_nodes):
                if not isinstance(base_node, dict):
                    continue
                if not self._should_merge_nodes_fast(base_node, patch_node):
                    continue
                name_overlap = len(self._build_name_set(base_node).intersection(self._build_name_set(patch_node)))
                action_overlap = len(self._action_keys(base_node).intersection(self._action_keys(patch_node)))
                score = (2.0 * name_overlap) + (3.0 * action_overlap)
                if score > best_score:
                    best_score = score
                    target_idx = idx

            if target_idx is None:
                base_nodes.append(dict(patch_node))
                continue

            base_node = base_nodes[target_idx]
            merged_actions = list(base_node.get("actions") or [])
            merged_actions.extend(
                patch_node.get("actions")
                if isinstance(patch_node.get("actions"), list)
                else []
            )
            base_node["actions"] = self._dedupe_actions(merged_actions)

            merged_alt = list(base_node.get("alt_names") or [])
            seen_alt = {self._norm_name(x) for x in merged_alt if self._norm_name(x)}
            for alt in (
                patch_node.get("alt_names")
                if isinstance(patch_node.get("alt_names"), list)
                else []
            ):
                alt_s = str(alt).strip()
                alt_k = self._norm_name(alt_s)
                if alt_s and alt_k and alt_k not in seen_alt:
                    merged_alt.append(alt_s)
                    seen_alt.add(alt_k)
            base_node["alt_names"] = merged_alt

            if not str(base_node.get("classification", "")).strip():
                base_node["classification"] = patch_node.get("classification", "")

            base_nodes[target_idx] = base_node

        return {
            "nodes": base_nodes,
            "relations": (
                base_payload.get("relations", [])
                if isinstance(base_payload.get("relations"), list)
                else []
            ),
            "summarization": str(base_payload.get("summarization", "") or ""),
        }

    @staticmethod
    def _split_text_for_small_chunk_extraction(text: str) -> List[str]:
        raw_text = str(text or "").strip()
        if not raw_text:
            return []
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=max(256, int(EXTRACTION_SMALL_CHUNK_SIZE or 4000)),
            chunk_overlap=max(0, int(EXTRACTION_SMALL_CHUNK_OVERLAP or 250)),
        )
        chunks = [c.strip() for c in splitter.split_text(raw_text) if str(c).strip()]
        return chunks or [raw_text]

    def _build_small_chunk_blocks(
        self, chapter_blocks: Sequence[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        out: List[List[Dict[str, Any]]] = []
        for block in chapter_blocks:
            sid = block.get("source_id")
            cid = block.get("chapter_id", sid)
            text = block.get("text")
            if not isinstance(sid, int) or not isinstance(cid, int) or not isinstance(text, str):
                continue
            chunks = self._split_text_for_small_chunk_extraction(text)
            for chunk_idx, chunk_text in enumerate(chunks):
                out.append(
                    [
                        {
                            "source_id": sid,
                            "chapter_id": cid,
                            "text": chunk_text,
                            "chunk_idx": chunk_idx,
                        }
                    ]
                )
        return out

    def _build_required_entities(
        self, chapter_blocks: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not PRE_NER_ENABLED:
            return []
        if not ENABLE_PRE_NER_HELPER:
            return []
        return build_required_entities_from_chapter_blocks(
            chapter_blocks=chapter_blocks,
            max_entities_per_batch=PRE_NER_MAX_ENTITIES_PER_BATCH,
        )

    def _find_missing_required_entities(
        self,
        payload: Dict[str, Any],
        required_entities: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not required_entities:
            return []
        present = self._payload_entity_keys(payload)
        missing: List[Dict[str, Any]] = []
        for req in required_entities:
            match_key = str(req.get("match_key") or self._entity_match_key(req.get("main_name")))
            if not match_key:
                continue
            if match_key not in present:
                missing.append(req)
        return missing

    @staticmethod
    def _inject_required_entities_as_fallback(
        payload: Dict[str, Any],
        required_entities: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
        relations = payload.get("relations") if isinstance(payload.get("relations"), list) else []
        present_names = {
            str(node.get("main_name", "")).strip()
            for node in nodes
            if isinstance(node, dict)
        }

        added = 0
        for req in required_entities:
            main_name = str(req.get("main_name", "")).strip()
            if not main_name or main_name in present_names:
                continue
            actions = req.get("actions") if isinstance(req.get("actions"), list) else []
            nodes.append(
                {
                    "main_name": main_name,
                    "alt_names": [],
                    "actions": actions,
                    "classification": str(req.get("classification") or "??????"),
                }
            )
            present_names.add(main_name)
            added += 1

        if added:
            logger.warning(
                "Required-entity fallback injected %s nodes after retries",
                added,
            )

        return {
            "nodes": nodes,
            "relations": relations,
            "summarization": str(payload.get("summarization", "") or ""),
        }

    @staticmethod
    def _classification_from_endpoint_type(node_type: Any) -> str:
        raw = str(node_type or "").strip()
        if not raw:
            return "\u0442\u0435\u0440\u043c\u0438\u043d"

        explicit = {
            "EntityClassification.PERSON": "\u043f\u0435\u0440\u0441\u043e\u043d\u0430\u0436",
            "EntityClassification.PLACE": "\u043c\u0435\u0441\u0442\u043e",
            "EntityClassification.ORG": "\u043e\u0440\u0433\u0430\u043d\u0438\u0437\u0430\u0446\u0438\u044f",
            "EntityClassification.TERM": "\u0442\u0435\u0440\u043c\u0438\u043d",
            "EntityClassification.FORCE": "\u0441\u0438\u043b\u0430 \u043f\u0440\u0438\u0440\u043e\u0434\u044b",
        }
        if raw in explicit:
            return explicit[raw]

        raw_l = raw.lower().replace("\u0451", "\u0435")
        if any(token in raw_l for token in ("\u043f\u0435\u0440\u0441\u043e\u043d\u0430", "person", "character")):
            return "\u043f\u0435\u0440\u0441\u043e\u043d\u0430\u0436"
        if any(token in raw_l for token in ("\u043c\u0435\u0441\u0442", "place", "location")):
            return "\u043c\u0435\u0441\u0442\u043e"
        if any(token in raw_l for token in ("\u043e\u0440\u0433\u0430\u043d", "org")):
            return "\u043e\u0440\u0433\u0430\u043d\u0438\u0437\u0430\u0446\u0438\u044f"
        if any(token in raw_l for token in ("\u0441\u0438\u043b\u0430", "force", "nature")):
            return "\u0441\u0438\u043b\u0430 \u043f\u0440\u0438\u0440\u043e\u0434\u044b"
        return "\u0442\u0435\u0440\u043c\u0438\u043d"

    @staticmethod
    def _extract_parenthetical_aliases(name: str) -> List[str]:
        if not isinstance(name, str):
            return []
        aliases: List[str] = []
        for chunk in re.findall(r"\(([^)]+)\)", name):
            for part in re.split(r"[,/;]| или | and ", chunk):
                val = (part or "").strip()
                if val:
                    aliases.append(val)
        return aliases

    def _merge_identity_alias_nodes(
        self, payload: Dict[str, Any], source_id: Any
    ) -> Dict[str, Any]:
        """
        Merge duplicates within one extraction payload using deterministic
        cheap guards (name-set/evidence/context overlap + class conflict checks).
        """
        nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
        relations = (
            payload.get("relations") if isinstance(payload.get("relations"), list) else []
        )
        if len(nodes) < 2:
            return payload

        n = len(nodes)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        merge_accepted = 0
        merge_rejected = 0
        for i in range(n):
            for j in range(i + 1, n):
                if not isinstance(nodes[i], dict) or not isinstance(nodes[j], dict):
                    continue
                if self._should_merge_nodes_fast(nodes[i], nodes[j]):
                    union(i, j)
                    merge_accepted += 1
                else:
                    merge_rejected += 1

        groups: Dict[int, List[int]] = {}
        for idx in range(n):
            groups.setdefault(find(idx), []).append(idx)

        if all(len(g) == 1 for g in groups.values()):
            return payload

        merged_nodes: List[Dict[str, Any]] = []
        name_remap: Dict[str, str] = {}

        for _, idxs in groups.items():
            base_idx = sorted(
                idxs,
                key=lambda k: (
                    len(str(nodes[k].get("main_name", ""))),
                    len(nodes[k].get("actions") or []),
                ),
                reverse=True,
            )[0]
            base = dict(nodes[base_idx])
            canonical = str(base.get("main_name", "")).strip()
            if not canonical:
                continue

            alt_set: set[str] = set(str(a).strip() for a in (base.get("alt_names") or []) if str(a).strip())
            actions = list(base.get("actions") or [])

            for idx in idxs:
                node = nodes[idx]
                main = str(node.get("main_name", "")).strip()
                if main and main != canonical:
                    alt_set.add(main)
                for alias in node.get("alt_names") or []:
                    alias_val = str(alias).strip()
                    if alias_val and alias_val != canonical:
                        alt_set.add(alias_val)
                for act in node.get("actions") or []:
                    if isinstance(act, dict):
                        actions.append(act)
                if main:
                    name_remap[self._norm_name(main)] = canonical

            seen = set()
            dedup_actions: List[Dict[str, Any]] = []
            for act in actions:
                if not isinstance(act, dict):
                    continue
                key = (
                    act.get("source_id"),
                    act.get("chapter_id"),
                    str(act.get("description", "")).strip(),
                )
                if key in seen:
                    continue
                seen.add(key)
                dedup_actions.append(act)

            base["alt_names"] = sorted(alt_set)
            base["actions"] = dedup_actions
            merged_nodes.append(base)

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            s = self._norm_name(rel.get("source_node_id"))
            t = self._norm_name(rel.get("target_node_id"))
            if s in name_remap:
                rel["source_node_id"] = name_remap[s]
            if t in name_remap:
                rel["target_node_id"] = name_remap[t]

        logger.info(
            "Identity alias merge for source_id=%s: nodes %s -> %s accepted=%s rejected=%s",
            source_id,
            len(nodes),
            len(merged_nodes),
            merge_accepted,
            merge_rejected,
        )
        return {
            "nodes": merged_nodes,
            "relations": relations,
            "summarization": str(payload.get("summarization", "") or ""),
        }

    def _enforce_nodes_relations_closure(
        self, payload: Dict[str, Any], source_id: Any
    ) -> Dict[str, Any]:
        """
        Enforce invariant:
        - every relation endpoint has a node;
        - every node participates in at least one relation endpoint.
        """
        fallback_source_id = self._first_source_id(source_id) or 0
        nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
        relations = (
            payload.get("relations") if isinstance(payload.get("relations"), list) else []
        )

        node_by_norm: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            main_name = str(node.get("main_name", "")).strip()
            n = self._norm_name(main_name)
            if n and n not in node_by_norm:
                node_by_norm[n] = node

        endpoint_norms: set[str] = set()
        endpoint_meta: Dict[str, Dict[str, Any]] = {}
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            for side in ("source", "target"):
                id_key = f"{side}_node_id"
                type_key = f"{side}_node_type"
                endpoint = str(rel.get(id_key, "")).strip()
                n = self._norm_name(endpoint)
                if not n:
                    continue
                endpoint_norms.add(n)
                if n not in endpoint_meta:
                    endpoint_meta[n] = {
                        "main_name": endpoint,
                        "classification": self._classification_from_endpoint_type(
                            rel.get(type_key)
                        ),
                        "evidence": rel.get("descriptions") or [],
                    }

        missing_endpoint_nodes = [n for n in endpoint_norms if n not in node_by_norm]
        for norm_name in missing_endpoint_nodes:
            meta = endpoint_meta.get(norm_name, {})
            main_name = str(meta.get("main_name") or norm_name).strip() or norm_name
            evidence = meta.get("evidence")
            actions = []
            if isinstance(evidence, list) and evidence:
                first = evidence[0]
                if isinstance(first, dict):
                    actions = [
                        {
                            "source_id": int(first.get("source_id", fallback_source_id)),
                            "chapter_id": int(
                                first.get("chapter_id", first.get("source_id", fallback_source_id))
                            ),
                            "description": str(first.get("description", "Упомянут в отношениях")).strip()
                            or "Упомянут в отношениях",
                        }
                    ]
            nodes.append(
                {
                    "main_name": main_name,
                    "alt_names": [],
                    "actions": actions,
                    "classification": meta.get("classification", "термин"),
                }
            )
            node_by_norm[norm_name] = nodes[-1]

        if missing_endpoint_nodes:
            logger.warning(
                "Closure auto-fix for source_id=%s: added_nodes=%s added_relations=%s",
                source_id,
                len(missing_endpoint_nodes),
                0,
            )

        return {
            "nodes": nodes,
            "relations": relations,
            "summarization": str(payload.get("summarization", "") or ""),
        }

    @staticmethod
    def _validate_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(ExtractionPayload, "model_validate"):
            model = ExtractionPayload.model_validate(payload)
            return model.model_dump(mode="json")
        model = ExtractionPayload.parse_obj(payload)
        return model.dict()

    @staticmethod
    def _validate_node_item(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(ExtractedNode, "model_validate"):
                return ExtractedNode.model_validate(node).model_dump(mode="json")
            return ExtractedNode.parse_obj(node).dict()
        except Exception:
            return None

    @staticmethod
    def _validate_relation_item(rel: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            if hasattr(ExtractedRelation, "model_validate"):
                return ExtractedRelation.model_validate(rel).model_dump(mode="json")
            return ExtractedRelation.parse_obj(rel).dict()
        except Exception:
            return None

    def _salvage_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Best-effort salvage: keep only individually valid nodes/relations instead of
        dropping the whole batch when a few items are malformed.
        """
        salvaged_nodes: List[Dict[str, Any]] = []
        for node in payload.get("nodes", []) if isinstance(payload.get("nodes"), list) else []:
            if not isinstance(node, dict):
                continue
            node_copy = dict(node)
            node_copy["classification"] = self._classification_from_endpoint_type(
                node_copy.get("classification")
            )
            valid_node = self._validate_node_item(node_copy)
            if valid_node is not None:
                salvaged_nodes.append(valid_node)

        salvaged_relations: List[Dict[str, Any]] = []
        for rel in payload.get("relations", []) if isinstance(payload.get("relations"), list) else []:
            if not isinstance(rel, dict):
                continue
            rel_copy = dict(rel)
            rel_copy["source_node_type"] = (
                str(rel_copy.get("source_node_type", "")).strip() or "термин"
            )
            rel_copy["target_node_type"] = (
                str(rel_copy.get("target_node_type", "")).strip() or "термин"
            )
            rel_copy["type"] = str(rel_copy.get("type", "")).strip() or "СВЯЗАНО_С"
            valid_rel = self._validate_relation_item(rel_copy)
            if valid_rel is not None:
                salvaged_relations.append(valid_rel)

        return {
            "nodes": salvaged_nodes,
            "relations": salvaged_relations,
            "summarization": str(payload.get("summarization", "") or ""),
        }

    @staticmethod
    def _empty_payload() -> Dict[str, Any]:
        return {"nodes": [], "relations": [], "summarization": ""}

    def _write_invalid_debug(
        self,
        batch_idx: int,
        source_ids: Sequence[int],
        raw_result: Any,
        error_text: str,
    ) -> None:
        debug_dir = RESULTS_DIR / "_invalid_extraction_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        source_tag = "_".join(str(sid) for sid in source_ids) if source_ids else "unknown"
        debug_path = debug_dir / f"batch_{batch_idx:03d}_{source_tag}.json"

        raw_dict = self._as_dict(raw_result)
        debug_payload = {
            "batch_idx": batch_idx,
            "source_ids": list(source_ids),
            "error": error_text,
            "raw_result": raw_dict if raw_dict is not None else str(raw_result),
        }
        debug_path.write_text(
            json.dumps(debug_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def _retry_normalize_batch_result(
        self,
        batch_idx: int,
        source_ids: Sequence[int],
        chapter_blocks: Sequence[Dict[str, Any]],
        required_entities: Sequence[Dict[str, Any]],
        initial_result: Any,
    ) -> Dict[str, Any]:
        current_result = initial_result
        try:
            normalized = self._normalize_result(current_result, tuple(source_ids))
        except Exception as ex:
            logger.warning(
                "Extraction validation failed (single-pass): batch=%s source_ids=%s error=%s",
                batch_idx,
                source_ids,
                ex,
            )
            self._write_invalid_debug(
                batch_idx=batch_idx,
                source_ids=source_ids,
                raw_result=current_result,
                error_text=str(ex),
            )
            raw_payload = self._as_dict(current_result)
            if isinstance(raw_payload, dict):
                coerced = self._coerce_payload_schema(raw_payload, tuple(source_ids))
                salvaged = self._salvage_payload(coerced)
                logger.warning(
                    "Using salvaged payload after single-pass validation: batch=%s source_ids=%s nodes=%s relations=%s",
                    batch_idx,
                    source_ids,
                    len(salvaged.get("nodes", [])),
                    len(salvaged.get("relations", [])),
                )
                normalized = salvaged
            else:
                return self._empty_payload()

        missing_sources = self._find_missing_source_ids_in_payload(
            payload=normalized,
            expected_source_ids=source_ids,
        )
        if missing_sources:
            logger.warning(
                "Extraction source coverage missing (single-pass): batch=%s source_ids=%s missing=%s. Keeping partial payload.",
                batch_idx,
                source_ids,
                missing_sources,
            )

        if not PRE_NER_COVERAGE_RETRY:
            return normalized

        missing_required = self._find_missing_required_entities(
            payload=normalized,
            required_entities=required_entities,
        )
        if not missing_required:
            return normalized

        logger.warning(
            "Required entities missing: batch=%s source_ids=%s missing_count=%s. Running targeted add-missing pass.",
            batch_idx,
            source_ids,
            len(missing_required),
        )
        try:
            patch_payload = await self._invoke_missing_entities_patch(
                source_ids=source_ids,
                chapter_blocks=chapter_blocks,
                current_payload=normalized,
                missing_required=missing_required,
            )
            normalized = self._append_missing_nodes_payload(
                base_payload=normalized,
                patch_payload=patch_payload,
            )
            normalized = self._normalize_result(normalized, tuple(source_ids))
        except Exception as patch_ex:
            logger.warning(
                "Targeted add-missing pass failed for batch=%s source_ids=%s: %s",
                batch_idx,
                source_ids,
                patch_ex,
            )

        missing_after_patch = self._find_missing_required_entities(
            payload=normalized,
            required_entities=required_entities,
        )
        if missing_after_patch:
            logger.warning(
                "Required entities still missing after add-missing pass: batch=%s source_ids=%s missing_count=%s. Injecting fallback nodes.",
                batch_idx,
                source_ids,
                len(missing_after_patch),
            )
            normalized = self._inject_required_entities_as_fallback(
                payload=normalized,
                required_entities=missing_after_patch,
            )
            return self._validate_payload(normalized)

        return normalized

    @staticmethod
    def _find_missing_source_ids_in_payload(
        payload: Dict[str, Any],
        expected_source_ids: Sequence[int],
    ) -> List[int]:
        present: set[int] = set()

        nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            actions = node.get("actions") if isinstance(node.get("actions"), list) else []
            for action in actions:
                if not isinstance(action, dict):
                    continue
                sid = action.get("source_id")
                if isinstance(sid, int):
                    present.add(int(sid))

        relations = payload.get("relations") if isinstance(payload.get("relations"), list) else []
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            descriptions = rel.get("descriptions") if isinstance(rel.get("descriptions"), list) else []
            for desc in descriptions:
                if not isinstance(desc, dict):
                    continue
                sid = desc.get("source_id")
                if isinstance(sid, int):
                    present.add(int(sid))

        expected = {int(v) for v in expected_source_ids if isinstance(v, int)}
        return sorted(expected - present)

    async def _invoke_raw_llm(
        self,
        llm_input: Dict[str, Any],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Any:
        """
        Invoke LLM without parser to avoid silent parser drops in batch mode.
        JSON parsing and pydantic validation are handled in ExtractionService.
        """
        chain = self.extractor.prompt_template | self.extractor.llm
        if semaphore is not None:
            async with semaphore:
                raw = await chain.ainvoke(llm_input)
        else:
            raw = await chain.ainvoke(llm_input)
        return self.extractor._process_output(raw)

    async def _invoke_raw_llm_with(
        self,
        extractor: Any,
        llm_input: Dict[str, Any],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Any:
        chain = extractor.prompt_template | extractor.llm
        if semaphore is not None:
            async with semaphore:
                raw = await chain.ainvoke(llm_input)
        else:
            raw = await chain.ainvoke(llm_input)
        return extractor._process_output(raw)

    async def _invoke_extraction_for_batch(
        self,
        source_ids: Sequence[int],
        chapter_blocks: Sequence[Dict[str, Any]],
        required_entities: Optional[Sequence[Dict[str, Any]]] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Any]:
        source_tuple = tuple(int(sid) for sid in source_ids)
        required_entities = list(required_entities or [])

        if required_entities:
            logger.info(
                "Pre-NER helper for source_ids=%s produced required_entities=%s",
                source_tuple,
                len(required_entities),
            )
        required_mentions = [
            str(item.get("main_name", "")).strip()
            for item in required_entities
            if isinstance(item, dict) and str(item.get("main_name", "")).strip()
        ]

        node_input = self.extractor.make_user_prompt(
            text="",
            source_id=source_tuple,
            chapter_blocks=chapter_blocks,
            required_mentions=required_mentions,
        )
        nodes_raw = await self._invoke_raw_llm_with(
            self.extractor,
            node_input,
            semaphore=semaphore,
        )
        nodes_payload = self._as_dict_with(self.extractor, nodes_raw)
        if nodes_payload is None:
            raise ValueError("nodes extraction returned non-json payload")

        known_nodes = nodes_payload.get("nodes")
        if not isinstance(known_nodes, list):
            known_nodes = []

        rel_input = self.extractor_relations.make_user_prompt(
            text="",
            known_nodes=known_nodes,
            source_id=source_tuple,
            chapter_blocks=chapter_blocks,
        )
        try:
            relations_raw = await self._invoke_raw_llm_with(
                self.extractor_relations,
                rel_input,
                semaphore=semaphore,
            )
            relations_payload = (
                self._as_dict_with(self.extractor_relations, relations_raw) or {}
            )
            relations = (
                relations_payload.get("relations")
                if isinstance(relations_payload.get("relations"), list)
                else []
            )
        except Exception as ex:  # noqa: BLE001
            logger.warning(
                "Relations extraction failed for source_ids=%s; using empty relations: %s",
                source_tuple,
                ex,
            )
            relations = []

        return {
            "nodes": nodes_payload.get("nodes", []),
            "relations": relations,
            "summarization": str(nodes_payload.get("summarization", "") or ""),
        }

    async def _invoke_extraction_for_batch_chunked(
        self,
        source_ids: Sequence[int],
        chapter_blocks: Sequence[Dict[str, Any]],
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> Dict[str, Any]:
        """
        Chunk-level extraction for one chapter-batch:
        1) run names+relations on each small chunk asynchronously,
        2) append-merge all chunk payloads,
        3) return merged payload (clean merge is applied later in normalization).
        """
        chunk_groups = self._build_small_chunk_blocks(chapter_blocks)
        if not chunk_groups:
            return {"nodes": [], "relations": [], "summarization": ""}

        async def _run_chunk(chunk_blocks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
            return await self._invoke_extraction_for_batch(
                source_ids=source_ids,
                chapter_blocks=chunk_blocks,
                required_entities=[],
                semaphore=semaphore,
            )

        chunk_results = await asyncio.gather(
            *[_run_chunk(chunk_blocks) for chunk_blocks in chunk_groups],
            return_exceptions=True,
        )

        merged: Dict[str, Any] = {"nodes": [], "relations": [], "summarization": ""}
        failed_chunks = 0
        for result in chunk_results:
            if isinstance(result, Exception):
                failed_chunks += 1
                continue
            if not isinstance(result, dict):
                failed_chunks += 1
                continue
            merged = self._append_extraction_payloads(merged, result)

        if failed_chunks:
            logger.warning(
                "Small-chunk extraction had failed chunks: source_ids=%s failed=%s total=%s",
                tuple(source_ids),
                failed_chunks,
                len(chunk_groups),
            )
        logger.info(
            "Small-chunk extraction merged: source_ids=%s chunks=%s nodes=%s relations=%s",
            tuple(source_ids),
            len(chunk_groups),
            len(merged.get("nodes", [])),
            len(merged.get("relations", [])),
        )
        return merged

    async def _invoke_missing_entities_patch(
        self,
        source_ids: Sequence[int],
        chapter_blocks: Sequence[Dict[str, Any]],
        current_payload: Dict[str, Any],
        missing_required: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        source_tuple = tuple(int(sid) for sid in source_ids)
        grouped_mentions = self._group_missing_mentions_for_retry(missing_required)
        total_mentions = sum(len(v) for v in grouped_mentions.values())
        if total_mentions == 0:
            return {"nodes": [], "relations": [], "summarization": ""}

        existing_nodes = (
            current_payload.get("nodes", [])
            if isinstance(current_payload.get("nodes"), list)
            else []
        )
        patch_extractor = self.extractor_missing_nodes or self.extractor

        # Split by class and then by chunk; run all patch chunks asynchronously.
        chunk_size = 50
        class_order = ["персонаж", "место", "организация", "термин"]
        mention_chunks: List[tuple[str, List[str]]] = []
        for cls in class_order:
            mentions = grouped_mentions.get(cls, [])
            for i in range(0, len(mentions), chunk_size):
                mention_chunks.append((cls, mentions[i : i + chunk_size]))

        async def _run_patch_chunk(
            chunk_class: str,
            chunk_mentions: Sequence[str],
        ) -> List[Dict[str, Any]]:
            if self.extractor_missing_nodes is not None:
                node_input = self.extractor_missing_nodes.make_user_prompt(
                    text="",
                    source_id=source_tuple,
                    chapter_blocks=chapter_blocks,
                    missing_mentions=chunk_mentions,
                    existing_nodes=existing_nodes,
                )
            else:
                node_input = self.extractor.make_user_prompt(
                    text="",
                    source_id=source_tuple,
                    chapter_blocks=chapter_blocks,
                    required_mentions=chunk_mentions,
                    existing_nodes=existing_nodes,
                    retry_mode="add_missing_only",
                )

            nodes_raw = await self._invoke_raw_llm_with(patch_extractor, node_input)
            nodes_payload = self._as_dict_with(patch_extractor, nodes_raw)
            if nodes_payload is None:
                raise ValueError("missing-entities retry returned non-json payload")

            # Hard filter: keep only nodes from this chunk.
            allowed_keys = {
                self._entity_match_key(x)
                for x in chunk_mentions
                if self._entity_match_key(x)
            }
            filtered_nodes: List[Dict[str, Any]] = []
            for node in (
                nodes_payload.get("nodes", [])
                if isinstance(nodes_payload.get("nodes"), list)
                else []
            ):
                if not isinstance(node, dict):
                    continue
                node_keys = {self._entity_match_key(node.get("main_name"))}
                for alias in (
                    node.get("alt_names")
                    if isinstance(node.get("alt_names"), list)
                    else []
                ):
                    node_keys.add(self._entity_match_key(alias))
                node_keys.discard("")
                if node_keys.intersection(allowed_keys):
                    filtered_nodes.append(node)
            return filtered_nodes

        chunk_results = await asyncio.gather(
            *[_run_patch_chunk(cls, chunk) for cls, chunk in mention_chunks],
            return_exceptions=True,
        )

        merged_by_main: Dict[str, Dict[str, Any]] = {}
        for idx, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Missing-node chunk retry failed: source_ids=%s chunk_idx=%s error=%s",
                    source_tuple,
                    idx,
                    result,
                )
                continue
            for node in result:
                main_name = str(node.get("main_name", "")).strip()
                if not main_name:
                    continue
                key = self._norm_name(main_name)
                if not key:
                    continue
                if key not in merged_by_main:
                    merged_by_main[key] = dict(node)
                    continue
                base = merged_by_main[key]
                merged_alt = list(base.get("alt_names") or [])
                for alt in node.get("alt_names") if isinstance(node.get("alt_names"), list) else []:
                    if isinstance(alt, str) and alt.strip() and alt not in merged_alt:
                        merged_alt.append(alt)
                base["alt_names"] = merged_alt
                merged_actions = list(base.get("actions") or [])
                merged_actions.extend(node.get("actions") if isinstance(node.get("actions"), list) else [])
                base["actions"] = self._dedupe_actions(merged_actions)
                if (not str(base.get("classification") or "").strip()) and str(node.get("classification") or "").strip():
                    base["classification"] = node.get("classification")
                merged_by_main[key] = base

        return {
            "nodes": list(merged_by_main.values()),
            "relations": [],
            "summarization": str(current_payload.get("summarization", "") or ""),
        }

    async def _invoke_raw_batch(
        self,
        groups_meta: List[Dict[str, Any]],
        concurrency: int,
    ) -> List[Any]:
        semaphore = asyncio.Semaphore(max(1, int(concurrency or 1)))

        async def _one(group: Dict[str, Any]) -> Any:
            try:
                if EXTRACTION_USE_SMALL_CHUNKS:
                    return await self._invoke_extraction_for_batch_chunked(
                        source_ids=group["source_ids"],
                        chapter_blocks=group["chapter_blocks"],
                        semaphore=semaphore,
                    )
                return await self._invoke_extraction_for_batch(
                    source_ids=group["source_ids"],
                    chapter_blocks=group["chapter_blocks"],
                    required_entities=group.get("required_entities") or [],
                    semaphore=semaphore,
                )
            except Exception as ex:  # noqa: BLE001
                return ex

        tasks = [_one(group) for group in groups_meta]
        return await asyncio.gather(*tasks)

    def _normalize_result(self, result: Any, source_id: Any) -> Dict[str, Any]:
        payload = self._as_dict(result)
        if payload is None:
            raise ValueError(
                f"Extraction payload for source_id={source_id} is invalid ({type(result)})"
            )

        coerced = self._coerce_payload_schema(payload, source_id)
        coerced = self._merge_identity_alias_nodes(coerced, source_id)
        coerced = self._enforce_nodes_relations_closure(coerced, source_id)
        return self._validate_payload(coerced)

    @staticmethod
    def _extract_chapter_blocks(doc: Document) -> Optional[List[Dict[str, Any]]]:
        metadata = doc.metadata or {}

        blocks = metadata.get("chapter_blocks") or metadata.get("batch_chapters")
        if isinstance(blocks, list):
            normalized = []
            for item in blocks:
                if not isinstance(item, dict):
                    continue
                sid = item.get("source_id")
                text = item.get("text")
                if isinstance(sid, int) and isinstance(text, str) and text.strip():
                    normalized.append(
                        {
                            "source_id": sid,
                            "chapter_id": item.get("chapter_id", sid),
                            "text": text,
                        }
                    )
            return normalized or None

        chapter_texts = metadata.get("chapter_texts")
        source_id = metadata.get("source_id")
        if isinstance(chapter_texts, list) and isinstance(source_id, (tuple, list)):
            source_ids = [sid for sid in source_id if isinstance(sid, int)]
            normalized = []
            for sid, text in zip(source_ids, chapter_texts):
                if isinstance(text, str) and text.strip():
                    normalized.append(
                        {"source_id": sid, "chapter_id": sid, "text": text}
                    )
            return normalized or None

        return None

    def _count_name_hits(self, payload: Dict[str, Any], token: str) -> int:
        target = self._norm_name(token)
        if not target:
            return 0
        nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
        count = 0
        for node in nodes:
            if not isinstance(node, dict):
                continue
            names = [str(node.get("main_name", ""))]
            names.extend(
                str(x) for x in (node.get("alt_names") if isinstance(node.get("alt_names"), list) else [])
            )
            names_blob = " | ".join(names)
            if target in self._norm_name(names_blob):
                count += 1
        return count

    async def extract_from_texts_async(
        self,
        texts: List[Document],
        concurrency: int = 5,
    ) -> List[str]:
        groups_meta: List[Dict[str, Any]] = []
        all_summarizations: List[str] = ["" for _ in texts]

        valid_entries: List[Dict[str, Any]] = []
        for idx, text in enumerate(texts):
            source_id = text.metadata.get("source_id")
            if not isinstance(source_id, int):
                logger.warning("Text at idx=%s has invalid source_id and is skipped", idx)
                continue
            valid_entries.append(
                {
                    "index": idx,
                    "source_id": source_id,
                    "chapter_id": int(text.metadata.get("chapter_id", source_id)),
                    "text": text.page_content,
                }
            )

        if not valid_entries:
            logger.warning("No valid texts for extraction")
            return all_summarizations

        chapter_batch_size = max(1, int(EXTRACTION_CHAPTER_BATCH_SIZE or BATCH_SIZE or 5))

        for start in range(0, len(valid_entries), chapter_batch_size):
            batch_entries = valid_entries[start : start + chapter_batch_size]
            source_tuple = tuple(entry["source_id"] for entry in batch_entries)
            chapter_blocks = [
                {
                    "source_id": entry["source_id"],
                    "chapter_id": entry["chapter_id"],
                    "text": entry["text"],
                }
                for entry in batch_entries
            ]
            required_entities = self._build_required_entities(chapter_blocks)
            groups_meta.append(
                {
                    "source_ids": source_tuple,
                    "indices": [entry["index"] for entry in batch_entries],
                    "chapter_blocks": chapter_blocks,
                    "required_entities": required_entities,
                }
            )

        logger.info(
            "Starting extraction for %s chapter-batches (batch_size=%s) with concurrency=%s mode=%s",
            len(groups_meta),
            chapter_batch_size,
            concurrency,
            "small_chunks" if EXTRACTION_USE_SMALL_CHUNKS else "regular",
        )
        results = await self._invoke_raw_batch(groups_meta, concurrency=concurrency)

        batch_append_dir = RESULTS_DIR / "_batch_append_payloads"
        batch_clean_dir = RESULTS_DIR / "_batch_payloads"
        batch_report_dir = RESULTS_DIR / "_batch_reports"
        batch_append_dir.mkdir(parents=True, exist_ok=True)
        batch_clean_dir.mkdir(parents=True, exist_ok=True)
        batch_report_dir.mkdir(parents=True, exist_ok=True)

        for group_idx, (raw_result, meta) in enumerate(zip(results, groups_meta)):
            source_tuple = meta["source_ids"]
            batch_tag = f"batch_{group_idx:03d}_{source_tuple[0]}_{source_tuple[-1]}"
            try:
                logger.info(
                    "Extraction batch start: idx=%s source_ids=%s",
                    group_idx,
                    source_tuple,
                )
                raw_payload = self._as_dict(raw_result)
                append_payload: Dict[str, Any]
                if isinstance(raw_payload, dict):
                    append_payload = self._coerce_payload_schema(raw_payload, tuple(source_tuple))
                else:
                    append_payload = self._empty_payload()

                append_dump_path = batch_append_dir / f"{batch_tag}.append.json"
                append_dump_path.write_text(
                    json.dumps(append_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                normalized = await self._retry_normalize_batch_result(
                    batch_idx=group_idx,
                    source_ids=source_tuple,
                    chapter_blocks=meta["chapter_blocks"],
                    required_entities=meta.get("required_entities") or [],
                    initial_result=raw_result,
                )
                summary_text = normalized.get("summarization", "")

                batch_dump_path = batch_clean_dir / f"{batch_tag}.json"
                batch_dump_path.write_text(
                    json.dumps(normalized, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                required_entities = meta.get("required_entities") or []
                missing_before = self._find_missing_required_entities(
                    payload=append_payload,
                    required_entities=required_entities,
                )
                missing_after = self._find_missing_required_entities(
                    payload=normalized,
                    required_entities=required_entities,
                )
                batch_report = {
                    "batch_idx": group_idx,
                    "source_ids": list(source_tuple),
                    "mode": "small_chunks" if EXTRACTION_USE_SMALL_CHUNKS else "regular",
                    "nodes_before_append": len(append_payload.get("nodes", [])),
                    "relations_before_append": len(append_payload.get("relations", [])),
                    "nodes_after_clean": len(normalized.get("nodes", [])),
                    "relations_after_clean": len(normalized.get("relations", [])),
                    "required_entities_count": len(required_entities),
                    "missing_required_before_clean": len(missing_before),
                    "missing_required_after_clean": len(missing_after),
                    "key_hits": {
                        "aenarion": self._count_name_hits(normalized, "аэнарион"),
                        "aenarion_guard": self._count_name_hits(normalized, "аэнарион защит"),
                        "ciandros": self._count_name_hits(normalized, "циандрос"),
                        "silanna": self._count_name_hits(normalized, "силанна"),
                    },
                    "append_payload_path": str(append_dump_path),
                    "clean_payload_path": str(batch_dump_path),
                }
                batch_report_path = batch_report_dir / f"{batch_tag}.report.json"
                batch_report_path.write_text(
                    json.dumps(batch_report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                for sid in source_tuple:
                    if sid % 5 == 0:
                        logger.info(
                            "Extraction progress marker: chapter source_id=%s",
                            sid,
                        )

                for text_idx in meta["indices"]:
                    if 0 <= text_idx < len(all_summarizations):
                        all_summarizations[text_idx] = summary_text
                logger.info(
                    "Extraction batch finish: idx=%s source_ids=%s nodes=%s relations=%s",
                    group_idx,
                    source_tuple,
                    len(normalized.get("nodes", [])),
                    len(normalized.get("relations", [])),
                )
            except Exception as ex:
                logger.error(
                    "Error while handling extraction batch idx=%s source_ids=%s: %s",
                    group_idx,
                    source_tuple,
                    ex,
                    exc_info=True,
                )

        logger.info(
            "Extraction completed: %s/%s summaries",
            len([s for s in all_summarizations if s]),
            len(texts),
        )
        return all_summarizations

    def extract_from_texts(
        self,
        texts: List[Document],
        concurrency: Optional[int] = None,
    ) -> List[str]:
        concurrency = concurrency or PARALLEL_CONCURRENCY
        return asyncio.run(self.extract_from_texts_async(texts, concurrency))
