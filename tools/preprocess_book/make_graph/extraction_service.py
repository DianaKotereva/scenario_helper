import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from pydantic import BaseModel

from tools.preprocess_book.config.preprocess_settings import (
    BATCH_SIZE,
    EXTRACTION_CHAPTER_BATCH_SIZE,
    EXTRACTION_VALIDATION_RETRY_COUNT,
    PARALLEL_CONCURRENCY,
    RESULTS_DIR,
)
from tools.preprocess_book.prompts.extract_names import (
    ExtractedNode,
    ExtractedRelation,
    ExtractionPayload,
)

logger = logging.getLogger(__name__)


class ExtractionService:
    """Service for extracting entities and relations from chapter texts."""

    def __init__(self, extractor, extractor_relations=None):
        if extractor_relations is None:
            raise ValueError("extractor_relations is required for two-stage extraction")
        self.extractor = extractor
        self.extractor_relations = extractor_relations

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
                        "quotes": [text[:200]],
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
                            "quotes": [text[:200]],
                        }
                    )
                continue

            if not isinstance(item, dict):
                continue

            sid = item.get("source_id", fallback_source_id)
            cid = item.get("chapter_id", sid)
            desc = (item.get("description") or "").strip()
            quotes_raw = item.get("quotes") or []
            quotes = [str(q).strip() for q in quotes_raw if str(q).strip()]

            if not isinstance(sid, int):
                sid = fallback_source_id
            if not isinstance(cid, int):
                cid = sid
            if not desc:
                continue
            if not quotes:
                quotes = [desc[:200]]

            out.append(
                {
                    "source_id": sid,
                    "chapter_id": cid,
                    "description": desc,
                    "quotes": quotes,
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
        Merge obvious duplicates within one extraction payload.
        Conservative rule:
        - merge only when one node main_name appears in another node alt_names
          (or explicit parenthetical aliases in main_name).
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

        norm_main: List[str] = []
        norm_aliases: List[set[str]] = []
        classifications: List[str] = []

        for node in nodes:
            main = str(node.get("main_name", "")).strip()
            aliases_raw = node.get("alt_names") if isinstance(node.get("alt_names"), list) else []
            aliases = [str(a).strip() for a in aliases_raw if str(a).strip()]
            aliases.extend(self._extract_parenthetical_aliases(main))
            norm_main.append(self._norm_name(main))
            norm_aliases.append({self._norm_name(a) for a in aliases if self._norm_name(a)})
            classifications.append(str(node.get("classification", "")).strip())

        for i in range(n):
            for j in range(i + 1, n):
                if not norm_main[i] or not norm_main[j]:
                    continue
                if classifications[i] and classifications[j] and classifications[i] != classifications[j]:
                    continue

                merge_by_alias = (
                    norm_main[i] in norm_aliases[j] or norm_main[j] in norm_aliases[i]
                )
                if merge_by_alias:
                    union(i, j)

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
            "Identity alias merge for source_id=%s: nodes %s -> %s",
            source_id,
            len(nodes),
            len(merged_nodes),
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
                            "quotes": [
                                str(q).strip()
                                for q in (first.get("quotes") or [])
                                if str(q).strip()
                            ]
                            or ["Упомянут в отношениях"],
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
        initial_result: Any,
    ) -> Dict[str, Any]:
        retry_limit = max(0, int(EXTRACTION_VALIDATION_RETRY_COUNT or 0))
        current_result = initial_result
        last_normalized_payload: Optional[Dict[str, Any]] = None

        for attempt in range(retry_limit + 1):
            try:
                normalized = self._normalize_result(current_result, tuple(source_ids))
                last_normalized_payload = normalized
                missing_sources = self._find_missing_source_ids_in_payload(
                    payload=normalized,
                    expected_source_ids=source_ids,
                )
                if missing_sources:
                    if attempt >= retry_limit:
                        logger.warning(
                            "Extraction source coverage still missing after retries: batch=%s source_ids=%s missing=%s. Keeping partial payload.",
                            batch_idx,
                            source_ids,
                            missing_sources,
                        )
                        return normalized
                    raise ValueError(
                        f"Extraction source coverage missing for source_ids={missing_sources}"
                    )
                return normalized
            except Exception as ex:
                if attempt >= retry_limit:
                    if last_normalized_payload is not None:
                        logger.warning(
                            "Extraction validation failed after retries, keeping last normalized payload: batch=%s source_ids=%s error=%s",
                            batch_idx,
                            source_ids,
                            ex,
                        )
                        return last_normalized_payload
                    logger.warning(
                        "Extraction validation failed after retries: batch=%s source_ids=%s error=%s",
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
                            "Using salvaged payload after validation retries: batch=%s source_ids=%s nodes=%s relations=%s",
                            batch_idx,
                            source_ids,
                            len(salvaged.get("nodes", [])),
                            len(salvaged.get("relations", [])),
                        )
                        return salvaged
                    return self._empty_payload()

                logger.warning(
                    "Extraction validation failed; retry %s/%s for batch=%s source_ids=%s: %s",
                    attempt + 1,
                    retry_limit,
                    batch_idx,
                    source_ids,
                    ex,
                )
                try:
                    current_result = await self._invoke_extraction_for_batch(
                        source_ids=source_ids,
                        chapter_blocks=chapter_blocks,
                    )
                except Exception as retry_ex:
                    logger.warning(
                        "LLM retry call failed for batch=%s source_ids=%s: %s",
                        batch_idx,
                        source_ids,
                        retry_ex,
                    )
                    current_result = {"error": str(retry_ex), "nodes": [], "relations": [], "summarization": ""}

        return self._empty_payload()

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

    async def _invoke_raw_llm(self, llm_input: Dict[str, Any]) -> Any:
        """
        Invoke LLM without parser to avoid silent parser drops in batch mode.
        JSON parsing and pydantic validation are handled in ExtractionService.
        """
        chain = self.extractor.prompt_template | self.extractor.llm
        raw = await chain.ainvoke(llm_input)
        return self.extractor._process_output(raw)

    async def _invoke_raw_llm_with(self, extractor: Any, llm_input: Dict[str, Any]) -> Any:
        chain = extractor.prompt_template | extractor.llm
        raw = await chain.ainvoke(llm_input)
        return extractor._process_output(raw)

    async def _invoke_extraction_for_batch(
        self,
        source_ids: Sequence[int],
        chapter_blocks: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        source_tuple = tuple(int(sid) for sid in source_ids)

        node_input = self.extractor.make_user_prompt(
            text="",
            source_id=source_tuple,
            chapter_blocks=chapter_blocks,
        )
        nodes_raw = await self._invoke_raw_llm_with(self.extractor, node_input)
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

    async def _invoke_raw_batch(
        self,
        groups_meta: List[Dict[str, Any]],
        concurrency: int,
    ) -> List[Any]:
        semaphore = asyncio.Semaphore(max(1, int(concurrency or 1)))

        async def _one(group: Dict[str, Any]) -> Any:
            async with semaphore:
                try:
                    return await self._invoke_extraction_for_batch(
                        source_ids=group["source_ids"],
                        chapter_blocks=group["chapter_blocks"],
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
            groups_meta.append(
                {
                    "source_ids": source_tuple,
                    "indices": [entry["index"] for entry in batch_entries],
                    "chapter_blocks": chapter_blocks,
                }
            )

        logger.info(
            "Starting extraction for %s chapter-batches (batch_size=%s) with concurrency=%s",
            len(groups_meta),
            chapter_batch_size,
            concurrency,
        )
        results = await self._invoke_raw_batch(groups_meta, concurrency=concurrency)

        for group_idx, (raw_result, meta) in enumerate(zip(results, groups_meta)):
            source_tuple = meta["source_ids"]
            try:
                logger.info(
                    "Extraction batch start: idx=%s source_ids=%s",
                    group_idx,
                    source_tuple,
                )
                normalized = await self._retry_normalize_batch_result(
                    batch_idx=group_idx,
                    source_ids=source_tuple,
                    chapter_blocks=meta["chapter_blocks"],
                    initial_result=raw_result,
                )
                summary_text = normalized.get("summarization", "")

                batch_dump_dir = RESULTS_DIR / "_batch_payloads"
                batch_dump_dir.mkdir(parents=True, exist_ok=True)
                batch_dump_path = batch_dump_dir / (
                    f"batch_{group_idx:03d}_{source_tuple[0]}_{source_tuple[-1]}.json"
                )
                batch_dump_path.write_text(
                    json.dumps(normalized, ensure_ascii=False, indent=2),
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
