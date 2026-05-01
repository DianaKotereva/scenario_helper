import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.output_parsers import JsonOutputParser

from src.llm_core.llm_prompt_base import LLMBase
from src.utils.graph_search import BookNode

RUSSIAN_STOPWORDS = {
    "и", "или", "но", "а", "да", "ли", "же", "то", "ни", "не",
    "в", "во", "на", "с", "со", "к", "ко", "по", "из", "за", "от", "до", "при", "для", "у", "о", "об",
    "над", "под", "перед", "без", "через", "между", "про", "ради", "после",
    "что", "чтобы", "как", "если", "когда", "пока", "потому", "поэтому", "также", "тоже", "либо",
    "этот", "эта", "это", "эти", "того", "тому", "тем", "тех", "там", "тут", "здесь",
    "его", "ее", "её", "их", "ему", "ей", "ими", "них",
    "он", "она", "оно", "они", "мы", "вы", "я", "ты",
    "который", "которая", "которые", "которого", "которому", "которых",
    "был", "была", "были", "быть", "есть", "нет", "это", "тот", "та", "те",
}


system_prompt = """
Ты — агент верификации сущностей в литературном графе знаний.
Нужно определить, относятся ли два описания к одной и той же сущности.

Важно:
1) Совпадение имени само по себе недостаточно.
2) Для тёзок разных поколений (например, «отец/сын/внук») по умолчанию ставь `is_same_entity=false`.
3) Разрешай merge тёзок только при явном текстовом доказательстве одной идентичности:
- прямое указание на переименование/алиас,
- явное подтверждение, что это тот же персонаж в иной роли/форме.
4) Если есть конфликт родственных якорей, ставь `is_same_entity=false` и заполняй `hard_conflict_flags`.

Критерии merge (должны быть совместимы):
- классификация сущности;
- уникальные идентификаторы (main_name/устойчивые alias);
- совместимость действий и биографии;
- отсутствие противоречий по родству/поколению/роли.

Автоматический reject:
- разные classification;
- конфликт уникальных идентификаторов без явного моста;
- конфликт поколений/родства;
- взаимно исключающие биографические факты.

Требуемый JSON-ответ (строго, без дополнительного текста):
{
  "is_same_entity": bool,
  "confidence": "высокий|средний|низкий",
  "key_evidence": [str, ...],
  "conflicting_attributes": [str, ...],
  "hard_conflict_flags": [str, ...],
  "kinship_anchors": {
    "child_of": [str, ...],
    "parent_of": [str, ...],
    "grandchild_of": [str, ...]
  },
  "merge_blocked_by": str | null
}

Правила для `kinship_anchors`:
- заполняй только по данным из входа (без выдумывания),
- если данных нет, возвращай пустые списки.
"""


class Verification(LLMBase):
    """Класс для верификации сущностей."""

    def __init__(
        self,
        llm,
        parser: JsonOutputParser = None,
        snippets_top_k: int = 5,
    ):
        if parser is None:
            parser = JsonOutputParser()
        super().__init__(
            llm=llm,
            system_prompt=system_prompt,
            parser=parser,
            parse_json=True,
        )
        self.snippets_top_k = max(1, int(snippets_top_k))
        self._chunk_index_by_source_id: Dict[int, List[Dict[str, str]]] | None = None

    @staticmethod
    def _default_chunks_path() -> Path:
        return (
            Path(__file__).resolve().parents[3]
            / "processed_data"
            / "full_book_batch5_async_deepseek_20260413_021200"
            / "snapshot"
            / "chunks.jsonl"
        )

    def _load_chunk_index(self) -> Dict[int, List[Dict[str, str]]]:
        if self._chunk_index_by_source_id is not None:
            return self._chunk_index_by_source_id

        chunks_path = Path(
            os.getenv(
                "VERIFICATION_CHUNKS_JSONL_PATH",
                str(self._default_chunks_path()),
            )
        )
        index: Dict[int, List[Dict[str, str]]] = {}
        if chunks_path.exists():
            with chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    sid = row.get("source_id")
                    if not isinstance(sid, int):
                        continue
                    chunk_id = str(row.get("chunk_id", ""))
                    text = str(row.get("text", ""))
                    if not text:
                        continue
                    text_norm = self._normalize_search_text(text)
                    stems = self._extract_terms(text_norm, max_terms=0)
                    index.setdefault(sid, []).append(
                        {
                            "chunk_id": chunk_id,
                            "text": text,
                            "text_norm": text_norm,
                            "stems": stems,
                        }
                    )
        self._chunk_index_by_source_id = index
        return index

    @staticmethod
    def _normalize_search_text(value: str) -> str:
        return (value or "").lower().replace("ё", "е")

    @staticmethod
    def _big_chunk_sort_key(chunk_id: str) -> tuple[int, int, str]:
        value = str(chunk_id or "").strip()
        m = re.match(r"^ch_?(\d+)_(\d+)$", value)
        if m:
            return (int(m.group(1)), int(m.group(2)), value)
        return (10**9, 10**9, value)

    @staticmethod
    def _stem_token(token: str) -> str:
        t = Verification._normalize_search_text(token)
        t = re.sub(r"[^a-zа-я-]", "", t)
        if len(t) < 3:
            return t
        # Lightweight stemmer: trim common suffixes, keep stable core.
        suffixes = (
            "иями", "ями", "ами", "ией", "ией", "ией", "ого", "ему", "ому", "ыми",
            "ими", "иях", "иях", "ах", "ях", "ия", "ья", "ью", "иям", "иях", "ов",
            "ев", "ей", "ам", "ям", "ом", "ем", "ой", "ей", "ый", "ий", "ая", "яя",
            "ое", "ее", "ых", "их", "ую", "юю", "а", "я", "ы", "и", "о", "е", "у",
            "ю", "ь",
        )
        for suf in suffixes:
            if len(t) - len(suf) >= 3 and t.endswith(suf):
                return t[: -len(suf)]
        return t

    @staticmethod
    def _extract_terms(text: str, max_terms: int = 16) -> List[str]:
        words = re.findall(r"[А-Яа-яA-Za-zЁё][А-Яа-яA-Za-zЁё-]{2,}", text or "")
        terms: List[str] = []
        for w in words:
            wl = Verification._stem_token(w)
            if wl in RUSSIAN_STOPWORDS:
                continue
            if wl not in terms:
                terms.append(wl)
            if max_terms > 0 and len(terms) >= max_terms:
                break
        return terms

    @staticmethod
    def _extract_name_terms(values: List[str]) -> List[str]:
        out: List[str] = []
        for value in values:
            words = re.findall(r"[А-Яа-яA-Za-zЁё][А-Яа-яA-Za-zЁё-]{1,}", value or "")
            for w in words:
                wl = Verification._stem_token(w)
                if wl not in out:
                    out.append(wl)
        return out

    @staticmethod
    def _has_word(text_norm: str, term: str) -> bool:
        if not term:
            return False
        stem = Verification._stem_token(term)
        if not stem:
            return False
        tokens = Verification._extract_terms(text_norm, max_terms=0)
        return stem in tokens

    @staticmethod
    def _centered_snippet(text: str, term: str, radius: int = 280) -> str:
        if not text:
            return ""
        idx = Verification._normalize_search_text(text).find(
            Verification._normalize_search_text(term or "")
        )
        if idx < 0:
            return text[: 2 * radius].strip()
        left = max(0, idx - radius)
        right = min(len(text), idx + radius)
        return text[left:right].strip()

    def _resolve_action_snippet(
        self,
        source_id: int,
        action_text: str,
        main_name_terms: List[str],
        alt_name_terms: List[str],
    ) -> Dict[str, Any]:
        index = self._load_chunk_index()
        candidates = index.get(source_id, [])
        if not candidates:
            return {"big_chunk_ids": [], "text_snippets": []}

        main_terms = [self._stem_token(t) for t in (main_name_terms or []) if t and len(t) >= 3]
        main_terms = [t for t in main_terms if t]
        alt_terms = [self._stem_token(t) for t in (alt_name_terms or []) if t and len(t) >= 3]
        alt_terms = [t for t in alt_terms if t]
        alt_terms = [t for t in alt_terms if t not in set(main_terms)]
        action_terms = [t for t in self._extract_terms(action_text, max_terms=0) if t and len(t) >= 3]
        all_terms = list(dict.fromkeys(main_terms + alt_terms + action_terms))
        # Priority:
        # 1) main_name + action words
        # 2) main_name
        # 3) alt_names + action words
        # 4) alt_names
        g1_main_plus_action: List[tuple[int, Dict[str, str], List[str], List[str], List[str], List[str]]] = []
        g2_main_only: List[tuple[int, Dict[str, str], List[str], List[str], List[str], List[str]]] = []
        g3_alt_plus_action: List[tuple[int, Dict[str, str], List[str], List[str], List[str], List[str]]] = []
        g4_alt_only: List[tuple[int, Dict[str, str], List[str], List[str], List[str], List[str]]] = []

        for chunk in candidates:
            chunk_stems = set(chunk.get("stems", []))
            main_hits = [t for t in main_terms if t in chunk_stems]
            alt_hits = [t for t in alt_terms if t in chunk_stems]
            action_hits = [t for t in action_terms if t in chunk_stems]
            total_hits = [t for t in all_terms if t in chunk_stems]
            score = len(total_hits)
            pack = (score, chunk, main_hits, alt_hits, action_hits, total_hits)
            if main_hits and action_hits:
                g1_main_plus_action.append(pack)
            elif main_hits:
                g2_main_only.append(pack)
            elif alt_hits and action_hits:
                g3_alt_plus_action.append(pack)
            elif alt_hits:
                g4_alt_only.append(pack)

        def _sort_key(
            item: tuple[int, Dict[str, str], List[str], List[str], List[str], List[str]]
        ) -> tuple[int, int, int]:
            score, _, main_hits, alt_hits, action_hits, _ = item
            return (score, len(main_hits) + len(alt_hits), len(action_hits))

        g1_main_plus_action.sort(key=_sort_key, reverse=True)
        g2_main_only.sort(key=_sort_key, reverse=True)
        g3_alt_plus_action.sort(key=_sort_key, reverse=True)
        g4_alt_only.sort(key=_sort_key, reverse=True)

        ordered_chunks: List[Dict[str, str]] = []
        seen_ids = set()
        for group in (g1_main_plus_action, g2_main_only, g3_alt_plus_action, g4_alt_only):
            for _, chunk, _, _, _, _ in group:
                cid = chunk.get("chunk_id")
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                ordered_chunks.append(chunk)

        if not ordered_chunks:
            return {"big_chunk_ids": [], "text_snippets": []}

        # Use best anchor from the highest-priority chunk:
        # main_name > alt_names > action words.
        top_chunk_stems = set(ordered_chunks[0].get("stems", []))
        top_main_hits = [t for t in main_terms if t in top_chunk_stems]
        top_alt_hits = [t for t in alt_terms if t in top_chunk_stems]
        top_action_hits = [t for t in action_terms if t in top_chunk_stems]
        best_term = ""
        if top_main_hits:
            best_term = sorted(top_main_hits, key=len, reverse=True)[0]
        elif top_alt_hits:
            best_term = sorted(top_alt_hits, key=len, reverse=True)[0]
        elif top_action_hits:
            best_term = sorted(top_action_hits, key=len, reverse=True)[0]

        # Keep prompt size stable: top-k snippets per action.
        top_chunks = ordered_chunks[: self.snippets_top_k]
        text_snippets = []
        for chunk in top_chunks:
            text_snippets.append(
                {
                    "big_chunk_id": chunk.get("chunk_id"),
                    "text_snippet": self._centered_snippet(chunk["text"], best_term, radius=260),
                }
            )
        return {
            "big_chunk_ids": [c.get("chunk_id") for c in top_chunks],
            "text_snippets": text_snippets,
        }

    def make_user_prompt(
        self,
        node: BookNode,
        new_node: Dict[str, Any],
        source_id: Tuple[int, ...],
        last_n: int = -50,
        existing_kinship_aliases: List[str] | None = None,
        new_kinship_aliases: List[str] | None = None,
    ) -> Dict[str, Any]:
        def _normalize_actions_with_node_snippets(
            actions: Any,
            main_name_terms: List[str],
            alt_name_terms: List[str],
        ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            """
            Keep full biography context as structured list and collect node-level snippets.
            """
            out: List[Dict[str, Any]] = []
            snippets_map: Dict[str, Dict[str, Any]] = {}
            snippets_order: List[str] = []
            fallback_sid = int(source_id[0]) if source_id else 0

            def _collect_snippets(snippet_pack: Dict[str, Any]) -> None:
                for item in snippet_pack.get("text_snippets", []) or []:
                    if not isinstance(item, dict):
                        continue
                    chunk_id = str(item.get("big_chunk_id", "") or "").strip()
                    text = str(item.get("text_snippet", "") or "").strip()
                    if not chunk_id or not text:
                        continue
                    # Dedup by chunk id at node level; keep most-recent hit order.
                    snippets_map[chunk_id] = {
                        "big_chunk_id": chunk_id,
                        "text_snippet": text,
                    }
                    if chunk_id in snippets_order:
                        snippets_order.remove(chunk_id)
                    snippets_order.append(chunk_id)

            def _finalize_snippets(actions_count: int) -> List[Dict[str, Any]]:
                # Keep node-level snippets bounded by configurable top-k.
                snippets = [snippets_map[cid] for cid in snippets_order if cid in snippets_map]
                snippets = snippets[-self.snippets_top_k :]
                return sorted(
                    snippets,
                    key=lambda x: self._big_chunk_sort_key(str(x.get("big_chunk_id", ""))),
                )

            if isinstance(actions, str):
                text = actions.strip()
                if text:
                    snippet_pack = self._resolve_action_snippet(
                        source_id=fallback_sid,
                        action_text=text,
                        main_name_terms=main_name_terms,
                        alt_name_terms=alt_name_terms,
                    )
                    _collect_snippets(snippet_pack)
                    out.append(
                        {
                            "source_id": fallback_sid,
                            "chapter_id": fallback_sid,
                            "description": text,
                        }
                    )
                snippets = _finalize_snippets(actions_count=len(out))
                return out, snippets

            if not isinstance(actions, list):
                return out, []

            for item in actions:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        snippet_pack = self._resolve_action_snippet(
                            source_id=fallback_sid,
                            action_text=text,
                            main_name_terms=main_name_terms,
                            alt_name_terms=alt_name_terms,
                        )
                        _collect_snippets(snippet_pack)
                        out.append(
                            {
                                "source_id": fallback_sid,
                                "chapter_id": fallback_sid,
                                "description": text,
                            }
                        )
                    continue

                if isinstance(item, dict):
                    desc = str(item.get("description", "")).strip()
                    if not desc:
                        continue
                    sid = item.get("source_id", fallback_sid)
                    cid = item.get("chapter_id", sid)
                    resolved_sid = sid if isinstance(sid, int) else fallback_sid
                    snippet_pack = self._resolve_action_snippet(
                        source_id=resolved_sid,
                        action_text=desc,
                        main_name_terms=main_name_terms,
                        alt_name_terms=alt_name_terms,
                    )
                    _collect_snippets(snippet_pack)
                    out.append(
                        {
                            "source_id": resolved_sid,
                            "chapter_id": cid if isinstance(cid, int) else fallback_sid,
                            "description": desc,
                        }
                    )
                    continue

                # Handle Action dataclass objects from BookNode.actions.
                desc = str(getattr(item, "action", "")).strip()
                if not desc:
                    continue
                sid_raw = getattr(item, "source_id", fallback_sid)
                cid_raw = getattr(item, "chapter_id", sid_raw)
                if isinstance(sid_raw, (tuple, list)):
                    sid = sid_raw[0] if sid_raw else fallback_sid
                else:
                    sid = sid_raw
                cid = cid_raw if isinstance(cid_raw, int) else sid
                resolved_sid = sid if isinstance(sid, int) else fallback_sid
                snippet_pack = self._resolve_action_snippet(
                    source_id=resolved_sid,
                    action_text=desc,
                    main_name_terms=main_name_terms,
                    alt_name_terms=alt_name_terms,
                )
                _collect_snippets(snippet_pack)
                out.append(
                    {
                        "source_id": resolved_sid,
                        "chapter_id": cid if isinstance(cid, int) else fallback_sid,
                        "description": desc,
                    }
                )

            snippets = _finalize_snippets(actions_count=len(out))
            return out, snippets

        main_name = node.main_name
        alt_names = node.alt_names
        classification = node.classification
        existing_main_terms = self._extract_name_terms([str(main_name)])
        existing_alt_terms = self._extract_name_terms([str(a) for a in (alt_names or [])])
        new_main_terms = self._extract_name_terms([str(new_node.get("main_name", ""))])
        new_alt_terms = self._extract_name_terms([str(a) for a in (new_node.get("alt_names", []) or [])])
        existing_actions = list(node.actions) if isinstance(node.actions, list) else node.actions
        if isinstance(existing_actions, list) and isinstance(last_n, int) and last_n != 0:
            existing_actions = existing_actions[last_n:]
        all_prev_actions, existing_snippets = _normalize_actions_with_node_snippets(
            existing_actions, existing_main_terms, existing_alt_terms
        )
        new_actions, new_snippets = _normalize_actions_with_node_snippets(
            new_node.get("actions", ""),
            new_main_terms,
            new_alt_terms,
        )

        payload = {
            "task": "entity_verification",
            "source_id": list(source_id),
            "existing_entity": {
                "main_name": main_name,
                "alt_names": alt_names,
                "kinship_aliases": existing_kinship_aliases or [],
                "classification": classification,
                "actions": all_prev_actions,
                "snippets": existing_snippets,
            },
            "new_entity": {
                "main_name": new_node.get("main_name", ""),
                "alt_names": new_node.get("alt_names", []),
                "kinship_aliases": new_kinship_aliases or [],
                "classification": new_node.get("classification", ""),
                "actions": new_actions,
                "snippets": new_snippets,
            },
            "output_schema_hint": {
                "is_same_entity": "bool",
                "confidence": "высокий|средний|низкий",
                "key_evidence": "list[str]",
                "conflicting_attributes": "list[str]",
                "hard_conflict_flags": "list[str]",
                "kinship_anchors": {
                    "child_of": "list[str]",
                    "parent_of": "list[str]",
                    "grandchild_of": "list[str]",
                },
                "merge_blocked_by": "str|null",
            },
        }

        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        return {"messages": [("user", user_prompt)]}
