import logging
from typing import Any, Dict, List, Tuple

from src.utils.graph_search import Action, AllBookNodes, BookNode
from tools.preprocess_book.make_graph.verification_service import VerificationService

logger = logging.getLogger(__name__)


class NodeProcessor:
    def __init__(self, verification_service: VerificationService):
        self.verification_service = verification_service

    @staticmethod
    def _to_source_tuple(value: Any, fallback: Tuple[int, ...]) -> Tuple[int, ...]:
        if isinstance(value, int):
            return (value,)
        if isinstance(value, tuple) and value and isinstance(value[0], int):
            return value
        if isinstance(value, list) and value and isinstance(value[0], int):
            return tuple(value)
        return fallback

    @staticmethod
    def _format_action_text(description: str, quotes: List[str], chapter_id: Any) -> str:
        desc = (description or "").strip()
        if not desc:
            return ""
        return desc

    @staticmethod
    def _name_match(a: str, b: str) -> bool:
        a_l = (a or "").strip().lower()
        b_l = (b or "").strip().lower()
        if not a_l or not b_l:
            return False
        return a_l in b_l or b_l in a_l

    @staticmethod
    def _normalize_classification(value: Any) -> str:
        raw = str(value or "").strip()
        mapping = {
            "EntityClassification.PERSON": "\u043f\u0435\u0440\u0441\u043e\u043d\u0430\u0436",
            "EntityClassification.PLACE": "\u043c\u0435\u0441\u0442\u043e",
            "EntityClassification.ORG": "\u043e\u0440\u0433\u0430\u043d\u0438\u0437\u0430\u0446\u0438\u044f",
            "EntityClassification.TERM": "\u0442\u0435\u0440\u043c\u0438\u043d",
            "EntityClassification.FORCE": "\u0441\u0438\u043b\u0430 \u043f\u0440\u0438\u0440\u043e\u0434\u044b",
        }
        return mapping.get(raw, raw)

    @staticmethod
    def _collect_node_source_ids(node_obj: BookNode) -> Tuple[int, ...]:
        source_ids = set()
        for action in node_obj.actions or []:
            sid = getattr(action, "source_id", None)
            if isinstance(sid, int):
                source_ids.add(int(sid))
            elif isinstance(sid, tuple):
                source_ids.update(int(v) for v in sid if isinstance(v, int))
            elif isinstance(sid, list):
                source_ids.update(int(v) for v in sid if isinstance(v, int))
        return tuple(sorted(source_ids))

    def _collect_candidate_mains(
        self,
        names: List[str],
        all_book_nodes: AllBookNodes,
        current_sid: int,
    ) -> List[str]:
        """Collect candidate node keys for verification without relying on single alias mapping."""
        candidates = set()

        # Fast path via names_list index.
        for alias, main_key in all_book_nodes.names_list.items():
            if any(self._name_match(alias, name) for name in names):
                if main_key in all_book_nodes.nodes:
                    candidates.add(main_key)

        # Robust path: scan current graph to avoid alias-map collisions.
        for node_key, node_obj in all_book_nodes.nodes.items():
            node_names = [node_obj.main_name] + list(node_obj.alt_names or [])
            if any(self._name_match(left, right) for left in names for right in node_names):
                candidates.add(node_key)

        def _has_previous_source(node_obj: BookNode) -> bool:
            for sid in self._collect_node_source_ids(node_obj):
                if sid < current_sid:
                    return True
            return False

        return [
            key
            for key in candidates
            if key in all_book_nodes.nodes and _has_previous_source(all_book_nodes.nodes[key])
        ]

    @staticmethod
    def _make_internal_key(
        main_name: str,
        source_id: Tuple[int, ...],
        all_book_nodes: AllBookNodes,
    ) -> str:
        sid = source_id[0] if source_id else -1
        base = f"{main_name}__sid{sid}"
        if base not in all_book_nodes.nodes:
            return base
        counter = 2
        while f"{base}_{counter}" in all_book_nodes.nodes:
            counter += 1
        return f"{base}_{counter}"

    def _extract_action_objects(self, item: Dict[str, Any], source_id: Tuple[int, ...]) -> List[Action]:
        raw_actions = item.get("actions")
        fallback_sid = source_id[0] if source_id else 0
        actions: List[Action] = []

        if isinstance(raw_actions, str):
            text = raw_actions.strip()
            if text:
                actions.append(Action(action=text, source_id=source_id))
            return actions

        if not isinstance(raw_actions, list):
            return actions

        for action_item in raw_actions:
            if isinstance(action_item, str):
                text = action_item.strip()
                if text:
                    actions.append(Action(action=text, source_id=source_id))
                continue

            if not isinstance(action_item, dict):
                continue

            sid = action_item.get("source_id", fallback_sid)
            cid = action_item.get("chapter_id", sid)
            desc = action_item.get("description")
            quotes = action_item.get("quotes") if isinstance(action_item.get("quotes"), list) else []

            text = self._format_action_text(description=str(desc or ""), quotes=quotes, chapter_id=cid)
            if not text:
                continue

            action_source_id = self._to_source_tuple(sid, source_id)
            chapter_id = int(cid) if isinstance(cid, int) else action_source_id[0]
            quotes_clean = [q.strip() for q in (quotes or []) if isinstance(q, str) and q.strip()]
            actions.append(
                Action(
                    action=text,
                    source_id=action_source_id,
                    chapter_id=chapter_id,
                    quotes=quotes_clean,
                )
            )

        return actions

    def process_input(
        self,
        input_data: List[Dict[str, Any]],
        rel_inputs: List[Dict[str, Any]],
        source_id: Tuple[int, ...],
        all_book_nodes: AllBookNodes,
        last_n: int = -10,
    ) -> AllBookNodes:
        current_sid = source_id[0] if source_id else 0

        for item in input_data:
            try:
                main_name = str(item.get("main_name", "")).strip()
                if not main_name:
                    continue

                alt_names_raw = item.get("alt_names")
                alt_names = [
                    str(name).strip()
                    for name in (alt_names_raw or [])
                    if isinstance(name, str) and str(name).strip()
                ]
                names = [main_name] + alt_names

                existing_mains = self._collect_candidate_mains(
                    names=names,
                    all_book_nodes=all_book_nodes,
                    current_sid=current_sid,
                )

                new_classification = self._normalize_classification(item.get("classification"))

                target_main = None
                for existing_main in existing_mains:
                    if self.verification_service.verify_nodes(
                        all_book_nodes.nodes[existing_main], item, source_id, last_n
                    ):
                        target_main = existing_main
                        break

                if target_main is not None:
                    target_node = all_book_nodes.nodes[target_main]
                else:
                    # Non-destructive homonym handling:
                    # if same main_name exists but verify=False, create unique internal key.
                    if main_name in all_book_nodes.nodes:
                        target_main = self._make_internal_key(main_name, source_id, all_book_nodes)
                    else:
                        target_main = main_name
                    target_node = BookNode(
                        main_name=main_name,
                        classification=new_classification,
                        alt_names=alt_names,
                        actions=[],
                    )
                    all_book_nodes.nodes[target_main] = target_node

                for name in names:
                    # Keep first mapping for ambiguous aliases; add disambiguated alias for twins.
                    if name not in all_book_nodes.names_list or all_book_nodes.names_list.get(name) == target_main:
                        all_book_nodes.names_list[name] = target_main
                    else:
                        disambiguated = f"{name}__sid{current_sid}"
                        all_book_nodes.names_list[disambiguated] = target_main
                # Always map internal key itself for deterministic relation canonicalization.
                all_book_nodes.names_list[target_main] = target_main

                if target_main != main_name and main_name not in target_node.alt_names:
                    target_node.alt_names.append(main_name)

                for alt in alt_names:
                    if alt != target_main and alt not in target_node.alt_names:
                        target_node.alt_names.append(alt)

                for action_obj in self._extract_action_objects(item, source_id):
                    if action_obj not in target_node.actions:
                        target_node.actions.append(action_obj)
            except Exception as ex:
                logger.error(
                    "Error while processing node '%s' for source_id=%s: %s",
                    item.get("main_name", "unknown"),
                    source_id,
                    ex,
                    exc_info=True,
                )
                continue

        return all_book_nodes
