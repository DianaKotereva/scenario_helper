from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Tuple

from langchain_core.output_parsers import JsonOutputParser

from src.utils.graph_search import Action, BookNode
from tools.preprocess_book.prompts.verificator import Verification
from tools.preprocess_book.utils.llm_factory import create_llm

logger = logging.getLogger(__name__)


class LLMJudge:
    def __init__(
        self,
        llm_type: str | None = None,
        enabled: bool = True,
        verification_last_n: int = -50,
    ):
        self.enabled = enabled
        self.verification_last_n = int(verification_last_n)
        self._verifier = None
        if enabled:
            llm = create_llm(llm_type=llm_type)
            self._verifier = Verification(llm=llm, parser=JsonOutputParser())

    @staticmethod
    def _cluster_to_book_node(cluster) -> BookNode:
        actions = [Action(action=a, source_id=sid) for a, sid in cluster.actions]
        return BookNode(
            main_name=cluster.canonical_name,
            classification=cluster.classification,
            alt_names=list(cluster.alt_names),
            actions=actions,
        )

    @staticmethod
    def _is_transient_error(err: Exception) -> bool:
        text = str(err).lower()
        markers = [
            "api connection error",
            "connection error",
            "connecterror",
            "unexpected eof while reading",
            "timed out",
            "timeout",
            "temporary failure",
            "ssl",
            "502",
            "503",
            "504",
            "invalid json output",
            "output_parsing_failure",
            "jsondecodeerror",
        ]
        return any(m in text for m in markers)

    @staticmethod
    def _normalize_payload(payload: Any) -> Dict[str, Any]:
        if isinstance(payload, bool):
            return {
                "is_same_entity": payload,
                "confidence": "unknown",
                "key_evidence": [],
                "conflicting_attributes": [],
                "hard_conflict_flags": [],
                "kinship_anchors": {
                    "child_of": [],
                    "parent_of": [],
                    "grandchild_of": [],
                },
                "merge_blocked_by": None,
            }
        if isinstance(payload, dict):
            normalized = dict(payload)
            normalized["is_same_entity"] = bool(normalized.get("is_same_entity", False))
            normalized["confidence"] = str(normalized.get("confidence", "unknown"))
            normalized["key_evidence"] = list(normalized.get("key_evidence", []) or [])
            normalized["conflicting_attributes"] = list(
                normalized.get("conflicting_attributes", []) or []
            )
            normalized["hard_conflict_flags"] = list(
                normalized.get("hard_conflict_flags", []) or []
            )
            kinship = normalized.get("kinship_anchors") or {}
            normalized["kinship_anchors"] = {
                "child_of": list((kinship.get("child_of") or [])),
                "parent_of": list((kinship.get("parent_of") or [])),
                "grandchild_of": list((kinship.get("grandchild_of") or [])),
            }
            normalized["merge_blocked_by"] = normalized.get("merge_blocked_by")
            return normalized
        return {
            "is_same_entity": False,
            "confidence": "low",
            "key_evidence": ["Unexpected verifier output"],
            "conflicting_attributes": [],
            "hard_conflict_flags": [],
            "kinship_anchors": {
                "child_of": [],
                "parent_of": [],
                "grandchild_of": [],
            },
            "merge_blocked_by": "unexpected_verifier_output",
        }

    @staticmethod
    def _fail_closed_payload(reason: str, err: Exception | None = None) -> Dict[str, Any]:
        evidence = [reason]
        if err is not None:
            evidence.append(f"{type(err).__name__}: {err}")
        return {
            "is_same_entity": False,
            "confidence": "low",
            "key_evidence": evidence,
            "conflicting_attributes": [],
            "hard_conflict_flags": ["judge_fail_closed"],
            "kinship_anchors": {
                "child_of": [],
                "parent_of": [],
                "grandchild_of": [],
            },
            "merge_blocked_by": "judge_output_parse_error",
        }

    def _invoke_once(self, cluster, new_node: Dict[str, Any], source_id: Tuple[int, ...]) -> Dict[str, Any]:
        payload = self._verifier.invoke(
            node=self._cluster_to_book_node(cluster),
            new_node=new_node,
            source_id=source_id,
            last_n=self.verification_last_n,
            existing_kinship_aliases=list(getattr(cluster, "kinship_aliases", [])),
            new_kinship_aliases=list(new_node.get("kinship_aliases", []) or []),
        )
        return self._normalize_payload(payload)

    def verify(self, cluster, new_node: Dict[str, Any], source_id: Tuple[int, ...]) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "is_same_entity": False,
                "confidence": "disabled",
                "key_evidence": ["LLM judge disabled"],
                "conflicting_attributes": [],
                "hard_conflict_flags": [],
                "kinship_anchors": {
                    "child_of": [],
                    "parent_of": [],
                    "grandchild_of": [],
                },
                "merge_blocked_by": "judge_disabled",
            }

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                return self._invoke_once(cluster, new_node, source_id)
            except Exception as err:  # noqa: BLE001
                if attempt < max_attempts and self._is_transient_error(err):
                    delay = min(6, 1.5 * attempt)
                    logger.warning(
                        "Transient LLM judge error (attempt %s/%s): %s. Retry in %.1fs",
                        attempt,
                        max_attempts,
                        err,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                logger.error(
                    "LLM judge verify failed after retries (sync). Using fail-closed decision. error=%s",
                    err,
                )
                return self._fail_closed_payload(
                    "LLM judge failed after retries (sync)",
                    err,
                )

        return self._fail_closed_payload("LLM judge failed after retries (sync)")

    async def verify_async(
        self,
        cluster,
        new_node: Dict[str, Any],
        source_id: Tuple[int, ...],
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "is_same_entity": False,
                "confidence": "disabled",
                "key_evidence": ["LLM judge disabled"],
                "conflicting_attributes": [],
                "hard_conflict_flags": [],
                "kinship_anchors": {
                    "child_of": [],
                    "parent_of": [],
                    "grandchild_of": [],
                },
                "merge_blocked_by": "judge_disabled",
            }

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                return await asyncio.to_thread(self._invoke_once, cluster, new_node, source_id)
            except Exception as err:  # noqa: BLE001
                if attempt < max_attempts and self._is_transient_error(err):
                    delay = min(6, 1.5 * attempt)
                    logger.warning(
                        "Transient async LLM judge error (attempt %s/%s): %s. Retry in %.1fs",
                        attempt,
                        max_attempts,
                        err,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(
                    "LLM judge verify failed after retries (async). Using fail-closed decision. error=%s",
                    err,
                )
                return self._fail_closed_payload(
                    "LLM judge failed after retries (async)",
                    err,
                )

        return self._fail_closed_payload("LLM judge failed after retries (async)")
