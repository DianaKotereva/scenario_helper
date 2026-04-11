import logging
import os
import time
from typing import Any, Dict, Tuple

from src.utils.graph_search import BookNode

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

try:
    from openai import APIConnectionError, APITimeoutError
except Exception:  # pragma: no cover
    APIConnectionError = ()
    APITimeoutError = ()


logger = logging.getLogger(__name__)


class VerificationService:
    """Service for entity verification in graph merge."""

    def __init__(self, verificator):
        self.verificator = verificator
        self.connection_retry_count = max(
            1, int(os.getenv("VERIFICATION_CONNECTION_RETRY_COUNT", "3"))
        )

    def _is_retryable_connection_error(self, err: Exception) -> bool:
        """Detect transient transport/network failures from LLM stack."""

        checked = set()
        cur = err
        while cur and id(cur) not in checked:
            checked.add(id(cur))

            if APIConnectionError and isinstance(cur, APIConnectionError):
                return True
            if APITimeoutError and isinstance(cur, APITimeoutError):
                return True
            if httpx and isinstance(
                cur,
                (
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.ReadTimeout,
                    httpx.WriteTimeout,
                    httpx.RemoteProtocolError,
                    httpx.NetworkError,
                ),
            ):
                return True

            msg = str(cur).lower()
            if any(
                marker in msg
                for marker in (
                    "connection error",
                    "unexpected_eof_while_reading",
                    "eof occurred in violation of protocol",
                    "timed out",
                    "connection reset",
                    "temporarily unavailable",
                )
            ):
                return True

            cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)

        return False

    def verify_nodes(
        self,
        node: BookNode,
        new_node: Dict[str, Any],
        source_id: Tuple[int],
        last_n: int = -10,
    ) -> bool:
        # Fast guard: different classifications => not the same entity.
        if node.classification != new_node.get("classification"):
            return False

        for attempt in range(1, self.connection_retry_count + 1):
            try:
                res_verify = self.verificator.invoke(
                    node=node,
                    new_node=new_node,
                    source_id=source_id,
                    last_n=last_n,
                )
                if isinstance(res_verify, dict):
                    return res_verify.get("is_same_entity", False)
                return bool(res_verify)
            except Exception as e:
                is_retryable = self._is_retryable_connection_error(e)
                is_last_attempt = attempt >= self.connection_retry_count

                if is_retryable and not is_last_attempt:
                    backoff_sec = min(0.5 * (2 ** (attempt - 1)), 3.0)
                    logger.warning(
                        "Connection error during verification %s vs %s (attempt %s/%s). Retrying in %.1fs: %s",
                        node.main_name,
                        new_node.get("main_name"),
                        attempt,
                        self.connection_retry_count,
                        backoff_sec,
                        e,
                    )
                    time.sleep(backoff_sec)
                    continue

                logger.error(
                    "Error during node verification %s vs %s (attempt %s/%s): %s",
                    node.main_name,
                    new_node.get("main_name"),
                    attempt,
                    self.connection_retry_count,
                    e,
                    exc_info=True,
                )
                return False

        return False
