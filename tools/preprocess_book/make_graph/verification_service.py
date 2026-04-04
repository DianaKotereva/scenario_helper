import logging
from typing import Any, Dict, Tuple

from src.utils.graph_search import BookNode

logger = logging.getLogger(__name__)


class VerificationService:
    """Сервис для верификации сущностей."""

    def __init__(self, verificator):
        self.verificator = verificator

    def verify_nodes(
        self,
        node: BookNode,
        new_node: Dict[str, Any],
        source_id: Tuple[int],
        last_n: int = -10,
    ) -> bool:
        # Быстрая проверка: разные классификации = разные сущности
        if node.classification != new_node.get("classification"):
            return False

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
            logger.error(
                "Ошибка при верификации узлов %s и %s: %s",
                node.main_name,
                new_node.get("main_name"),
                e,
                exc_info=True,
            )
            return False
