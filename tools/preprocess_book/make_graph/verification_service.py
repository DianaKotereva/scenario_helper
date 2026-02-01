import logging
from typing import Dict, Any, Tuple
from src.utils.graph_search import BookNode

logger = logging.getLogger(__name__)


class VerificationService:
    """Сервис для верификации сущностей."""
    
    def __init__(self, verificator):
        """
        Args:
            verificator: Экземпляр Verification класса для верификации
        """
        self.verificator = verificator

    def verify_nodes(
        self,
        node: BookNode,
        new_node: Dict[str, Any],
        source_id: Tuple[int],
        last_n: int = -10,
    ) -> bool:
        """
        Проверяет, относятся ли существующий и новый узлы к одной сущности.
        
        Args:
            node: Существующий узел BookNode
            new_node: Новый узел в виде словаря
            source_id: ID источника
            last_n: Количество последних действий для сравнения
            
        Returns:
            True если сущности одинаковые, False иначе
        """
        # Быстрая проверка: разные классификации = разные сущности
        if node.classification != new_node.get("classification"):
            return False
        
        try:
            res_verify = self.verificator.invoke(
                node=node,
                new_node=new_node,
                source_id=source_id,
                last_n=last_n
            )
            
            # Обработка результата (может быть dict или уже bool)
            if isinstance(res_verify, dict):
                return res_verify.get("is_same_entity", False)
            return bool(res_verify)
            
        except Exception as e:
            logger.error(
                f"Ошибка при верификации узлов {node.main_name} и {new_node.get('main_name')}: {e}",
                exc_info=True
            )
            return False


# TODO: добавить батчевый режим для верификации
