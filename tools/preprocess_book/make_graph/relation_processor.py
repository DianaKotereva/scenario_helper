import logging
from typing import List, Dict, Tuple
from src.utils.graph_search import (
    AllBookNodes,
    AllBooksEdges,
    BookEdges,
    Description,
)

logger = logging.getLogger(__name__)


class RelationProcessor:
    """Процессор для обработки отношений между узлами."""
    
    @staticmethod
    def process_relations(
        all_book_nodes: AllBookNodes,
        rel_inputs: List[Dict],
        rel_graphs: AllBooksEdges,
        source_id: Tuple[int],
    ) -> AllBooksEdges:
        """
        Обрабатывает отношения между узлами и обновляет граф отношений.
        
        Args:
            all_book_nodes: Объект AllBookNodes с узлами
            rel_inputs: Список словарей с данными отношений
            rel_graphs: Объект AllBooksEdges для обновления
            source_id: ID источника данных
            
        Returns:
            Обновленный объект AllBooksEdges
        """
        all_mains = list(all_book_nodes.nodes.keys())
        rel_inputs = [i for i in rel_inputs if i]
        rel_inputs = [
            i
            for i in rel_inputs
            if i.get("source_node_id") in all_mains 
            and i.get("target_node_id") in all_mains
        ]

        for rel in rel_inputs:
            try:
                source_id_rel = rel.get("source_node_id")
                target_id_rel = rel.get("target_node_id")
                
                if not source_id_rel or not target_id_rel:
                    logger.warning(f"Пропущено отношение с пустыми ID: {rel}")
                    continue
                
                pair = tuple(sorted([source_id_rel, target_id_rel]))
                description = Description(
                    rel.get("description", ""), 
                    type=rel.get("type", ""), 
                    source_id=source_id
                )

                if pair[0] == source_id_rel:
                    object_1 = source_id_rel
                    object_2 = target_id_rel
                else:
                    object_1 = target_id_rel
                    object_2 = source_id_rel

                target_rel = rel_graphs.relationships.get(
                    pair,
                    BookEdges(
                        object_1=object_1, 
                        object_2=object_2, 
                        description=[description]
                    ),
                )

                if description not in target_rel.description:
                    target_rel.description.append(description)

                rel_graphs.relationships[pair] = target_rel
                
            except Exception as e:
                logger.error(
                    f"Ошибка при обработке отношения для source_id {source_id}: {e}",
                    exc_info=True
                )
                continue

        return rel_graphs
