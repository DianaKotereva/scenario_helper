import logging
from typing import List, Dict, Tuple
from tools.preprocess_book.make_graph.verification_service import VerificationService
from src.utils.graph_search import BookNode, Action, AllBookNodes

logger = logging.getLogger(__name__)


class NodeProcessor:
    def __init__(self, verification_service: VerificationService):
        self.verification_service = verification_service

    def process_input(
        self,
        input_data: List[Dict],
        rel_inputs: List[Dict],
        source_id: Tuple[int],
        all_book_nodes: AllBookNodes,
        last_n: int = -10,
    ) -> AllBookNodes:
        """
        Обрабатывает входные данные и обновляет граф узлов.
        
        Args:
            input_data: Список словарей с данными узлов
            rel_inputs: Список словарей с данными отношений (не используется, но оставлен для совместимости)
            source_id: ID источника данных
            all_book_nodes: Объект AllBookNodes для обновления
            last_n: Количество последних действий для верификации
            
        Returns:
            Обновленный объект AllBookNodes
        """
        for item in input_data:
            try:
                names = [item["main_name"]] + item["alt_names"]
                all_names_to_use = [
                    use_n
                    for use_n in list(all_book_nodes.names_list.keys())
                    if max(
                        list(
                            map(
                                lambda x: x.lower() in use_n.lower()
                                or use_n.lower() in x.lower(),
                                names,
                            )
                        )
                    )
                ]
                existing_mains = {
                    all_book_nodes.names_list[name] for name in all_names_to_use
                }

                target_nodes = []
                for target_main in existing_mains:
                    if self.verification_service.verify_nodes(
                        all_book_nodes.nodes[target_main], item, source_id, last_n
                    ):
                        target_nodes.append(target_main)
                        break

                if target_nodes:
                    target_main = target_nodes[0]
                    target_node = all_book_nodes.nodes[target_main]
                else:
                    target_main = item["main_name"]
                    target_node = BookNode(
                        main_name=target_main,
                        classification=item["classification"],
                        alt_names=item["alt_names"],
                        actions=[Action(action=item["actions"], source_id=source_id)],
                    )
                    all_book_nodes.nodes[target_main] = target_node

                for name in names:
                    all_book_nodes.names_list[name] = target_main

                if (
                    target_main != item["main_name"]
                    and item["main_name"] not in target_node.alt_names
                ):
                    target_node.alt_names.append(item["main_name"])

                for alt in item["alt_names"]:
                    if alt != target_main and alt not in target_node.alt_names:
                        target_node.alt_names.append(alt)

                if Action(item["actions"], source_id=source_id) not in target_node.actions:
                    target_node.actions.append(
                        Action(action=item["actions"], source_id=source_id)
                    )
            except Exception as e:
                logger.error(
                    f"Ошибка при обработке узла {item.get('main_name', 'unknown')} "
                    f"для source_id {source_id}: {e}",
                    exc_info=True
                )
                continue

        return all_book_nodes
