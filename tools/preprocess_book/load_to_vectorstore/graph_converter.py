"""
Модуль для преобразования BookGraph в Document объекты.

Содержит функции для конвертации узлов и отношений графа знаний
в формат Document для загрузки в векторное хранилище.
"""

import logging
from typing import List
from langchain_core.documents import Document
from src.utils.graph_search import BookGraph

logger = logging.getLogger(__name__)


def book_graph_to_documents(book_graph: BookGraph) -> List[Document]:
    """
    Преобразует BookGraph в список Document объектов для векторного хранилища.
    
    Создает Document объекты из:
    - Узлов графа (nodes) - с метаданными {"source": "nodes", "name": node_name}
    - Отношений графа (relations) - с метаданными {"source": "relations", "name": (object_1, object_2)}
    
    Args:
        book_graph: Граф знаний книги
        
    Returns:
        Список Document объектов с узлами и отношениями
    """
    documents = []
    
    try:
        # Преобразуем узлы графа
        for node_name, node in book_graph.nodes.nodes.items():
            try:
                # Объединяем все действия узла в один текст
                actions_texts = []
                for action in node.actions:
                    if hasattr(action, 'action') and action.action:
                        actions_texts.append(action.action)
                        # Используем source_id из первого действия
                        source_id = action.source_id if hasattr(action, 'source_id') else None
                
                if not actions_texts:
                    # Если нет действий, пропускаем узел
                    continue
                
                # Если source_id не найден, используем 0 как fallback
                if source_id is None:
                    source_id = (0,)
                
                page_content = " ".join(actions_texts)
                
                # Создаем Document для узла
                node_doc = Document(
                    page_content=page_content,
                    metadata={
                        "source_id": source_id,
                        "source": "nodes",
                        "name": node_name,
                        "classification": node.classification,
                    }
                )
                documents.append(node_doc)
                
            except Exception as e:
                logger.error(f"Ошибка при преобразовании узла '{node_name}': {e}", exc_info=True)
                continue
        
        # Преобразуем отношения графа
        for rel_key, relation in book_graph.relationships.relationships.items():
            try:
                # Объединяем все описания отношения
                descriptions_texts = []
                source_ids = set()
                
                for desc in relation.description:
                    if hasattr(desc, 'description') and desc.description:
                        descriptions_texts.append(desc.description)
                    if hasattr(desc, 'source_id'):
                        source_ids.add(desc.source_id)
                
                if not descriptions_texts:
                    # Если нет описаний, пропускаем отношение
                    continue
                
                # Используем первый source_id или fallback
                source_id = list(source_ids)[0] if source_ids else (0,)
                
                page_content = " ".join(descriptions_texts)
                
                # Создаем Document для отношения
                rel_doc = Document(
                    page_content=page_content,
                    metadata={
                        "source_id": source_id,
                        "source": "relations",
                        "name": (relation.object_1, relation.object_2),
                    }
                )
                documents.append(rel_doc)
                
            except Exception as e:
                logger.error(
                    f"Ошибка при преобразовании отношения '{rel_key}': {e}",
                    exc_info=True
                )
                continue
        
        logger.info(
            f"Преобразовано {len([d for d in documents if d.metadata.get('source') == 'nodes'])} узлов "
            f"и {len([d for d in documents if d.metadata.get('source') == 'relations'])} отношений"
        )
        
    except Exception as e:
        logger.error(f"Критическая ошибка при преобразовании графа: {e}", exc_info=True)
        raise
    
    return documents
