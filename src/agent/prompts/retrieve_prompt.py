"""
Промпт и агент для поиска и обработки информации из графа знаний.

Модуль содержит промпт и класс RetrieveAgent для поиска информации
в векторном хранилище и графе знаний, а также формирования ответа.
"""

import logging
from typing import Any, Dict, Optional, Set, Tuple, List

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from src.agent.vector_store.retriever import retriever
from src.llm_core.llm_core import llm
from src.llm_core.llm_prompt_base import LLMBase
from src.utils.graph_search import BookGraph, book_graph
from src.agent.prompts.output_models import RetrieveOutput
from src.config import settings

logger = logging.getLogger(__name__)

question_answer_prompt = """Ты — эксперт-аналитик, формирующий детальный и подробный ответ на вопрос пользователя по книге. 

# ЗАДАЧА:
Дать ответ ИСКЛЮЧИТЕЛЬНО на основе:
1. Исходный вопрос пользователя: str //Исходный вопрос, который задал пользователь
3. Собранный контекст: List[str] // Уже имеющийся у тебя контекст. Контекст состоит из:
    - Цитаты из книги (небольшие фрагменты из векторного поиска)
    - Суммаризации глав
    - Полные тексты глав (если доступны) - используются для более глубокого понимания контекста
    - Описание действий персонажей/объектов книги, поданные в формате **Действия персонажа** - описание действий персонажа
    - Описание взаимодействия между разными объектами книги, поданные в формате **Взаимодействие между ___ и ___** - описание взаимодействия между персонажами 

# ИНСТРУКЦИИ:

1. Анализ:
   - Сопоставь каждый элемент контекста с запросом
   - Выяви ключевые доказательные факты
   - Опирайся только на ту информацию в контексте, которая имеет отношение к исходному запросу пользователя, не используй нерелевантную информацию

2. Формирование ответа:
   - Структурируй ответ: тезис → доказательства → вывод
   - Используй ТОЧНЫЕ цитаты из контекста
   - Сохрани цепочку логических умозаключений
   - Напиши как можно более подробный ответ, опирающийся на контекст. Не упускай детали, опиши все максимально подробно.

3. Контроль качества:
   - Если информации НЕДОСТАТОЧНО → явно укажи это
   - Запрещены домыслы/интерпретации
   - Ответ должен покрывать ВСЕ аспекты запроса, быть максимально подробным, структурированным и точным. Постарайся ничего не упустить. 

# ПРИМЕРЫ:
Запрос: "Каковы мотивы главного героя в романе X?"
Правильный ответ: {{
  "answer": "Согласно анализу главных диалогов (с. 45-47) и авторским комментариям (с. 112):\n1. Основной мотив - ...\n2. Второстепенный мотив - ...\nВывод: Совокупность факторов указывает на..." 
}}

Недопустимый ответ: "Герой хотел добиться успеха, возможно из-за детских травм" 

# ТРЕБОВАНИЯ К ВЫВОДУ:
- ТОЛЬКО JSON-объект с ключом "answer"
- Ответ на языке оригинала запроса
- Четкая структура с указанием источников
- Минимум 3 доказательных пункта при наличии данных
- Объем: 150-300 слов

###
Ты должен вернуть ответ в следующем формате JSON:

{
  "answer": str // Детальный ответ (минимум 20 символов). Если информации недостаточно, явно укажи это.
}

# ТРЕБОВАНИЯ К ВАЛИДАЦИИ:
- answer: минимум 20 символов
- Ответ должен быть структурированным и содержать конкретную информацию
- Если информации недостаточно, явно укажи это в ответе

{{format_instructions}}

Сформируй ответ:"""


class RetrieveAgent(LLMBase):
    """
    Агент для поиска информации в векторном хранилище и графе знаний.
    
    Выполняет поиск по запросу пользователя, обрабатывает результаты
    из графа знаний (узлы и отношения) и формирует структурированный ответ.
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        system_prompt: str,
        retriever: BaseRetriever,
        book_graph: BookGraph,
        parser: Optional[BaseOutputParser] = None,
        load_template: bool = False,
    ):
        """
        Инициализирует RetrieveAgent.
        
        Args:
            llm: Инициализированная языковая модель
            system_prompt: Базовый системный промпт
            retriever: Ретривер для поиска в векторном хранилище
            book_graph: Граф знаний книги
            parser: Парсер для вывода (рекомендуется PydanticOutputParser с RetrieveOutput)
            load_template: Загружать ли промпт из файла
        """
        self.llm = llm
        self._system_prompt = system_prompt
        self._retriever = retriever
        self._prompt_template: Optional[ChatPromptTemplate] = None
        self._parser = parser
        self._book_graph = book_graph
        self._load_template = load_template

    def process_nodes(self, nodes_names: Set[str]) -> str:
        """
        Обрабатывает узлы графа и формирует текстовое описание их действий.
        
        Args:
            nodes_names: Множество имен узлов для обработки
            
        Returns:
            Текстовое описание действий всех узлов
            
        Raises:
            KeyError: Если узел не найден в графе
        """
        all_node_texts = []
        
        for node_name in nodes_names:
            try:
                if node_name not in self._book_graph.nodes.nodes:
                    logger.warning(f"Node '{node_name}' not found in graph, skipping")
                    continue
                    
                node = self._book_graph.nodes.nodes[node_name]
                actions = [action.action for action in node.actions if hasattr(action, 'action')]
                
                node_text = (
                    f"**Действия {node_name}**\n"
                    + "\n".join(actions)
                    + "\n****"
                )
                all_node_texts.append(node_text)
                
            except Exception as e:
                logger.error(f"Error processing node '{node_name}': {e}", exc_info=True)
                continue
        
        return "\n\n".join(all_node_texts)

    def process_rels(self, rel_names: Set[Tuple[str, str]]) -> str:
        """
        Обрабатывает отношения графа и формирует текстовое описание взаимодействий.
        
        Args:
            rel_names: Множество кортежей (имя_узла1, имя_узла2) для обработки
            
        Returns:
            Текстовое описание всех отношений
            
        Raises:
            KeyError: Если отношение не найдено в графе
        """
        all_rels_texts = []
        
        for rel_name in rel_names:
            try:
                if rel_name not in self._book_graph.relationships.relationships:
                    logger.warning(f"Relation '{rel_name}' not found in graph, skipping")
                    continue
                    
                rel = self._book_graph.relationships.relationships[rel_name]
                object_1 = rel.object_1
                object_2 = rel.object_2
                
                descriptions = [
                    desc.description 
                    for desc in rel.description 
                    if hasattr(desc, 'description')
                ]
                desc = "\n".join(descriptions) + "\n****"
                
                rel_text = (
                    f"**Взаимодействие между {object_1} и {object_2}**\n{desc}"
                )
                all_rels_texts.append(rel_text)
                
            except Exception as e:
                logger.error(f"Error processing relation '{rel_name}': {e}", exc_info=True)
                continue
        
        return "\n\n".join(all_rels_texts)

    def process_graph(
        self, 
        rels_result: List[Document], 
        nodes_result: List[Document]
    ) -> Dict[str, str]:
        """
        Обрабатывает результаты поиска из графа знаний.
        
        Извлекает имена узлов и отношений из метаданных документов,
        обрабатывает их и возвращает текстовые описания.
        
        Args:
            rels_result: Список документов с результатами поиска отношений
            nodes_result: Список документов с результатами поиска узлов
            
        Returns:
            Словарь с ключами "nodes_texts" и "rel_texts"
        """
        try:
            # Извлекаем имена узлов
            nodes_names: Set[str] = set()
            for doc in nodes_result:
                if hasattr(doc, 'metadata') and "name" in doc.metadata:
                    name = doc.metadata["name"]
                    if isinstance(name, str):
                        nodes_names.add(name)
            
            # Извлекаем имена отношений
            rel_names: Set[Tuple[str, str]] = set()
            for doc in rels_result:
                if hasattr(doc, 'metadata') and "name" in doc.metadata:
                    name = doc.metadata["name"]
                    if isinstance(name, tuple) and len(name) == 2:
                        rel_names.add(name)
                    elif isinstance(name, (list, tuple)) and len(name) >= 2:
                        rel_names.add((name[0], name[1]))
            
            # Добавляем узлы из отношений
            for rel_name in rel_names:
                nodes_names.add(rel_name[0])
                nodes_names.add(rel_name[1])
            
            # Обрабатываем узлы и отношения
            node_text = self.process_nodes(nodes_names)
            rels_text = self.process_rels(rel_names)
            
            return {"nodes_texts": node_text, "rel_texts": rels_text}
            
        except Exception as e:
            logger.error(f"Error processing graph results: {e}", exc_info=True)
            return {"nodes_texts": "", "rel_texts": ""}

    def invoke(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполняет поиск и формирует ответ на запрос пользователя.
        
        Args:
            **kwargs: Аргументы для поиска (должен содержать 'query')
            
        Returns:
            Словарь с ключом "answer" содержащим ответ
            
        Raises:
            ValueError: Если query не передан
            RuntimeError: Если поиск или формирование ответа не удалось
        """
        try:
            query = kwargs.get("query")
            if not query or not isinstance(query, str):
                raise ValueError("query must be a non-empty string")
            
            # Получаем входные данные для промпта
            chain_input = self.make_user_prompt_retrieve(query=query)
            
            # Выполняем поиск
            try:
                search_result = self._retriever.invoke(chain_input)
            except Exception as e:
                logger.error(f"Error in retriever.invoke: {e}", exc_info=True)
                return {"answer": "Ошибка при поиске информации. Попробуйте переформулировать вопрос."}
            
            # Разделяем результаты по источникам
            quote_result = [
                doc for doc in search_result 
                if hasattr(doc, 'metadata') and doc.metadata.get("source") == "book"
            ]
            nodes_result = [
                doc for doc in search_result 
                if hasattr(doc, 'metadata') and doc.metadata.get("source") == "nodes"
            ]
            rels_result = [
                doc for doc in search_result 
                if hasattr(doc, 'metadata') and doc.metadata.get("source") == "relations"
            ]
            sums_result = [
                doc for doc in search_result 
                if hasattr(doc, 'metadata') and doc.metadata.get("source") == "summary"
            ]
            
            # Обрабатываем граф знаний
            answer_input = self.process_graph(rels_result, nodes_result)
            
            # Форматируем цитаты и суммаризации
            quote_texts = "\n***\n".join([
                f"{doc.page_content}. Глава {doc.metadata.get('source_id', 'N/A')}"
                for doc in quote_result
                if hasattr(doc, 'page_content')
            ])
            
            sums_texts = "\n***\n".join([
                f"{doc.page_content}. Глава {doc.metadata.get('source_id', 'N/A')}"
                for doc in sums_result
                if hasattr(doc, 'page_content')
            ])
            
            answer_input["quote_texts"] = quote_texts
            answer_input["sums_texts"] = sums_texts
            
            # Извлекаем уникальные source_id из найденных чанков и суммаризаций
            source_ids = set()
            for doc in quote_result + sums_result:
                if hasattr(doc, 'metadata'):
                    source_id = doc.metadata.get("source_id")
                    if isinstance(source_id, int):
                        source_ids.add(source_id)
            
            # Получаем полные тексты глав (если включено)
            chapters_texts = ""
            if settings.INCLUDE_FULL_CHAPTERS and source_ids:
                try:
                    from src.agent.vector_store.chapter_retriever import ChapterRetriever
                    
                    chapter_retriever = ChapterRetriever()
                    
                    # Ограничиваем количество глав
                    source_ids_list = sorted(list(source_ids))[:settings.MAX_CHAPTERS_IN_CONTEXT]
                    chapters_dict = chapter_retriever.get_chapters_by_source_ids(
                        source_ids_list,
                        max_chapters=settings.MAX_CHAPTERS_IN_CONTEXT
                    )
                    
                    # Форматируем тексты глав с учетом ограничений по токенам
                    chapters_texts = self._format_chapters_text(
                        chapters_dict,
                        max_tokens_per_chapter=settings.MAX_CHAPTER_TOKENS
                    )
                    
                except Exception as e:
                    logger.warning(f"Не удалось загрузить главы: {e}", exc_info=True)
                    chapters_texts = ""
            
            answer_input["chapters_texts"] = chapters_texts
            
            # Формируем финальный ответ
            try:
                answer_input_query = self.make_user_prompt(query, answer_input)
                result = self.make_llm_chain().invoke(answer_input_query)
                output = self._process_output(result)
                
                # Обрабатываем результат
                if isinstance(output, dict):
                    return output
                elif isinstance(output, str):
                    return {"answer": output}
                else:
                    logger.warning(f"Unexpected output type: {type(output)}")
                    return {"answer": str(output)}
                    
            except Exception as e:
                logger.error(f"Error in LLM chain: {e}", exc_info=True)
                return {"answer": "Ошибка при формировании ответа. Попробуйте переформулировать вопрос."}
                
        except ValueError as e:
            logger.error(f"Validation error in invoke: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in invoke: {e}", exc_info=True)
            return {"answer": "Произошла неожиданная ошибка при обработке запроса."}

    def make_user_prompt_retrieve(self, query: str) -> str:
        """
        Форматирует запрос для ретривера.
        
        Args:
            query: Поисковый запрос
            
        Returns:
            Отформатированный запрос
        """
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        return query

    def make_user_prompt(self, query: str, answer_input: Dict[str, str]) -> Dict[str, Any]:
        """
        Форматирует пользовательский промпт для генерации ответа.
        
        Args:
            query: Исходный вопрос пользователя
            answer_input: Словарь с обработанными данными из графа и поиска
            
        Returns:
            Словарь с сообщениями для LLM в формате {"messages": [("user", prompt)]}
        """
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        
        if not isinstance(answer_input, dict):
            answer_input = {}
        
        message = [f"**Исходный вопрос пользователя:** {query}"]
        
        if answer_input.get("nodes_texts"):
            message.extend([
                "==== Описание выбранных персонажей книги: ====",
                answer_input["nodes_texts"],
            ])

        if answer_input.get("rel_texts"):
            message.extend([
                "==== Описание взаимодействия персонажей книги: ====",
                answer_input["rel_texts"],
            ])

        if answer_input.get("sums_texts"):
            message.extend([
                "==== Суммаризация глав книги: ====",
                answer_input["sums_texts"]
            ])

        if answer_input.get("quote_texts"):
            message.extend([
                "==== Цитаты из книги: ====",
                answer_input["quote_texts"]
            ])

        if answer_input.get("chapters_texts"):
            message.extend([
                "==== Полные тексты глав книги: ====",
                answer_input["chapters_texts"]
            ])

        user_prompt = "\n\n".join(message)
        messages = {"messages": [("user", user_prompt)]}
        
        logger.debug(f"Formatted prompt for answer generation (query length: {len(query)})")
        return messages

    def _format_chapters_text(
        self, 
        chapters_dict: Dict[int, str],
        max_tokens_per_chapter: Optional[int] = None
    ) -> str:
        """
        Форматирует тексты глав для добавления в промпт.
        
        Args:
            chapters_dict: Словарь {source_id: chapter_text}
            max_tokens_per_chapter: Максимальное количество токенов на главу
            
        Returns:
            Отформатированная строка с текстами глав
        """
        if not chapters_dict:
            return ""
        
        # Импортируем функцию подсчета токенов
        from tools.preprocess_book.load_to_vectorstore.chunk_splitter import calculate_tokens
        
        formatted_chapters = []
        for source_id, chapter_text in sorted(chapters_dict.items()):
            # Обрезаем главу по токенам если слишком длинная
            text_to_add = chapter_text
            if max_tokens_per_chapter:
                chapter_tokens = calculate_tokens(chapter_text)
                if chapter_tokens > max_tokens_per_chapter:
                    # Обрезаем текст (приблизительно, сохраняя начало)
                    # Используем простую обрезку по символам (4 символа на токен)
                    max_chars = max_tokens_per_chapter * 4
                    text_to_add = chapter_text[:max_chars] + "\n[... текст обрезан ...]"
                    logger.debug(f"Глава {source_id} обрезана: {chapter_tokens} -> {max_tokens_per_chapter} токенов")
            
            formatted_chapters.append(
                f"==== Глава {source_id} ====\n{text_to_add}"
            )
        
        return "\n\n".join(formatted_chapters)


# Создаем парсер с Pydantic моделью для валидации
retrieve_parser = PydanticOutputParser(pydantic_object=RetrieveOutput)

# Обновляем промпт с инструкциями по форматированию
question_answer_prompt_with_format = question_answer_prompt.replace(
    "{{format_instructions}}",
    retrieve_parser.get_format_instructions()
)

retrieve_agent = RetrieveAgent(
    llm,
    question_answer_prompt_with_format,
    retriever=retriever,
    book_graph=book_graph,
    parser=retrieve_parser,
)
