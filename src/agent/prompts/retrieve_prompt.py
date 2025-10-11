from typing import Any, Dict, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from src.agent.vector_store.retriever import retriever
from src.llm_core.llm_core import llm
from src.llm_core.llm_prompt_base import LLMBase
from src.utils.graph_search import BookGraph, book_graph

question_answer_prompt = """Ты — эксперт-аналитик, формирующий детальный и подробный ответ на вопрос пользователя по книге. 

# ЗАДАЧА:
Дать ответ ИСКЛЮЧИТЕЛЬНО на основе:
1. Исходный вопрос пользователя: str //Исходный вопрос, который задал пользователь
3. Собранный контекст: List[str] // Уже имеющийся у тебя контекст. Контекст состоит из:
    - Цитаты из книги
    - Суммаризации глав
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
{{
  "answer": str
}}

Сформируй ответ:"""


class RetrieveAgent(LLMBase):
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
        Args:
            llm: Initialized language model
            system_prompt: Base system prompt template
        """
        self.llm = llm
        self._system_prompt = system_prompt
        self._retriever = retriever
        self._prompt_template: Optional[ChatPromptTemplate] = None
        self._parser = parser
        self._book_graph = book_graph
        self._load_template = load_template

    # def make_llm_chain(self) -> Runnable:
    #     """Create runnable LLM chain with prompt template"""
    #     question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
    #     chain = create_retrieval_chain(self._retriever, question_answer_chain)
    #     return chain

    def process_nodes(self, nodes_names):
        all_node_texts = []
        for node_name in nodes_names:
            node = self._book_graph.nodes.nodes[node_name]
            node_text = (
                f"**Действия {node_name}**"
                + "\n"
                + "\n".join([i.action for i in node.actions])
                + "****"
            )
            all_node_texts.append(node_text)
        all_node_texts = "\n\n".join(all_node_texts)
        return all_node_texts

    def process_rels(self, rel_names):
        all_rels_texts = []
        for rel_name in rel_names:
            rel = self._book_graph.relationships.relationships[rel_name]
            object_1 = rel.object_1
            object_2 = rel.object_2
            desc = "\n".join([i.description for i in rel.description]) + "****"
            rel_text = f"**Взаимодействие между {object_1} и {object_2}**" + "\n" + desc
            all_rels_texts.append(rel_text)
        all_rels_texts = "\n\n".join(all_rels_texts)
        return all_rels_texts

    def process_graph(self, rels_result, nodes_result):
        nodes_names = set([i.metadata["name"] for i in nodes_result])
        rel_names = set(
            [
                i.metadata["name"]
                if isinstance(i.metadata["name"], tuple)
                else (i.metadata["name"][0], i.metadata["name"][1])
                for i in rels_result
            ]
        )
        for rel_name in rel_names:
            name_1 = rel_name[0]
            name_2 = rel_name[1]
            nodes_names.add(name_1)
            nodes_names.add(name_2)

        node_text = self.process_nodes(nodes_names)
        rels_text = self.process_rels(rel_names)

        return {"nodes_texts": node_text, "rel_texts": rels_text}

    def invoke(self, **kwargs: Dict[str, Any]) -> str:
        """Execute LLM chain synchronously"""

        chain_input = self.make_user_prompt_retrieve(**kwargs)
        search_result = self._retriever.invoke(chain_input)
        quote_result = [i for i in search_result if i.metadata["source"] == "book"]
        nodes_result = [i for i in search_result if i.metadata["source"] == "nodes"]
        rels_result = [i for i in search_result if i.metadata["source"] == "relations"]
        sums_result = [i for i in search_result if i.metadata["source"] == "summary"]
        answer_input = self.process_graph(rels_result, nodes_result)

        quote_texts = "\n***\n".join(
            [f"{i.page_content}. Глава {i.metadata['source_id']}" for i in quote_result]
        )
        sums_texts = "\n***\n".join(
            [f"{i.page_content}. Глава {i.metadata['source_id']}" for i in sums_result]
        )
        answer_input["quote_texts"] = quote_texts
        answer_input["sums_texts"] = sums_texts

        # result = self.make_llm_chain().invoke(chain_input)
        answer_input_query = self.make_user_prompt(chain_input, answer_input)
        result = self.make_llm_chain().invoke(answer_input_query)
        return self._process_output(result)

    def make_user_prompt_retrieve(self, query):
        return query

    def make_user_prompt(self, query: str, answer_input: dict):
        message = [f"**Исходный вопрос пользователя:** {query}"]
        if answer_input.get("nodes_texts", ""):
            message.extend(
                [
                    "==== Описание выбранных персонажей книги: ====",
                    answer_input["nodes_texts"],
                ]
            )

        if answer_input.get("rel_texts", ""):
            message.extend(
                [
                    "==== Описание взаимодействия персонажей книги: ====",
                    answer_input["rel_texts"],
                ]
            )

        if answer_input.get("sums_texts", ""):
            message.extend(
                ["==== Суммаризация глав книги: ====", answer_input["sums_texts"]]
            )

        if answer_input.get("quote_texts", ""):
            message.extend(["==== Цитаты из книги: ====", answer_input["quote_texts"]])

        user_prompt = "\n\n".join(message)
        messages = {"messages": [("user", user_prompt)]}

        return messages


retrieve_agent = RetrieveAgent(
    llm,
    question_answer_prompt,
    retriever=retriever,
    book_graph=book_graph,
    parser=JsonOutputParser(),
)
