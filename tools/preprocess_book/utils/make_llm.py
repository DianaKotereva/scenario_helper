from src.llm_core.llm_prompt_base import LLMBase
from src.utils.graph_search import BookNode
from typing import Tuple


class ExtractNames(LLMBase):
    def make_user_prompt(self, text: str) -> dict:
        user_prompt = f"Текст: {text}"
        return {"messages": [("user", user_prompt)]}


class Verification(LLMBase):
    def make_user_prompt(
        self, node: BookNode, new_node: dict, source_id: Tuple[int], last_n: int = -10
    ) -> dict:
        main_name = node.main_name
        alt_names = node.alt_names
        classification = node.classification
        all_prev_actions = ". ".join([i.action for i in node.actions][last_n:])
        user_prompt = [
            "**Существующее**: {{",
            f'"main_name": {main_name},',
            f'"alt_names": {alt_names}',
            f'"classification": {classification}',
            f'"actions": {all_prev_actions}',
            "}}",
            "\n\n\n",
            "**Новое**: {{",
            f'"main_name": {new_node["main_name"]}, ',
            f'"alt_names": {new_node["alt_names"]}',
            f'"classification": {new_node["classification"]}',
            f'"actions": {new_node["actions"]}',
            "}}",
        ]
        user_prompt = "\n".join(user_prompt)
        messages = {"messages": [("user", user_prompt)]}
        return messages
