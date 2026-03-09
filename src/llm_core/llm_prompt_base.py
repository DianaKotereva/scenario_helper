import asyncio
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import json_repair
from pydantic import BaseModel
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable


class LLMBase(ABC):
    """Abstract base class for LLM-powered agents with sync/async support."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        system_prompt: str,
        parser: Optional[BaseOutputParser] = None,
        parse_json: bool = False,
        load_template: bool = False,
    ):
        """
        Args:
            llm: Initialized language model
            system_prompt: Base system prompt template
        """
        self.llm = llm
        self._system_prompt = system_prompt
        self._prompt_template: Optional[ChatPromptTemplate] = None
        self._parser = parser
        self._parse_json = parse_json
        self._load_template = load_template

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        """Cached prompt template property"""
        if self._prompt_template is None:
            self._prompt_template = self._create_prompt_template()
        return self._prompt_template

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Construct prompt template from system message"""
        if self._load_template:
            with open(self._system_prompt, "r", encoding="utf-8") as file:
                self._system_prompt = file.read()
            self._load_template = False

        return ChatPromptTemplate.from_messages(
            [SystemMessage(content=self._system_prompt), ("placeholder", "{messages}")]
        )

    def make_llm_chain(self) -> Runnable:
        """Create runnable LLM chain with prompt template"""
        if self._parser:
            return self.prompt_template | self.llm | self._parser
        else:
            return self.prompt_template | self.llm

    @abstractmethod
    def make_user_prompt(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Abstract method to format user input for LLM"""
        raise NotImplementedError

    def invoke(self, **kwargs: Dict[str, Any]) -> str:
        """Execute LLM chain synchronously"""
        chain_input = self.make_user_prompt(**kwargs)
        result = self.make_llm_chain().invoke(chain_input)
        return self._process_output(result)

    async def ainvoke(self, **kwargs: Dict[str, Any]) -> str:
        """Execute LLM chain asynchronously"""
        chain_input = self.make_user_prompt(**kwargs)
        result = await self.make_llm_chain().ainvoke(chain_input)
        return self._process_output(result)

    async def _process_one(
        self, input_data: dict, semaphore: asyncio.Semaphore
    ) -> AIMessage:
        try:
            async with semaphore:
                res = await self.make_llm_chain().ainvoke(input_data)
                return self._process_output(res)
        except Exception:
            return AIMessage(content="")

    async def batch(
        self, inputs: list, concurrency: int = None, semaphore: asyncio.Semaphore = None
    ) -> list:
        if not semaphore:
            if concurrency:
                semaphore = asyncio.Semaphore(concurrency)
            else:
                raise ValueError("Нужно задать либо количество потоков, либо семафор!")
        tasks = [self._process_one(inp, semaphore) for inp in inputs]
        res = await asyncio.gather(*tasks)
        return res

    def parse_json_content(self, string):
        string_re = re.sub(
            " +",
            " ",
            string[string.find("{") : string.rfind("}") + 1].replace("\n", ""),
        )
        string_re = (
            string_re.replace("{ ", "{")
            .replace(" }", "}")
            .replace("{\n", "{")
            .replace("\n}", "}")
            .replace("false", "False")
            .replace("true", "True")
        )  # noqa: E501

        string_re = string_re.replace("{'", '{"').replace("'}", '"}')
        return json_repair.loads(string_re)

    def _process_output(self, output: Any) -> Any:
        """
        Uniform output processing.
        
        Если используется PydanticOutputParser, результат уже будет Pydantic моделью.
        Если используется JsonOutputParser, результат будет словарем.
        Если парсера нет, возвращаем строку.
        """
        # Если используется PydanticOutputParser, результат уже валидирован
        if isinstance(output, BaseModel):
            return output
        
        # Если парсер уже обработал вывод, возвращаем как есть
        if isinstance(output, dict):
            return output
        
        if hasattr(output, "content"):
            if self._parse_json:
                try:
                    res = self.parse_json_content(output.content)
                except:  # noqa: E722
                    res = output.content
                return res
            return output.content
        return output

    def __call__(self, **kwargs: Dict[str, Any]) -> Union[str, Any]:
        """Alias for invoke"""
        return self.invoke(**kwargs)
