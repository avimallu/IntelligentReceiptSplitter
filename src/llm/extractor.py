import json
from pathlib import Path
from typing import Any, Callable, Literal, Type, overload

import yaml
from ollama import chat
from pydantic import BaseModel

from src.llm.models import (
    ItemizedAmounts,
    Message,
    ReceiptAmount,
    ReceiptDate,
    ReceiptExtracted,
    ReceiptItemized,
    ReceiptMerchant,
)


class LLMExtractor:
    def __init__(
        self,
        model: str,
        prompt_path: str | Path = "./src/llm/prompts.yaml",
        chat_function: Callable = chat,
    ):
        """

        Args:
            model: A model name that has been downloaded by ``ollama``.
            prompt_path: A str or Path to a YAML file with various prompts.
            chat_function: Typically ollama.chat, input not required.
        """
        self.model = model
        self.prompt_path = Path(prompt_path)
        self.chat_function = chat_function
        with open(self.prompt_path, "r") as f:
            self.prompts = yaml.safe_load(f)

    def get_chat_response(
        self,
        messages: list[Message],
        structured_output_format: Type[BaseModel] | None = None,
    ) -> Any:
        """

        Args:
            messages: The conversation so far.
            structured_output_format: The dictionary format of the pydantic model schema. Can be ``None``.

        Returns:
            Depends on the model chosen, and if `structured_output_format` is provided.
        """
        response = self.chat_function(
            model=self.model,
            messages=messages,
            format=structured_output_format.model_json_schema()
            if structured_output_format is not None
            else None,
            options={"temperature": 0},
        )
        message = response.message.content
        if structured_output_format is not None:
            try:
                return json.loads(message)
            except json.JSONDecodeError:
                return None
        else:
            return message

    def load_substituted_prompt(self, prompt_name: str, **kwargs) -> str:
        prompt = self.prompts[prompt_name]
        for patt, repl in kwargs.items():
            prompt = prompt.replace(f"[[ {patt} ]]", repl)
        return prompt

    @overload
    def extract_fields(
        self,
        receipt_str: str,
        structured_response_model: Type[BaseModel],
        prompt_name: Literal[
            "merchant",
            "receipt_date",
        ],
        key_to_extract: str | None = None,
    ) -> str: ...
    @overload
    def extract_fields(
        self,
        receipt_str: str,
        structured_response_model: Type[BaseModel],
        prompt_name: Literal[
            "total",
            "tip",
            "tax",
        ],
        key_to_extract: str | None = None,
        default_value: Any = None,
    ) -> float: ...
    @overload
    def extract_fields(
        self,
        receipt_str: str,
        structured_response_model: Type[BaseModel],
        prompt_name: Literal["receipt_items"],
        key_to_extract: str | None = None,
        default_value: Any = None,
    ) -> list[ItemizedAmounts]: ...
    def extract_fields(
        self,
        receipt_str: str,
        structured_response_model: Type[BaseModel],
        prompt_name: Literal[
            "merchant",
            "receipt_date",
            "total",
            "tip",
            "tax",
            "receipt_items",
        ],
        key_to_extract: str | None = None,
        default_value: Any = None,
    ) -> str | float | list[ItemizedAmounts] | Any:
        messages: list[Message] = [
            {
                "role": "system",
                "content": self.load_substituted_prompt(
                    prompt_name, **{"receipt_string": receipt_str}
                ),
            }
        ]
        output = self.get_chat_response(messages, structured_response_model)
        if output is None:
            return default_value
        if key_to_extract is not None:
            output = output[key_to_extract]
        return output

    def forward(self, receipt_string: str) -> ReceiptExtracted:
        return {
            field: self.extract_fields(
                receipt_string,
                structured_output_model,
                f"extract_{field}",
                key_to_extract,
                default_value,
            )
            for field, structured_output_model, key_to_extract, default_value in [
                ("merchant", ReceiptMerchant, "name", None),
                ("receipt_date", ReceiptDate, "date", None),
                ("total", ReceiptAmount, "amount", None),
                ("tip", ReceiptAmount, "amount", None),
                ("tax", ReceiptAmount, "amount", None),
                ("receipt_items", ReceiptItemized, "ItemizedReceipt", []),
            ]
        }
