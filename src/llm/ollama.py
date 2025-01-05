import json
import yaml
import gradio as gr
from ollama import chat
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from typing import TypedDict, Optional, Literal, Any, Type, Annotated


class Message(TypedDict):
    role: Literal["user", "assistant", "system", "tool"]
    content: Optional[str]
    images: Optional[list[str] | bytes | list[Path]]


class ReceiptMerchant(BaseModel):
    name: str

class ReceiptDate(BaseModel):
    date: datetime

def is_valid_currency_code(value: str) -> bool:
    return len(value) == 3 and value.isupper() and value.isalpha()

class ReceiptAmount(BaseModel):
    currency: Annotated[str, is_valid_currency_code]
    amount: float

class Amount(TypedDict):
    currency: Annotated[str, is_valid_currency_code]
    amount: float

class ReceiptItemized(BaseModel):
    class ReceiptLineItemAmount(BaseModel):
        name: str
        currency: Annotated[str, is_valid_currency_code]
        amount: float
    ItemizedReceipt: list[ReceiptLineItemAmount]

class ItemizedAmounts(TypedDict):
    name: str
    currency: Annotated[str, is_valid_currency_code]
    amount: float

class ReceiptExtracted(TypedDict):
    merchant: str
    receipt_date: datetime
    total: Amount
    tip: Amount
    tax: Amount
    item_amounts: list[ItemizedAmounts]

class LLMExtractor:
    def __init__(self, model: str, prompt_path: str | Path = "./src/llm/prompts.yaml"):
        """

        Args:
            model: A model name that has been downloaded by ``ollama``.
            prompt_path: A str or Path to a YAML file with various prompts.
        """
        self.model = model
        self.prompt_path = Path(prompt_path)
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
        response = chat(
            model=self.model, messages=messages, format=structured_output_format.model_json_schema(),
            options={"temperature": 0}
        )
        message = response.message.content
        if structured_output_format is not None:
            return json.loads(message)
        else:
            return message

    def load_substituted_prompt(self, prompt_name: str, **kwargs) -> str:
        prompt = self.prompts[prompt_name]
        for patt, repl in kwargs.items():
            prompt = prompt.replace(f"[[ {patt} ]]", repl)
        return prompt

    def extract_merchant_name(self, receipt_string: str) -> str:
        messages: list[Message] = [
            {
                "role": "system",
                "content": self.load_substituted_prompt(
                    "extract_merchant_name", **{"receipt_string": receipt_string}
                ),
            },
        ]
        return self.get_chat_response(messages, ReceiptMerchant)['name']

    def extract_receipt_date(self, receipt_string: str) -> datetime:
        messages: list[Message] = [
            {
                "role": "system",
                "content": self.load_substituted_prompt(
                    "extract_receipt_date", **{"receipt_string": receipt_string}
                ),
            },
        ]
        receipt_date = self.get_chat_response(messages, ReceiptDate)['date']
        return datetime.fromisoformat(receipt_date)

    def extract_receipt_total_amount(self, receipt_string: str) -> Amount:
        messages: list[Message] = [
            {
                "role": "system",
                "content": self.load_substituted_prompt(
                    "extract_receipt_total_amount", **{"receipt_string": receipt_string}
                ),
            },
        ]
        currency_amount = self.get_chat_response(messages, ReceiptAmount)
        return currency_amount

    def extract_receipt_tax_amount(self, receipt_string: str) -> Amount:
        messages: list[Message] = [
            {
                "role": "system",
                "content": self.load_substituted_prompt(
                    "extract_receipt_tax_amount", **{"receipt_string": receipt_string}
                ),
            },
        ]
        currency_amount = self.get_chat_response(messages, ReceiptAmount)
        return currency_amount

    def extract_receipt_tip_amount(self, receipt_string: str) -> Amount:
        messages: list[Message] = [
            {
                "role": "system",
                "content": self.load_substituted_prompt(
                    "extract_receipt_tip_amount", **{"receipt_string": receipt_string}
                ),
            },
        ]
        currency_amount = self.get_chat_response(messages, ReceiptAmount)
        return currency_amount

    def extract_receipt_items(self, receipt_string: str) -> list[ItemizedAmounts]:
        messages: list[Message] = [
            {
                "role": "system",
                "content": self.load_substituted_prompt(
                    "extract_receipt_items", **{"receipt_string": receipt_string}
                ),
            },
        ]
        itemized_amounts = self.get_chat_response(messages, ReceiptItemized)['ItemizedReceipt']
        return itemized_amounts

    def forward(self, receipt_string) -> ReceiptExtracted:
        merchant = self.extract_merchant_name(receipt_string)
        receipt_date = self.extract_receipt_date(receipt_string)
        total_amount = self.extract_receipt_total_amount(receipt_string)
        tip_amount = self.extract_receipt_tip_amount(receipt_string)
        tax_amount = self.extract_receipt_tax_amount(receipt_string)
        item_amounts = self.extract_receipt_items(receipt_string)
        return {
            "merchant": merchant,
            "receipt_date": receipt_date,
            "total": total_amount,
            "tip": tip_amount,
            "tax": tax_amount,
            "item_amounts": item_amounts
        }
