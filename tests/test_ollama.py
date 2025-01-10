from datetime import date

import ollama
import pytest
from pydantic import BaseModel
from typing import Type, Any

from src.llm.extractor import LLMExtractor

from src.llm.models import (
    ItemizedAmounts,
    Message,
    ReceiptAmount,
    ReceiptDate,
    ReceiptExtracted,
    ReceiptItemized,
    ReceiptMerchant,
)


@pytest.fixture
def ollama_model_name(request):
    """Must be the name of an ollama model that has already been downloaded to the system."""
    param = request.config.getoption("--ollama-model-name")
    return param


def test_ollama_response(ollama_model_name: str):
    """Make sure that there is a response (and not an error from ollama)."""
    messages: list[Message] = [{"role": "user", "content": "Hello"}]
    llm_extractor = LLMExtractor(model=ollama_model_name)
    output = llm_extractor.get_chat_response(messages=messages)
    assert isinstance(output, str)


def test_ollama_invalid_model_name():
    messages: list[Message] = [{"role": "user", "content": "Hello"}]
    llm_extractor = LLMExtractor(model="ba-ba-black-sheep")
    with pytest.raises(ollama._types.ResponseError):
        llm_extractor.get_chat_response(messages=messages)


def test_ollama_structured_response(ollama_model_name):
    class ModelData(BaseModel):
        name: str
        knowledge_cutoff: date

    messages: list[Message] = [{"role": "user", "content": "Tell me about yourself"}]
    llm_extractor = LLMExtractor(model=ollama_model_name)
    output = llm_extractor.get_chat_response(
        messages=messages,
        structured_output_format=ModelData,
    )
    assert ModelData.model_validate(output)


@pytest.fixture
def llm_extractor(ollama_model_name) -> LLMExtractor:
    return LLMExtractor(model=ollama_model_name)


@pytest.fixture
def receipt_string() -> str:
    with open("assets/pytest/receipt_ocr_test.txt", "r") as f:
        receipt_string = "".join(f.readlines())
    return receipt_string


def receipt_test_cases():
    return [
        (ReceiptMerchant, "extract_merchant", "name", "walmart"),
        (ReceiptDate, "extract_receipt_date", "date", "2017-07-28"),
        (ReceiptAmount, "extract_total", "amount", 98.21),
        (ReceiptAmount, "extract_tip", "amount", 0),
        (ReceiptAmount, "extract_tax", "amount", 4.59),
    ]


@pytest.mark.parametrize("cls,prompt,field,expected", receipt_test_cases())
def test_extract_fields(
    cls: Type[BaseModel],
    prompt: str,
    field: str,
    expected: Any,
    receipt_string,
    llm_extractor,
    ollama_model_name,
):
    """
    Provides a slightly complex receipt to the LLM used to check if it is able to extract
    basic information in it. If it doesn't pass this, then you may need a different model.
    """
    result = llm_extractor.extract_fields(receipt_string, cls, prompt, field, None)
    if isinstance(expected, str):
        result = result.lower()
    assert result == expected


def test_extract_items(llm_extractor, receipt_string):
    """
    Provides a slightly complex receipt to the LLM used to check if it is able to extract
    the list of items in it. If it doesn't pass this, then you may need a different model.

    The exact items cannot be always identical as there will be some inherent variance in LLM output.
    Therefore, check if a few control totals match, at least approximately.
    """
    items = llm_extractor.extract_fields(
        receipt_string, ReceiptItemized, "extract_receipt_items", "ItemizedReceipt", []
    )
    assert 22 <= len(items) <= 26
    assert 85 <= round(sum(x["amount"] for x in items), 0) <= 95
