from datetime import date

import ollama
import pytest
from pydantic import BaseModel

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


def test_llm_extractor(ollama_model_name):
    """
    Provides a slightly complex receipt to the LLM used to check if it is able to extract
    basic information in it. If it doesn't pass this, then you may need a different model.
    """
    llm_extractor = LLMExtractor(model=ollama_model_name)
    with open("data/pytest/receipt_ocr_test.txt", "r") as f:
        receipt_ocr_text = "".join(f.readlines())
    assert llm_extractor.extract_fields("merchant", ReceiptMerchant, "name", None) == "Walmart"
    # assert llm_extractor.extract_merchant_name(receipt_ocr_text) == "Walmart"
    # assert llm_extractor.extract_receipt_date(receipt_ocr_text) == date.fromisoformat(
    #     "2017-07-28"
    # )
    # assert llm_extractor.extract_receipt_total_amount(receipt_ocr_text) == {
    #     "currency": "USD",
    #     "amount": 98.21,
    # }
    # assert llm_extractor.extract_receipt_tip_amount(receipt_ocr_text) == {
    #     "currency": "USD",
    #     "amount": 0,
    # }
    # assert llm_extractor.extract_receipt_tax_amount(receipt_ocr_text) == {
    #     "currency": "USD",
    #     "amount": 4.59,
    # }
    # items = llm_extractor.extract_receipt_items(receipt_ocr_text)
    # # The exact items in these cannot be always identical as there will be some inherent variance in LLM output.
    # # Thus, we check if a few control totals match, at least approximately.
    # assert len(items) == 26
    # assert round(sum(x["amount"] for x in items), 0) == 94
