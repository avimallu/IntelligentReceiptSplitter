from src.llm.ollama import LLMExtractor
from datetime import date


def test_ollama_response():
    messages: list[Message] = [{"role": "user", "content": "Hello"}]
    output = get_chat_response(model="gemma2:27b", messages=messages)
    assert isinstance(output, str)


def test_ollama_structured_response():
    class ModelData(BaseModel):
        name: str
        knowledge_cutoff: date

    messages: list[Message] = [{"role": "user", "content": "Tell me about yourself"}]
    output = get_chat_response(
        model="gemma2:27b",
        messages=messages,
        structured_output_format=ModelData.model_json_schema(),
    )
    assert ModelData.model_validate(output)


def test_ollama_image_upload():
    messages: list[Message] = [
        {
            "role": "user",
            "content": "Run OCR on this image.",
            "images": ["./data/pytest/image_upload_test.png"],
        }
    ]
    output = get_chat_response(
        model="llava:34b",
        messages=messages,
    )
    print(output)
    assert False


def test_llm_extractor():
    llm_extractor = LLMExtractor(model="gemma2:27b")
    with open("data/pytest/receipt_ocr_test.txt", "r") as f:
        receipt_ocr_text = "".join(f.readlines())
    assert llm_extractor.extract_merchant_name(receipt_ocr_text) == 'Walmart'
    assert llm_extractor.extract_receipt_date(receipt_ocr_text) == date.fromisoformat('2017-07-28')
    assert llm_extractor.extract_receipt_total_amount(receipt_ocr_text) == {'currency': 'USD', 'amount': 98.21}
    assert llm_extractor.extract_receipt_tip_amount(receipt_ocr_text) == {'currency': 'USD', 'amount': 0}
    assert llm_extractor.extract_receipt_tax_amount(receipt_ocr_text) == {'currency': 'USD', 'amount': 4.59}
    items = llm_extractor.extract_receipt_items(receipt_ocr_text)
    # The exact items in these cannot be always identical as there will be some inherent variance in LLM output.
    # Thus, we check if a few control totals match, at least approximately.
    assert len(items) == 26
    assert round(sum(x["amount"] for x in items), 0) == 94
