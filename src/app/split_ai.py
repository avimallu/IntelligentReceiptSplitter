from src.llm.ollama import LLMExtractor, ReceiptExtracted
from src.ocr.surya import SuryaOCR
from PIL.Image import Image


class ReceiptReader:
    def __init__(self, llm_model: str):
        self.ocr_model = SuryaOCR()
        self.llm_model = LLMExtractor(model=llm_model)

    def get_ordered_text(self, image: Image) -> str:
        return self.ocr_model.ordered_ocr_text(image)

    def extract_components(self, receipt_string: str) -> ReceiptExtracted:
        return self.llm_model.forward(receipt_string)
