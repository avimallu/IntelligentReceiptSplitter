from src.ocr.surya import SuryaOCR
from PIL.Image import open as pil_open


def test_ocr_sample_image():
    image_path = pil_open("assets/pytest/OCR_test.png")
    ocr = SuryaOCR()
    output = ocr.ordered_ocr_text(image_path)
    assert output == "This is Line 1\nThis is Line 2\nThis is Line 5"
