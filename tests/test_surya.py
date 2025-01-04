from src.ocr.surya import SuryaOCR


def test_ocr_sample_image():
    image_path = "data/pytest/image_upload_test.png"
    output = ocr_image(image_path=image_path)
    assert output is not None
