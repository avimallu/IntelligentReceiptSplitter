from pathlib import Path
from PIL.Image import open as PIL_open
from PIL.Image import Image
from surya.ocr import run_ocr
from surya.model.detection.model import (
    load_model as load_det_model,
    load_processor as load_det_processor,
)
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from typing import Literal, TypedDict, overload, Optional
from collections import defaultdict
from more_itertools import bucket

Polygon = list[list[float]]
BBox = list[float]


class SuryaOCRResult(TypedDict):
    polygon: Polygon
    confidence: Optional[float]
    text: str
    bbox: BBox


class SplitAIOCRResult(TypedDict):
    polygons: list[Polygon]
    texts: list[str]
    bboxes: list[BBox]


class SuryaOCR:
    def __init__(self):
        self.det_processor = load_det_processor()
        self.det_model = load_det_model()
        self.rec_model = load_rec_model()
        self.rec_processor = load_rec_processor()

    @overload
    def ocr_image(
        self, image: Image, return_format: Literal["polygon"]
    ) -> list[Polygon]: ...
    @overload
    def ocr_image(self, image: Image, return_format: Literal["text"]) -> list[str]: ...
    @overload
    def ocr_image(self, image: Image, return_format: Literal["bbox"]) -> list[BBox]: ...
    @overload
    def ocr_image(
        self, image: Image, return_format: Literal["confidence"]
    ) -> list[BBox]: ...
    @overload
    def ocr_image(self, image: Image, return_format: None) -> SplitAIOCRResult: ...
    def ocr_image(
        self,
        image: Image,
        return_format: Literal["polygon", "text", "bbox", "confidence"] | None,
    ) -> SplitAIOCRResult | list[Polygon] | list[str] | list[BBox]:
        """
        Specify either a path to an image or image file itself (as a PIL image) to run OCR on.
        If both are provided, then the image file is prioritized.

        Args:
            image: the PIL image for Surya to process
            return_format: Specify one of the allowed values to return only this key from SuryaOCR's output.

        Returns:
            OCRResult, in Surya's format. See `<https://github.com/VikParuchuri/surya>` for details.
        """
        images = [image]
        langs = [["en"]]

        ocr_output = run_ocr(
            images,
            langs,
            self.det_model,
            self.det_processor,
            self.rec_model,
            self.rec_processor,
        )
        ocr_as_list: list[SuryaOCRResult] = ocr_output[0].model_dump()["text_lines"]
        if isinstance(return_format, str):
            return [x[return_format] for x in ocr_as_list]
        else:
            polygons = [x["polygon"] for x in ocr_as_list]
            texts = [x["text"] for x in ocr_as_list]
            bboxes = [x["bbox"] for x in ocr_as_list]
            return {
                "polygons": polygons,
                "texts": texts,
                "bboxes": bboxes,
            }

    def ordered_ocr_text(self, image: Image) -> str:
        split_ocr_result = self.ocr_image(image, None)
        ordered_text: list[str] = []
        for group in SuryaOCR.order_polygons(split_ocr_result["polygons"]):
            line = [split_ocr_result["texts"][x] for x in group]
            ordered_text += [" ".join(line)]
        return "\n".join(ordered_text)

    @staticmethod
    def get_centroid_of_bounding_polygon(
        polygon: Polygon,
    ) -> tuple[float, float]:
        """
        Function name is self-explanatory. Invariant of rotation. Any bounding polygon whose centroid differs
        significantly to another one is unlikely to be in the same line.

        Args:
            polygon: `[[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]]` coordinates of
                     the polygon, clockwise from top left.

        Returns:
            The centroid of the bounding polygon.
        """
        (x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4) = tuple(
            tuple(x) for x in polygon
        )

        return (x_1 + x_4) / 2, (y_1 + y_2) / 2

    @staticmethod
    def get_height_of_bounding_polygon(polygon: Polygon) -> float:
        """

        Args:
            polygon: `[[x_1, y_1], [x_2, y_2], [x_3, y_3], [x_4, y_4]]` coordinates of
                     the polygon, clockwise from top left.
        Returns:
            The height of the bounding polygon.
        """
        (x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4) = tuple(
            tuple(x) for x in polygon
        )

        return y_4 - y_1

    @staticmethod
    def order_polygons(polygons: list[Polygon]) -> list[list[int]]:
        """
        Given a list of bounding polygons (from an OCR framework) for recognized text, attempt
        to reorder them (left-to-right, top-to-bottom) that their text contents are expected to
        be read.

        This is intended to work independent of the orientation of the receipts, but currently
        is at a PoC stage where it is assumed that the receipt is horizontally positioned.

        Args:
            polygons: Bounding polygons of recognized text.

        Returns:
            A ``list[int]`` with the estimated line numbers.
        """
        x_midpoints, y_midpoints = zip(
            *[SuryaOCR.get_centroid_of_bounding_polygon(x) for x in polygons]
        )
        heights = [SuryaOCR.get_height_of_bounding_polygon(x) for x in polygons]
        threshold = 0.6
        y_ranges = [
            (
                midpoint - height * threshold * 0.5,
                midpoint,
                midpoint + height * threshold * 0.5,
            )
            for midpoint, height in zip(y_midpoints, heights)
        ]

        # Assign line groups
        line_groups: dict[int, int | None] = defaultdict(lambda: None)
        for idx, i_range in enumerate(y_ranges):
            if idx not in line_groups:
                line_groups[idx] = idx
            for jdx, j_range in [
                (jdx, j_range) for jdx, j_range in enumerate(y_ranges) if jdx > idx
            ]:
                if (i_range[0] <= j_range[1] <= i_range[2]) and jdx not in line_groups:
                    line_groups[jdx] = idx

        # Reorder by x_midpoints within a group:
        line_groups_reversed = [(group, key) for key, group in line_groups.items()]
        bucketed_line_groups = bucket(line_groups_reversed, key=lambda x: x[0])
        ordered_text = []
        for key in sorted(list(bucketed_line_groups)):
            idx_group = list(bucketed_line_groups[key])
            sorted_list = sorted(idx_group, key=lambda x: x_midpoints[x[1]])
            sorted_idx = [x[1] for x in sorted_list]
            ordered_text += [sorted_idx]

        return ordered_text
