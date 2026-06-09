"""
Tests for image preprocessing and OCR pipeline.
Run with: pytest tests/ -v
"""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.preprocessing import ImagePreprocessor


def make_test_image(text: str = "DOLIPRANE 500MG", size=(400, 200)) -> bytes:
    """Create a synthetic test image with text for testing."""
    try:
        import cv2
        img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        cv2.putText(img, text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()
    except Exception:
        # Minimal valid JPEG if cv2 fails
        import io
        from PIL import Image, ImageDraw
        img = Image.new("RGB", size, color="white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 80), text, fill="black")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()


class TestImagePreprocessor:
    @pytest.fixture
    def preprocessor(self):
        return ImagePreprocessor()

    @pytest.fixture
    def test_image_bytes(self):
        return make_test_image("DOLIPRANE 500MG 20 CP")

    def test_preprocess_returns_numpy_array(self, preprocessor, test_image_bytes):
        result = preprocessor.preprocess(test_image_bytes)
        assert isinstance(result, np.ndarray)

    def test_preprocess_returns_grayscale(self, preprocessor, test_image_bytes):
        result = preprocessor.preprocess(test_image_bytes)
        assert len(result.shape) == 2  # Grayscale = 2D

    def test_preprocess_multiple_views_returns_list(self, preprocessor, test_image_bytes):
        results = preprocessor.preprocess_multiple_views(test_image_bytes)
        assert isinstance(results, list)
        assert len(results) >= 2
        for name, img in results:
            assert isinstance(name, str)
            assert isinstance(img, np.ndarray)

    def test_preprocess_arabic_returns_array(self, preprocessor, test_image_bytes):
        result = preprocessor.preprocess_for_arabic(test_image_bytes)
        assert isinstance(result, np.ndarray)

    def test_handles_small_image(self, preprocessor):
        small_img = make_test_image(size=(100, 50))
        result = preprocessor.preprocess(small_img)
        assert isinstance(result, np.ndarray)
        # Should be upscaled
        assert result.shape[0] >= 50

    def test_handles_large_image(self, preprocessor):
        large_img = make_test_image(size=(5000, 3000))
        result = preprocessor.preprocess(large_img)
        assert isinstance(result, np.ndarray)
        # Should be downscaled
        assert max(result.shape) <= 3000


class TestOCRService:
    """
    OCR tests - these may be skipped if no OCR engine is installed.
    Install tesseract or easyocr to run these tests.
    """

    @pytest.fixture
    def ocr(self):
        from app.services.ocr_service import OCRService
        return OCRService()

    def test_ocr_service_initializes(self, ocr):
        assert ocr is not None

    def test_extract_text_returns_result(self, ocr):
        from app.services.ocr_service import OCRResult
        preprocessor = ImagePreprocessor()
        img_bytes = make_test_image("DOLIPRANE 500")
        img = preprocessor.preprocess(img_bytes)
        result = ocr.extract_text(img)
        assert isinstance(result, OCRResult)
        assert isinstance(result.raw_text, str)
        assert isinstance(result.tokens, list)

    def test_extract_text_empty_image(self, ocr):
        """Should not crash on blank image."""
        blank = np.ones((200, 400), dtype=np.uint8) * 255
        result = ocr.extract_text(blank)
        assert result is not None

    def test_clean_text_normalizes(self, ocr):
        raw = "  doli  prane   500mg  \n\r  "
        cleaned = ocr._clean_text(raw)
        assert "  " not in cleaned
        assert cleaned == cleaned.upper()

    def test_tokenize_splits_correctly(self, ocr):
        tokens = ocr._tokenize("DOLIPRANE 500MG 20 CP EFFER")
        assert "DOLIPRANE" in tokens
        assert "500MG" in tokens
