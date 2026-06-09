"""
Integration tests for the full recognition pipeline.
Run with: pytest tests/test_integration.py -v
"""
import pytest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_test_image(text: str = "DOLIPRANE 500MG", size=(400, 200)) -> bytes:
    try:
        import cv2
        img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
        cv2.putText(img, text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()
    except Exception:
        import io
        from PIL import Image, ImageDraw
        img = Image.new("RGB", size, color="white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 80), text, fill="black")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()


class TestRecognitionPipeline:
    @pytest.fixture(scope="class")
    def service(self):
        from app.services.recognition_service import RecognitionService
        return RecognitionService()

    def test_pipeline_initializes(self, service):
        assert service.preprocessor is not None
        assert service.ocr is not None
        assert service.db is not None

    def test_recognize_returns_response(self, service):
        from app.models.schemas import RecognitionResponse
        img = make_test_image("DOLIPRANE 500MG")
        result = service.recognize(img)
        assert isinstance(result, RecognitionResponse)
        assert result.success is True

    def test_recognize_returns_matches(self, service):
        img = make_test_image("DOLIPRANE 500MG 20 CP")
        result = service.recognize(img, top_n=5, threshold=40)
        # Should find at least the right medication
        assert isinstance(result.matches, list)

    def test_processing_time_tracked(self, service):
        img = make_test_image()
        result = service.recognize(img)
        assert result.processing_time_seconds > 0

    def test_text_extraction_only(self, service):
        from app.models.schemas import TextExtractionResponse
        img = make_test_image()
        result = service.extract_text_only(img)
        assert isinstance(result, TextExtractionResponse)
        assert result.success is True

    def test_search_database(self, service):
        results = service.search_database("DOLIPRANE", top_n=5)
        assert isinstance(results, list)
        if results:
            assert hasattr(results[0], "id")
            assert hasattr(results[0], "name")

    def test_list_medications_pagination(self, service):
        result = service.list_medications(page=1, page_size=10)
        assert "total" in result
        assert "items" in result
        assert len(result["items"]) <= 10

    def test_get_by_existing_id(self, service):
        result = service.get_by_id(59926)
        assert result is not None

    def test_get_by_nonexistent_id(self, service):
        result = service.get_by_id(999999)
        assert result is None

    def test_stats_endpoint(self, service):
        stats = service.get_stats()
        assert "total_medications" in stats
        assert stats["total_medications"] == 5031


class TestAPIEndpoints:
    """Test the FastAPI endpoints using TestClient."""

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "version" in resp.json()

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_stats_endpoint(self, client):
        resp = client.get("/api/v1/stats")
        assert resp.status_code == 200
        assert resp.json()["total_medications"] == 5031

    def test_list_medications(self, client):
        resp = client.get("/api/v1/medications?page=1&page_size=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5031
        assert len(data["items"]) == 5

    def test_search_endpoint(self, client):
        resp = client.get("/api/v1/search?q=DOLIPRANE&top_n=3")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    def test_recognize_endpoint_with_image(self, client):
        img_bytes = make_test_image("DOLIPRANE 500MG")
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        resp = client.post("/api/v1/recognize", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "matches" in data

    def test_recognize_invalid_format(self, client):
        files = {"file": ("test.pdf", b"fake pdf content", "application/pdf")}
        resp = client.post("/api/v1/recognize", files=files)
        assert resp.status_code == 400

    def test_medication_not_found(self, client):
        resp = client.get("/api/v1/medications/999999")
        assert resp.status_code == 404
