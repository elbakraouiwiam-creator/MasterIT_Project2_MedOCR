"""
Recognition Service - Orchestrates the full pipeline:
  Image → Preprocessing → OCR → Matching → Results
"""
import logging
import time
from typing import List, Optional

from app.services.preprocessing import ImagePreprocessor
from app.services.ocr_service import OCRService
from app.services.database import MedicationDatabase, MatchResult
from app.models.schemas import (
    RecognitionResponse, MatchedMedication,
    TextExtractionResponse, SearchResponse, SearchMatch
)
from app.config import settings

logger = logging.getLogger(__name__)


class RecognitionService:
    """
    Orchestrates the complete medication recognition pipeline.
    """

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.ocr = OCRService()
        self.db = MedicationDatabase(settings.PRODUCTS_DB_PATH)

    def recognize(
        self,
        image_bytes: bytes,
        top_n: int = 5,
        threshold: int = 60
    ) -> RecognitionResponse:
        """
        Full pipeline: image bytes → recognized medications.
        """
        start_time = time.time()

        # Step 1: Preprocess - get multiple versions
        preprocessed_versions = self.preprocessor.preprocess_multiple_views(image_bytes)

        # Step 2: OCR on all versions, merge results
        ocr_result = self.ocr.extract_text_multi(preprocessed_versions)

        # Step 3: Match against database
        # Try full text match first
        matches = self.db.search(
            ocr_result.cleaned_text,
            top_n=top_n,
            threshold=threshold
        )

        # If not enough results, try token-based matching
        if len(matches) < top_n and ocr_result.tokens:
            token_matches = self.db.search_by_tokens(
                ocr_result.tokens,
                top_n=top_n,
                threshold=threshold
            )
            # Merge, avoiding duplicates
            existing_ids = {m.id for m in matches}
            for tm in token_matches:
                if tm.id not in existing_ids:
                    matches.append(tm)
                    existing_ids.add(tm.id)

        matches = sorted(matches, key=lambda x: x.score, reverse=True)[:top_n]

        processing_time = round(time.time() - start_time, 3)

        return RecognitionResponse(
            success=True,
            extracted_text=ocr_result.cleaned_text,
            raw_ocr_text=ocr_result.raw_text,
            ocr_confidence=round(ocr_result.confidence, 3),
            ocr_engine=ocr_result.engine,
            matches=[
                MatchedMedication(
                    id=m.id,
                    name=m.name,
                    confidence=round(m.score / 100, 3),
                    match_score=round(m.score, 1),
                    match_type=m.match_type
                )
                for m in matches
            ],
            total_matches=len(matches),
            processing_time_seconds=processing_time
        )

    def extract_text_only(self, image_bytes: bytes) -> TextExtractionResponse:
        """OCR only - no database matching."""
        start_time = time.time()
        preprocessed = self.preprocessor.preprocess(image_bytes)
        ocr_result = self.ocr.extract_text(preprocessed)
        processing_time = round(time.time() - start_time, 3)

        return TextExtractionResponse(
            success=True,
            raw_text=ocr_result.raw_text,
            cleaned_text=ocr_result.cleaned_text,
            tokens=ocr_result.tokens,
            confidence=round(ocr_result.confidence, 3),
            engine=ocr_result.engine,
            processing_time_seconds=processing_time
        )

    def search_database(self, query: str, top_n: int = 10, threshold: int = 50) -> List[SearchMatch]:
        results = self.db.search(query, top_n=top_n, threshold=threshold)
        return [
            SearchMatch(id=r.id, name=r.name, score=round(r.score, 1), match_type=r.match_type)
            for r in results
        ]

    def list_medications(self, page: int = 1, page_size: int = 50, search: Optional[str] = None):
        return self.db.list_all(page=page, page_size=page_size, search=search)

    def get_by_id(self, medication_id: int):
        return self.db.get_by_id(medication_id)

    def get_stats(self):
        stats = self.db.get_stats()
        stats["ocr_engines"] = {
            "tesseract": self.ocr._tesseract_available,
            "easyocr": True  # always available as fallback
        }
        return stats
