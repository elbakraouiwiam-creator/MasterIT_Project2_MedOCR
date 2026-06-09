"""
Pydantic schemas for API request/response models.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class MatchedMedication(BaseModel):
    id: int = Field(..., description="Medication ID in the reference database")
    name: str = Field(..., description="Medication name as in the database")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Match confidence (0.0 to 1.0)")
    match_score: float = Field(..., description="Raw match score (0-100)")
    match_type: str = Field(..., description="How the match was found: exact/fuzzy/partial/token")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 60401,
                "name": "SPASFON 80MG B/30 CP",
                "confidence": 0.95,
                "match_score": 95.0,
                "match_type": "fuzzy"
            }
        }


class RecognitionResponse(BaseModel):
    success: bool
    extracted_text: str = Field(..., description="Cleaned OCR text from the image")
    raw_ocr_text: str = Field(..., description="Raw OCR output before cleaning")
    ocr_confidence: float = Field(..., description="Average OCR confidence (0.0-1.0)")
    ocr_engine: str = Field(..., description="OCR engine used")
    matches: List[MatchedMedication] = Field(..., description="Top matching medications")
    total_matches: int
    processing_time_seconds: float

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "extracted_text": "DOLIPRANE 500MG 16 CP",
                "raw_ocr_text": "DOLIPRANE 500MG 16 CP EFFER.",
                "ocr_confidence": 0.87,
                "ocr_engine": "tesseract",
                "matches": [
                    {
                        "id": 62562,
                        "name": "DOLIPRANE 500MG 16 CP EFFER.",
                        "confidence": 0.97,
                        "match_score": 97.0,
                        "match_type": "fuzzy"
                    }
                ],
                "total_matches": 1,
                "processing_time_seconds": 0.432
            }
        }


class TextExtractionResponse(BaseModel):
    success: bool
    raw_text: str
    cleaned_text: str
    tokens: List[str]
    confidence: float
    engine: str
    processing_time_seconds: float


class SearchMatch(BaseModel):
    id: int
    name: str
    score: float
    match_type: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchMatch]
    total: int
