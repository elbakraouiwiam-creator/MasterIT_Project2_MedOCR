"""
Recognition endpoints for the Medication Box Recognition API
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
import io

from app.services.recognition_service import RecognitionService
from app.models.schemas import RecognitionResponse, TextExtractionResponse, SearchResponse
from app.config import settings

router = APIRouter()
recognition_service = RecognitionService()


@router.post(
    "/recognize",
    response_model=RecognitionResponse,
    summary="Recognize medication from box image",
    description="""
    Upload an image of a medication box and get the recognized medication(s).

    The pipeline:
    1. Preprocesses the image (denoising, contrast enhancement)
    2. Extracts text using OCR (supports Arabic & French)
    3. Matches extracted text against the medication database
    4. Returns top matches with confidence scores

    **Supported formats:** JPG, JPEG, PNG, BMP, TIFF, WEBP
    **Max file size:** 10 MB
    """
)
async def recognize_medication(
    file: UploadFile = File(..., description="Image of the medication box"),
    top_n: int = Query(default=5, ge=1, le=20, description="Number of top results to return"),
    threshold: int = Query(default=60, ge=0, le=100, description="Minimum match confidence (0-100)")
):
    # Validate file type
    ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )

    # Read image bytes
    image_bytes = await file.read()
    if len(image_bytes) > settings.MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_IMAGE_SIZE // (1024*1024)} MB"
        )

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        result = recognition_service.recognize(
            image_bytes=image_bytes,
            top_n=top_n,
            threshold=threshold
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")


@router.post(
    "/extract-text",
    response_model=TextExtractionResponse,
    summary="Extract text from medication box image (OCR only)",
    description="Returns raw OCR-extracted text without medication matching."
)
async def extract_text(
    file: UploadFile = File(..., description="Image of the medication box")
):
    ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        result = recognition_service.extract_text_only(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")


@router.get(
    "/search",
    response_model=SearchResponse,
    summary="Search medication database by name",
    description="Search the medication database by text query (no image needed)."
)
async def search_medications(
    q: str = Query(..., min_length=2, description="Search query (medication name)"),
    top_n: int = Query(default=10, ge=1, le=50),
    threshold: int = Query(default=50, ge=0, le=100)
):
    try:
        results = recognition_service.search_database(q, top_n=top_n, threshold=threshold)
        return SearchResponse(query=q, results=results, total=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/medications",
    summary="List all medications in the database",
    description="Returns a paginated list of all medications in the reference database."
)
async def list_medications(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    search: Optional[str] = Query(default=None, description="Optional filter by name")
):
    try:
        return recognition_service.list_medications(page=page, page_size=page_size, search=search)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/medications/{medication_id}",
    summary="Get medication by ID"
)
async def get_medication(medication_id: int):
    result = recognition_service.get_by_id(medication_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Medication with ID {medication_id} not found")
    return result


@router.get(
    "/stats",
    summary="Database statistics"
)
async def get_stats():
    return recognition_service.get_stats()
