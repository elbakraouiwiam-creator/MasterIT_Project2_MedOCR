from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/", summary="Root endpoint")
async def root():
    return {
        "message": "Medication Box Recognition API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@router.get("/health", summary="Health check")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Medication Box Recognition API"
    }
