"""
Project #2: Medication Box Recognition API
Université Mohammed V - Faculté des Sciences, Rabat
Master IT - 2025/2026
Supervised by: Abdelhak Mahmoudi
Co-Supervised by: Saad Frihi and Yasine Lehmiani
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time

from app.routers import recognition, health
from app.config import settings

app = FastAPI(
    title="Medication Box Recognition API",
    description="""
    ## Medication Box Recognition API

    An intelligent API that recognizes medications from images of their boxes/packaging.

    ### Features
    - 📷 Image-based medication recognition using Computer Vision
    - 🔤 OCR text extraction (Arabic & French support)
    - 🔍 Fuzzy matching against a Moroccan medication database (5031 products)
    - 🚀 Fast and accurate results

    ### How it works
    1. Upload an image of a medication box
    2. The pipeline extracts text using OCR
    3. Text is matched against the medication database
    4. Returns the best matching medication(s) with confidence scores
    """,
    version="1.0.0",
    contact={
        "name": "Master IT - Université Mohammed V",
        "url": "https://fsr.ac.ma",
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(recognition.router, prefix="/api/v1", tags=["Recognition"])


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
