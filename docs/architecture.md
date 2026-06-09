# Architecture Documentation

## Medication Box Recognition API — Technical Design

### 1. Overview

The system is a RESTful API that accepts medication box images and returns matched medication names from a reference database of 5,031 Moroccan medications.

### 2. Component Breakdown

#### 2.1 ImagePreprocessor (`app/services/preprocessing.py`)

Prepares raw images for OCR by applying a sequence of transformations:

| Step | Technique | Purpose |
|------|-----------|---------|
| Resize | Bicubic interpolation | Normalize resolution for OCR |
| Grayscale | BGR→Gray | Remove color noise |
| Denoise | `fastNlMeansDenoising` | Remove JPEG compression artifacts |
| CLAHE | Adaptive histogram eq. | Improve local contrast |
| Binarize | Adaptive thresholding / Otsu | Create high-contrast B&W |
| Deskew | `minAreaRect` + affine | Correct text rotation |
| Morphological | Close kernel | Clean up small noise |

**Multi-view strategy:** The preprocessor generates 4 variants (standard, high-contrast, grayscale, inverted). All are passed to OCR and results are merged, maximizing text coverage.

#### 2.2 OCRService (`app/services/ocr_service.py`)

Two-engine design:

```
Primary: Tesseract OCR
  - Engines: OEM 3 (LSTM)
  - Modes: PSM 3 (fully auto)
  - Languages tried: fra+ara → fra+eng → eng
  
Fallback: EasyOCR
  - Languages: ['fr', 'ar', 'en']
  - Activated when Tesseract unavailable or returns empty text
```

Text cleaning:
- Normalize whitespace
- Remove non-printable characters
- Preserve: Latin, Arabic (U+0600–U+06FF), accented chars, digits
- Uppercase normalization

#### 2.3 MedicationDatabase (`app/services/database.py`)

In-memory indexed database built at startup:
- `_id_to_med`: dict mapping ID → medication object (O(1) lookup)
- `_name_to_med`: dict mapping normalized name → medication object
- `_normalized_names`: list of all normalized names (for rapidfuzz batch matching)

**Matching pipeline:**

```
query text
    │
    ├── Exact match (O(1) dict lookup)
    │
    ├── Prefix match (fast scan)
    │
    ├── Fuzzy token_set_ratio via rapidfuzz.process.extract
    │   (handles word reordering, OCR noise)
    │
    └── Partial ratio
        (handles substring matches for partial text)
```

For token-based search (when OCR returns individual words):
- Each token is searched independently
- Scores are accumulated: medication matching multiple tokens scores higher

#### 2.4 RecognitionService (`app/services/recognition_service.py`)

Orchestrator combining all components:

```python
def recognize(image_bytes, top_n, threshold):
    versions = preprocessor.preprocess_multiple_views(image_bytes)
    ocr_result = ocr.extract_text_multi(versions)
    
    # Full text search first
    matches = db.search(ocr_result.cleaned_text, ...)
    
    # Supplement with token search if needed
    if len(matches) < top_n:
        token_matches = db.search_by_tokens(ocr_result.tokens, ...)
        matches = merge(matches, token_matches)
    
    return RecognitionResponse(...)
```

### 3. API Layer

Built with FastAPI:
- **Async** request handling
- **Pydantic v2** validation for all inputs/outputs
- **CORS** enabled for all origins
- **Process time** header added to all responses
- **OpenAPI** docs at `/docs`

### 4. Performance Considerations

| Component | Typical time |
|-----------|-------------|
| Image preprocessing | 50–200ms |
| Tesseract OCR | 200–500ms |
| Database matching | 10–50ms |
| **Total** | **~0.3–0.8s** |

For production:
- Pre-load EasyOCR model at startup
- Add Redis caching for repeated images (hash-based)
- Consider GPU for EasyOCR

### 5. Handling Challenging Conditions

| Challenge | Solution |
|-----------|----------|
| Blur | CLAHE + denoising |
| Low lighting | Adaptive thresholding |
| Partial visibility | Token-based matching |
| Arabic text | Separate Arabic preprocessing mode |
| OCR noise | Fuzzy token_set_ratio matching |
| Word order variation | Token set ratio (order-insensitive) |
| Different box sizes | Auto-resize pipeline |
