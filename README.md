# 💊 Medication Box Recognition API

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi" />
  <img src="https://img.shields.io/badge/OCR-Tesseract%20%2B%20EasyOCR-orange" />
  <img src="https://img.shields.io/badge/Database-5031%20Medications-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> **Project #2 — Master IT 2025/2026**
> Université Mohammed V – Faculté des Sciences, Rabat
> Supervised by: **Abdelhak Mahmoudi** | Co-supervised by: **Saad Frihi** and **Yasine Lehmiani**

---

## 📋 Overview

An intelligent REST API that recognizes medications from images of their boxes or packaging using Computer Vision and OCR.

**Key capabilities:**
- 📷 Accepts images in JPG, PNG, BMP, TIFF, WEBP formats
- 🔤 Extracts text using Tesseract OCR (Arabic + French) with EasyOCR fallback
- 🔍 Matches extracted text against a reference database of **5,031 Moroccan medications**
- 🚀 Returns top matches with confidence scores in <1 second
- 🌐 RESTful API with interactive Swagger documentation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Application                  │
│                                                         │
│  POST /api/v1/recognize                                 │
│       │                                                 │
│       ▼                                                 │
│  ┌────────────────┐     ┌──────────────────────────┐   │
│  │ ImagePreprocessor│   │     OCR Service           │   │
│  │                 │──▶│  ┌──────────┐             │   │
│  │ • Resize        │   │  │Tesseract │ (primary)   │   │
│  │ • Denoise       │   │  └──────────┘             │   │
│  │ • CLAHE         │   │  ┌──────────┐             │   │
│  │ • Binarize      │   │  │ EasyOCR  │ (fallback)  │   │
│  │ • Deskew        │   │  └──────────┘             │   │
│  └────────────────┘     └──────────────────────────┘   │
│                                    │                    │
│                                    ▼                    │
│                         ┌──────────────────────┐       │
│                         │  MedicationDatabase   │       │
│                         │                       │       │
│                         │  • Exact match        │       │
│                         │  • Fuzzy match        │       │
│                         │  • Token match        │       │
│                         │  • Partial match      │       │
│                         │                       │       │
│                         │  5,031 medications    │       │
│                         └──────────────────────┘       │
│                                    │                    │
│                                    ▼                    │
│                         JSON Response with matches      │
└─────────────────────────────────────────────────────────┘
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- Tesseract OCR engine

#### Install Tesseract

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
Download installer from https://github.com/UB-Mannheim/tesseract/wiki

### Install Python dependencies

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/MasterIT_Project2_MedOCR.git
cd MasterIT_Project2_MedOCR

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the API

```bash
python -m app.main
# or
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API:** http://localhost:8000
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root / status |
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/recognize` | **Recognize medication from image** |
| `POST` | `/api/v1/extract-text` | Extract text only (OCR) |
| `GET` | `/api/v1/search?q=...` | Search medications by name |
| `GET` | `/api/v1/medications` | List all medications (paginated) |
| `GET` | `/api/v1/medications/{id}` | Get medication by ID |
| `GET` | `/api/v1/stats` | Database statistics |

### Example: Recognize medication

```bash
curl -X POST "http://localhost:8000/api/v1/recognize" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/medication_box.jpg"
```

**Response:**
```json
{
  "success": true,
  "extracted_text": "DOLIPRANE 500MG 16 CP EFFER",
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
    },
    {
      "id": 59926,
      "name": "DOLIPRANE 500MG 20 CP (H)",
      "confidence": 0.85,
      "match_score": 85.0,
      "match_type": "fuzzy"
    }
  ],
  "total_matches": 2,
  "processing_time_seconds": 0.432
}
```

### Example: Search by name

```bash
curl "http://localhost:8000/api/v1/search?q=DOLIPRANE&top_n=5"
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_database.py -v
pytest tests/test_integration.py -v

# Run with coverage
pip install pytest-cov
pytest tests/ --cov=app --cov-report=html
```

---

## 📁 Project Structure

```
MasterIT_Project2_MedOCR/
│
├── app/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Settings & configuration
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── routers/
│   │   ├── health.py            # Health check endpoints
│   │   └── recognition.py      # Recognition endpoints
│   └── services/
│       ├── preprocessing.py     # Image preprocessing pipeline
│       ├── ocr_service.py       # OCR (Tesseract + EasyOCR)
│       ├── database.py          # Medication DB + fuzzy matching
│       └── recognition_service.py # Orchestration
│
├── data/
│   └── produits.json            # 5,031 Moroccan medication reference DB
│
├── tests/
│   ├── test_database.py         # Database matching unit tests
│   ├── test_preprocessing.py    # Image preprocessing tests
│   └── test_integration.py      # Full pipeline + API tests
│
├── docs/
│   └── architecture.md          # Detailed architecture documentation
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧠 Technical Details

### Image Preprocessing Pipeline

1. **Load** — decode JPEG/PNG/etc. via OpenCV + PIL fallback
2. **Resize** — limit to 3000px max, upscale if < 800px (helps OCR)
3. **Grayscale** — convert BGR to grayscale
4. **Denoise** — `fastNlMeansDenoising` (removes JPEG artifacts)
5. **CLAHE** — adaptive histogram equalization (improves contrast)
6. **Adaptive Thresholding** — binarize for OCR
7. **Deskew** — correct slight rotation using `minAreaRect`
8. **Morphological cleanup** — remove small noise

### Matching Strategies

| Strategy | Use case | Scorer |
|----------|----------|--------|
| Exact | Perfect match | `==` |
| Prefix | Name starts with query | String prefix |
| Fuzzy token set | OCR noise, word order variation | `token_set_ratio` |
| Partial | Substring in name | `partial_ratio` |
| Multi-token | OCR produces separate words | Per-token aggregation |

### Supported Languages
- 🇫🇷 **French** — primary medication labeling language in Morocco
- 🇲🇦 **Arabic/Darija** — secondary labeling language
- 🇬🇧 **English** — some international brands

---

## 📊 Dataset

The `produits.json` database contains **5,031 medications** available on the Moroccan market, sourced from official pharmaceutical data.

Each entry:
```json
{ "value": "DOLIPRANE 500MG 20 CP (H)", "id": 59926 }
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👥 Team

| Role | Name |
|------|------|
| Supervisor | Abdelhak Mahmoudi |
| Co-Supervisor | Saad Frihi |
| Co-Supervisor | Yasine Lehmiani |

**Université Mohammed V — Faculté des Sciences, Rabat**
**Master IT — April 2025/2026**
