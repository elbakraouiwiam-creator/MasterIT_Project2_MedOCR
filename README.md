# 💊 Medication Box Recognition API

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi" />
  <img src="https://img.shields.io/badge/OCR-Tesseract%20%2B%20EasyOCR-orange" />
  <img src="https://img.shields.io/badge/Database-7,913%20Medications-purple" />
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
- 🔤 Extracts text using EasyOCR (Arabic + French) with EasyOCR fallback
- 🔍 Matches extracted text against a reference database of **7,913 Moroccan medications**
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
│  │ ImagePreprocessor│   │     OCR Service          │   │
│  │                 │──▶│                      │   │
│  │ • Resize        │   │  
│  │ • Denoise       │   │                       │   │
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
- 

#### OCR Engine
We use **EasyOCR** (bilingual Arabic/French support).
It is automatically installed via `pip install -r requirements.txt`.
No separate installation required.

### Install Python dependencies

```bash
# Clone the repository
git clone https://github.com/elbakraouiwiam-creator/MasterIT_Project2_MedOCR.git
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
| `GET` | `/` | Interface web |
| `GET` | `/health` | Health check |
| `POST` | `/recognize` | Reconnaissance depuis image |
| `GET` | `/medicaments` | Liste des médicaments |

### Example: Recognize medication

```bash
curl -X POST "http://localhost:8000/recognize" \
  -F "file=@/path/to/medication_box.jpg"
```

**Response:**
```json
{
  "statut": "identifie",
  "confiance": 0.789,
  "niveau_confiance": "haute",
  "meilleur_match": {
    "nom_commercial": "Doliprane 1000mg",
    "DCI": "Paracetamol",
    "dosage": "1000mg",
    "forme": "Comprime effervescent",
    "laboratoire": "Sanofi",
    "prix": "16.0"
  }
}
```



---

## 📁 Project Structure

```
MasterIT_Project2_MedOCR/

├── app/

│   ├── main.py              # FastAPI app entry point

│   ├── pipeline_ocr.py      # ImageLoader + Preprocessor + OCREngine (EasyOCR)

│   ├── matching.py          # Matching multi-signal (Levenshtein + TF-IDF)

│   └── reference/

│       └── medicaments_final.csv  # 7,913 Moroccan medications

│

├── evaluation/

│   ├── ground_truth.csv     # 150 images test dataset

│   ├── metrics.py           # Evaluation script

│   └── metrics_report.txt   # Results (F1=93.6%)

│

├── data/

│   └── produits.json        # 5,031 medications from pharmacist

│

├── requirements.txt

├── .gitignore

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

| Signal | Poids | Description |
|--------|-------|-------------|
| Levenshtein (rapidfuzz) | 45% | Distance d'edition entre chaines |
| TF-IDF cosinus | 35% | Similarite semantique par n-grammes |
| Score arabe | 20% | Matching specifique texte arabe |
| Bonus dosage | +15% | Bonus si dosage detecte correspond |

Seuil de confiance : **0.30** (optimise sur le dataset de validation)

### Supported Languages
- 🇫🇷 **French** — primary medication labeling language in Morocco
- 🇲🇦 **Arabic/Darija** — secondary labeling language
- 🇬🇧 **English** — some international brands

---

## 📊 Dataset

The reference database contains **7,913 medications** available on the Moroccan market, built by merging two sources:

- **CNOPS 2014** : official Moroccan pharmaceutical database (with PPV prices)
- **Pharmacist database** : 5,031 updated medications (`produits.json`)

Each entry example:
```json
{ "value": "DOLIPRANE 500MG 20 CP (H)", "id": 59926 }
```

Evaluation dataset: **150 annotated images** — Precision: 100% | Recall: 88% | F1: 93.6%

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
| Student | Chaimae Jai |
| Student | Wiam El Bakraoui |

**Université Mohammed V — Faculté des Sciences, Rabat**
**Master IT —  2025/2027**
