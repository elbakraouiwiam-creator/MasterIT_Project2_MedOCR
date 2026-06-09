# ══════════════════════════════════════════════════════════════
#  PHASE 4 — API REST FastAPI
#  Medication Box Recognition API — Version optimisée
#
#  Endpoints :
#    GET  /              -> page d'accueil avec test rapide
#    GET  /health        -> statut de l'API
#    GET  /medicaments   -> liste des médicaments en base
#    POST /recognize     -> reconnaitre un médicament
#
#  Lancer : uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# ══════════════════════════════════════════════════════════════

import os
import sys
import io
import re
import time
import unicodedata
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_ocr import Preprocessor, detect_language, extract_dosage
from matching import Matcher
import easyocr

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "reference" / "medicaments_final.csv"
GPU      = False

# ══════════════════════════════════════════════════════════════
#  INITIALISATION FASTAPI
# ══════════════════════════════════════════════════════════════
app = FastAPI(
    title       = "MedBox Recognition API",
    description = "API de reconnaissance de boites de medicaments marocains via OCR",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ══════════════════════════════════════════════════════════════
#  CHARGEMENT DES MODELES AU DEMARRAGE
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  Demarrage de MedBox Recognition API...")
print("="*55)

print("Chargement des modeles OCR...")
reader_ar = easyocr.Reader(["ar", "en"], gpu=GPU, verbose=False)
reader_fr = easyocr.Reader(["fr", "en"], gpu=GPU, verbose=False)
print("Modeles OCR prets")

preprocessor = Preprocessor(max_size=1024)

print("Chargement de la base medicaments...")
matcher = Matcher(str(CSV_PATH))
matcher.SEUIL_CONFIANCE = 0.30
print(f"Base chargee : {len(matcher.base.medicaments)} medicaments")
print("="*55 + "\n")


# ══════════════════════════════════════════════════════════════
#  CORRECTION ORIENTATION (optionnelle)
# ══════════════════════════════════════════════════════════════

def corriger_orientation(img: np.ndarray) -> np.ndarray:
    """
    Teste 4 rotations avec AR + FR et garde la meilleure.
    Activee seulement si corriger_rotation=True dans la requete.
    """
    rotations = {
        0:   img,
        90:  cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(img, cv2.ROTATE_180),
        270: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }

    meilleur_score = -1
    meilleure_img  = img
    meilleur_angle = 0

    for angle, img_rot in rotations.items():
        try:
            res_ar   = reader_ar.readtext(img_rot)
            res_fr   = reader_fr.readtext(img_rot)
            score_ar = sum(float(conf) for (_, _, conf) in res_ar
                          if conf is not None and float(conf) > 0.30)
            score_fr = sum(float(conf) for (_, _, conf) in res_fr
                          if conf is not None and float(conf) > 0.40)
            score    = score_ar + score_fr
        except Exception:
            score    = 0.0
            score_ar = 0.0
            score_fr = 0.0

        print(f"  Rotation {angle}deg score AR:{score_ar:.2f} FR:{score_fr:.2f}")

        if score > meilleur_score:
            meilleur_score = score
            meilleure_img  = img_rot
            meilleur_angle = angle

    print(f"  Meilleure orientation : {meilleur_angle}deg")
    return meilleure_img


# ══════════════════════════════════════════════════════════════
#  OCR PRINCIPAL
# ══════════════════════════════════════════════════════════════

def extraire_texte_ocr(img: np.ndarray, confidence_min: float = 0.25) -> dict:
    """Extrait le texte arabe et francais depuis une image."""
    try:
        raw_ar = reader_ar.readtext(img)
    except Exception:
        raw_ar = []
    try:
        raw_fr = reader_fr.readtext(img)
    except Exception:
        raw_fr = []

    raw  = raw_ar + raw_fr
    zones, textes_ar, textes_fr = [], [], []
    dosage, conf_total = None, []
    vus = set()

    for item in raw:
        try:
            bbox, text, conf = item
            text = str(text).strip()
            conf = float(conf) if conf is not None else 0.0

            if conf < confidence_min or not text or text in vus:
                continue
            vus.add(text)

            text = unicodedata.normalize("NFC", text)
            text = re.sub(r"[^\w\s\-\./+%\u0600-\u06FF]", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue

            lang = detect_language(text)
            d    = extract_dosage(text)
            if d and not dosage:
                dosage = d

            zones.append({
                "texte":     text,
                "langue":    lang,
                "confiance": round(conf, 3),
                "dosage":    d,
            })
            conf_total.append(conf)

            if "AR" in lang:
                textes_ar.append(text)
            else:
                textes_fr.append(text)

        except Exception:
            continue

    confiance_moy = 0.0
    if conf_total:
        confiance_moy = round(sum(conf_total) / len(conf_total), 3)

    return {
        "nb_zones":      len(zones),
        "texte_fr":      " | ".join(textes_fr),
        "texte_ar":      " | ".join(textes_ar),
        "texte_complet": " ".join(textes_ar + textes_fr),
        "dosage":        dosage,
        "confiance_moy": confiance_moy,
        "zones":         zones,
    }


# ══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def accueil():
    html = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MedBox Recognition API</title>
        <style>
            * { margin:0; padding:0; box-sizing:border-box; }
            body { font-family:-apple-system,sans-serif; background:#0f172a; color:#e2e8f0; min-height:100vh; display:flex; align-items:center; justify-content:center; padding:1rem; }
            .container { max-width:700px; width:100%; text-align:center; }
            .logo { font-size:4rem; margin-bottom:1rem; }
            h1 { font-size:2rem; color:#38bdf8; margin-bottom:0.5rem; }
            .subtitle { color:#94a3b8; margin-bottom:2rem; }
            .stats { display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin:2rem 0; }
            .stat { background:#1e293b; border-radius:12px; padding:1.5rem; border:1px solid #334155; }
            .stat-num { font-size:2rem; font-weight:700; color:#38bdf8; }
            .stat-label { font-size:0.85rem; color:#94a3b8; margin-top:0.3rem; }
            .endpoints { background:#1e293b; border-radius:12px; padding:1.5rem; text-align:left; border:1px solid #334155; margin-bottom:1rem; }
            .endpoints h3 { color:#38bdf8; margin-bottom:1rem; }
            .ep { display:flex; align-items:center; gap:0.8rem; padding:0.5rem 0; border-bottom:1px solid #334155; flex-wrap:wrap; }
            .ep:last-child { border:none; }
            .method { font-size:0.75rem; font-weight:700; padding:3px 8px; border-radius:4px; white-space:nowrap; }
            .get { background:#065f46; color:#6ee7b7; }
            .post { background:#1e3a5f; color:#93c5fd; }
            .path { font-family:monospace; color:#e2e8f0; }
            .desc { color:#94a3b8; font-size:0.85rem; margin-left:auto; }
            .upload-section { background:#1e293b; border-radius:12px; padding:1.5rem; margin-top:1rem; border:1px solid #334155; }
            .upload-section h3 { color:#38bdf8; margin-bottom:1rem; }
            input[type=file] { display:none; }
            .upload-btn { background:#1d4ed8; color:white; padding:0.6rem 1.5rem; border-radius:8px; cursor:pointer; border:none; font-size:1rem; display:inline-block; margin-bottom:0.8rem; }
            .option-row { display:flex; align-items:center; justify-content:center; gap:0.5rem; margin:0.5rem 0; font-size:0.9rem; color:#94a3b8; }
            .option-row input { width:16px; height:16px; cursor:pointer; }
            #result { margin-top:1rem; background:#0f172a; border-radius:8px; padding:1rem; text-align:left; display:none; line-height:1.9; }
            .success { color:#6ee7b7; font-weight:bold; font-size:1.1rem; }
            .error-msg { color:#f87171; font-weight:bold; font-size:1.1rem; }
            #preview { max-width:250px; border-radius:8px; margin:0.8rem auto; display:none; }
            .btn { display:inline-block; margin-top:1.5rem; padding:0.8rem 2rem; background:#38bdf8; color:#0f172a; border-radius:8px; text-decoration:none; font-weight:700; }
            .loading { color:#94a3b8; }
            .info-box { background:#1e3a5f; border-radius:8px; padding:0.8rem; margin:0.5rem 0; font-size:0.85rem; color:#93c5fd; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">💊</div>
            <h1>MedBox Recognition API</h1>
            <p class="subtitle">Reconnaissance de boites de medicaments marocains via OCR</p>

            <div class="stats">
                <div class="stat">
                    <div class="stat-num">2877</div>
                    <div class="stat-label">Medicaments en base</div>
                </div>
                <div class="stat">
                    <div class="stat-num">AR+FR</div>
                    <div class="stat-label">Langues supportees</div>
                </div>
                <div class="stat">
                    <div class="stat-num">1.0</div>
                    <div class="stat-label">Version API</div>
                </div>
            </div>

            <div class="endpoints">
                <h3>Endpoints disponibles</h3>
                <div class="ep">
                    <span class="method post">POST</span>
                    <span class="path">/recognize</span>
                    <span class="desc">Identifier un medicament</span>
                </div>
                <div class="ep">
                    <span class="method get">GET</span>
                    <span class="path">/health</span>
                    <span class="desc">Statut de l'API</span>
                </div>
                <div class="ep">
                    <span class="method get">GET</span>
                    <span class="path">/medicaments</span>
                    <span class="desc">Liste des medicaments</span>
                </div>
                <div class="ep">
                    <span class="method get">GET</span>
                    <span class="path">/docs</span>
                    <span class="desc">Documentation Swagger</span>
                </div>
            </div>

            <div class="upload-section">
                <h3>Test rapide</h3>

                <div class="info-box">
                    Mode normal : ~15-20 sec | Mode rotation : ~60-90 sec
                </div>

                <label for="file-input" class="upload-btn">
                    Choisir une photo de boite
                </label>
                <input type="file" id="file-input" accept="image/*"
                       onchange="testerImage(this)">

                <div class="option-row">
                    <input type="checkbox" id="rotation-check">
                    <label for="rotation-check">
                        Corriger orientation automatiquement
                        (photo a l'envers ou de cote)
                    </label>
                </div>

                <img id="preview">
                <div id="result"></div>
            </div>

            <a href="/docs" class="btn">Documentation Swagger</a>
        </div>

        <script>
        async function testerImage(input) {
            const file = input.files[0];
            if (!file) return;

            const preview = document.getElementById('preview');
            preview.src   = URL.createObjectURL(file);
            preview.style.display = 'block';

            const corriger = document.getElementById('rotation-check').checked;
            const result   = document.getElementById('result');
            result.style.display = 'block';
            result.innerHTML = corriger
                ? '<span class="loading">Analyse en cours avec correction rotation... (~60-90 sec)</span>'
                : '<span class="loading">Analyse en cours... (~15-20 sec)</span>';

            const formData = new FormData();
            formData.append('file', file);
            formData.append('corriger_rotation', corriger ? 'true' : 'false');

            try {
                const resp = await fetch('/recognize', {
                    method: 'POST',
                    body:   formData
                });
                const data = await resp.json();

                if (!resp.ok) {
                    result.innerHTML = '<span class="error-msg">Erreur : '
                        + (data.detail || 'Erreur serveur') + '</span>';
                    return;
                }

                const confiance = (data.confiance && !isNaN(data.confiance))
                    ? (data.confiance * 100).toFixed(0) + '%'
                    : 'N/A';

                if (data.statut === 'identifie' && data.meilleur_match) {
                    const m = data.meilleur_match;
                    const nomAr = m.nom_ar ? `<b>Nom AR :</b> ${m.nom_ar}<br>` : '';
                    const prix  = m.prix   ? `<b>Prix :</b> ${m.prix} DH<br>`  : '';
                    result.innerHTML = `
                        <div class="success">✅ Medicament identifie !</div><br>
                        <b>Nom :</b> ${m.nom_commercial || '-'}<br>
                        ${nomAr}
                        <b>DCI :</b> ${m.DCI || '-'}<br>
                        <b>Forme :</b> ${m.forme || '-'}<br>
                        <b>Laboratoire :</b> ${m.laboratoire || '-'}<br>
                        <b>Categorie :</b> ${m.categorie || '-'}<br>
                        ${prix}
                        <b>Confiance :</b> ${confiance}<br>
                        <b>Temps :</b> ${data.performance ? data.performance.temps_total_ms + 'ms' : '-'}
                    `;
                } else {
                    result.innerHTML = `
                        <div class="error-msg">❌ Medicament non identifie</div><br>
                        <b>Score :</b> ${confiance}<br>
                        <b>Texte FR detecte :</b> ${data.texte_ocr_fr || '-'}<br>
                        <b>Texte AR detecte :</b> ${data.texte_ocr_ar || '-'}<br>
                        <b>Zones OCR :</b> ${data.nb_zones_ocr || 0}<br>
                        <b>Conseil :</b> Essayez avec la correction de rotation activee.
                    `;
                }
            } catch(e) {
                result.innerHTML = '<span class="error-msg">Erreur reseau : '
                    + e.message + '</span>';
            }
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    return {
        "statut":         "ok",
        "version":        "1.0.0",
        "nb_medicaments": len(matcher.base.medicaments),
        "langues":        ["ar", "fr", "en"],
        "timestamp":      datetime.now().isoformat(),
    }


@app.get("/medicaments")
async def liste_medicaments(
    categorie: str = None,
    recherche: str = None,
    limite:    int = 50
):
    medicaments = matcher.base.medicaments

    if categorie:
        medicaments = [m for m in medicaments
                       if m.get("categorie","").lower() == categorie.lower()]
    if recherche:
        medicaments = [m for m in medicaments
                       if recherche.lower() in m["nom_commercial"].lower()]

    medicaments = medicaments[:limite]

    return {
        "total":       len(matcher.base.medicaments),
        "retournes":   len(medicaments),
        "medicaments": [
            {
                "nom_commercial": m["nom_commercial"],
                "DCI":            m.get("DCI",""),
                "dosage":         m.get("dosage",""),
                "forme":          m.get("forme",""),
                "laboratoire":    m.get("laboratoire",""),
                "categorie":      m.get("categorie",""),
                "nom_ar":         m.get("nom_ar",""),
            }
            for m in medicaments
        ]
    }


@app.post("/recognize")
async def reconnaitre(
    file:              UploadFile = File(...),
    corriger_rotation: Optional[str] = Form(default="false")
):
    """
    Identifie un medicament depuis une photo de boite.

    - **file** : image JPG, PNG ou WebP
    - **corriger_rotation** : true/false — corriger l'orientation automatiquement
      (active si photo a l'envers ou de cote, mais 4x plus lent)
    """
    t_start   = time.time()
    do_rotate = corriger_rotation.lower() == "true"

    # Verification format
    formats_ok = ["image/jpeg","image/png","image/webp",
                  "image/bmp","image/jpg"]
    if file.content_type not in formats_ok:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporte : {file.content_type}. "
                   f"Utilisez JPG, PNG ou WebP."
        )

    # Lecture image
    try:
        contenu = await file.read()
        img_pil = Image.open(io.BytesIO(contenu)).convert("RGB")
        img     = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Erreur lecture image : {str(e)}")

    h_orig, w_orig = img.shape[:2]

    # Correction orientation (optionnelle)
    orientation_corrigee = False
    if do_rotate:
        try:
            img = corriger_orientation(img)
            orientation_corrigee = True
        except Exception as e:
            print(f"Erreur orientation : {e}")

    # Pretraitement
    try:
        img_prep, infos_prep = preprocessor.traiter(img)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Erreur pretraitement : {str(e)}")

    # OCR
    try:
        t_ocr = time.time()
        ocr   = extraire_texte_ocr(img_prep)
        t_ocr = round((time.time() - t_ocr) * 1000)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Erreur OCR : {str(e)}")

    # Matching
    try:
        t_match = time.time()
        match   = matcher.identifier({"ocr": ocr})
        t_match = round((time.time() - t_match) * 1000)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Erreur matching : {str(e)}")

    t_total = round((time.time() - t_start) * 1000)

    # Protection NaN
    confiance = match.get("confiance", 0.0)
    if confiance is None or (isinstance(confiance, float)
                              and confiance != confiance):
        confiance = 0.0

    return JSONResponse(content={
        "statut":              match.get("statut", "non_identifie"),
        "confiance":           round(float(confiance), 3),
        "niveau_confiance":    match.get("niveau_confiance", "faible"),
        "meilleur_match":      match.get("meilleur_match"),
        "top3":                match.get("top3", []),
        "texte_ocr_fr":        ocr.get("texte_fr", ""),
        "texte_ocr_ar":        ocr.get("texte_ar", ""),
        "dosage_detecte":      ocr.get("dosage"),
        "nb_zones_ocr":        ocr.get("nb_zones", 0),
        "orientation_corrigee": orientation_corrigee,
        "image_info": {
            "nom_fichier":  file.filename,
            "dimensions":   f"{w_orig}x{h_orig}px",
            "niveau_flou":  infos_prep.get("niveau_flou", ""),
            "blur_score":   float(
                infos_prep.get("blur_score_avant", 0) or 0
            ),
        },
        "performance": {
            "temps_ocr_ms":      t_ocr,
            "temps_matching_ms": t_match,
            "temps_total_ms":    t_total,
        },
        "timestamp": datetime.now().isoformat(),
    })