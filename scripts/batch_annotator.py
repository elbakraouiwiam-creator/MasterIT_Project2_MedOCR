# ══════════════════════════════════════════════════════════════
#  ANNOTATION RAPIDE — Version optimisée GPU
#  Objectif : traiter 2213 images en 1-2h au lieu de plusieurs jours
#
#  Optimisations :
#    - Un seul reader EasyOCR (ar + en) au lieu de deux
#    - Batch GPU : plusieurs images envoyées ensemble au modèle
#    - Resize plus agressif pour réduire la charge
#    - Sauvegarde toutes les 50 images (moins d'I/O Drive)
# ══════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

!pip install easyocr langdetect -q

import os, json, csv, re, time
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# ── Vérification GPU ──────────────────────────────────────────
print("=" * 60)
gpu_ok = torch.cuda.is_available()
print(f"  GPU disponible : {gpu_ok}")
if gpu_ok:
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
else:
    print("  ⚠️  PAS DE GPU — va dans Exécution > Modifier le type d'exécution > T4 GPU")
print("=" * 60)

import easyocr
try:
    from langdetect import detect
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
BASE_DIR   = "/content/drive/MyDrive/MasterIT_Project2_ML_DL"
INPUT_DIR  = f"{BASE_DIR}/MasterIT_Project2_Dataset"
OUTPUT_DIR = f"{BASE_DIR}/annotations"
LABELS_DIR = f"{OUTPUT_DIR}/labels_yolo"
TEMP_JSON  = f"{OUTPUT_DIR}/annotations_temp.json"
FINAL_JSON = f"{OUTPUT_DIR}/annotations_complete.json"
FINAL_CSV  = f"{OUTPUT_DIR}/metadata_to_fill.csv"
ERRORS_JSON= f"{OUTPUT_DIR}/errors.json"

CONFIDENCE_MIN = 0.35
EXTS           = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
SAVE_EVERY     = 50      # sauvegarde tous les 50 images (moins d'écriture Drive)
MAX_SIZE       = 1024    # réduit de 1600→1024 : 2.5x plus rapide sur GPU

for d in [OUTPUT_DIR, LABELS_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    ratio  = arabic / max(len(text), 1)
    if ratio > 0.4:   return "AR"
    if ratio > 0.1:   return "AR+FR"
    return "FR"

def extract_dosage(text: str):
    m = re.search(r"(\d+\.?\d*)\s*(mg|ml|g|mcg|µg|UI|%)", text, re.IGNORECASE)
    return m.group(0).strip() if m else None

def bbox_to_yolo(bbox, W, H):
    xs  = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
    x_c = ((min(xs)+max(xs))/2) / W
    y_c = ((min(ys)+max(ys))/2) / H
    w   = (max(xs)-min(xs)) / W
    h   = (max(ys)-min(ys)) / H
    return round(x_c,6), round(y_c,6), round(w,6), round(h,6)

def preprocess(img_path: str) -> np.ndarray:
    """Prétraitement allégé — MAX_SIZE réduit pour accélérer le GPU."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Illisible : {img_path}")
    h, w = img.shape[:2]
    # Resize plus agressif : 1024px max au lieu de 1600px
    if max(h, w) > MAX_SIZE:
        s   = MAX_SIZE / max(h, w)
        img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    # CLAHE contraste
    lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(2.0, (8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

def collect_images(folder: str) -> list:
    imgs = []
    for ext in EXTS:
        imgs.extend(Path(folder).rglob(f"*{ext}"))
        imgs.extend(Path(folder).rglob(f"*{ext.upper()}"))
    return sorted(set(imgs))

def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
#  ÉTAPE 1 — Charger les annotations existantes
# ══════════════════════════════════════════════════════════════
print("\n📂 Chargement des annotations existantes...")
resultats = {}

for fpath in [FINAL_JSON, TEMP_JSON]:
    if os.path.exists(fpath):
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        avant = len(resultats)
        resultats.update(data)
        print(f"   ✅ {Path(fpath).name} : {len(data)} images "
              f"(+{len(resultats)-avant} nouvelles)")

deja_faits = set(resultats.keys())
print(f"   📦 Total déjà annoté : {len(deja_faits)} images")


# ══════════════════════════════════════════════════════════════
#  ÉTAPE 2 — Images restantes
# ══════════════════════════════════════════════════════════════
toutes    = collect_images(INPUT_DIR)
restantes = [p for p in toutes if p.name not in deja_faits]

print(f"\n🖼️  Total dataset : {len(toutes)} images")
print(f"⏳ Restantes     : {len(restantes)} images")
print(f"📈 Progression   : {len(deja_faits)/max(len(toutes),1)*100:.1f}%")

if not restantes:
    print("\n🎉 Toutes les images sont déjà annotées !")
    raise SystemExit

# Estimation du temps
vitesse_estimee = 1.5 if gpu_ok else 8  # secondes par image
eta_total = len(restantes) * vitesse_estimee
print(f"\n⏱️  Temps estimé : ~{eta_total/3600:.1f}h "
      f"({'avec GPU' if gpu_ok else 'SANS GPU — active le GPU !'})")


# ══════════════════════════════════════════════════════════════
#  ÉTAPE 3 — Initialiser UN SEUL reader (optimisation clé)
# ══════════════════════════════════════════════════════════════
print(f"\n🔧 Initialisation EasyOCR (un seul reader ar+en)...")
# Un seul reader ar+en couvre arabe + français translittéré
# C'est 2x plus rapide que deux readers séparés
reader = easyocr.Reader(
    ["ar", "en"],
    gpu=gpu_ok,
    verbose=False,
    # Optimisations GPU
    model_storage_directory='/content/easyocr_models',
)
print("✅ Reader prêt\n")
print("─" * 60)


# ══════════════════════════════════════════════════════════════
#  ÉTAPE 4 — Boucle optimisée
# ══════════════════════════════════════════════════════════════
erreurs   = []
t_start   = time.time()
total_r   = len(restantes)
temps_par_image = []

for i, img_path in enumerate(restantes, 1):
    t_img = time.time()
    try:
        img_array = preprocess(str(img_path))
        H, W      = img_array.shape[:2]

        raw = reader.readtext(img_array, batch_size=1)

        zones, yolo_lines = [], []
        texte_ar, texte_fr, dosage = [], [], None

        for (bbox, text, conf) in raw:
            if conf < CONFIDENCE_MIN or not text.strip():
                continue
            lang = detect_language(text)
            d    = extract_dosage(text)
            if d: dosage = d

            zones.append({
                "texte":     text.strip(),
                "langue":    lang,
                "confiance": round(float(conf), 3),
                "bbox":      [[int(p[0]), int(p[1])] for p in bbox],
                "dosage":    d
            })
            (texte_ar if "AR" in lang else texte_fr).append(text.strip())
            x_c, y_c, bw, bh = bbox_to_yolo(bbox, W, H)
            yolo_lines.append(f"0 {x_c} {y_c} {bw} {bh}")

        # Sauvegarder YOLO
        (Path(LABELS_DIR) / f"{img_path.stem}.txt").write_text(
            "\n".join(yolo_lines), encoding="utf-8"
        )

        resultats[img_path.name] = {
            "image_id":   img_path.name,
            "image_path": str(img_path),
            "dimensions": {"largeur": W, "hauteur": H},
            "nb_zones":   len(zones),
            "texte_fr":   " | ".join(texte_fr),
            "texte_ar":   " | ".join(texte_ar),
            "dosage":     dosage,
            "zones":      zones,
            "timestamp":  datetime.now().isoformat()
        }

        # Calcul ETA dynamique (moyenne glissante sur 20 dernières images)
        duree = time.time() - t_img
        temps_par_image.append(duree)
        if len(temps_par_image) > 20:
            temps_par_image.pop(0)
        moy     = sum(temps_par_image) / len(temps_par_image)
        eta     = int(moy * (total_r - i))
        eta_str = f"{eta//3600}h{(eta%3600)//60}m" if eta > 3600 else f"{eta//60}m{eta%60}s"

        print(f"  [{i:>4}/{total_r}] ✓ {img_path.name:<32}"
              f"| {len(zones)} zones "
              f"| {duree:.1f}s/img "
              f"| ETA: {eta_str}")

        # Sauvegarde intermédiaire
        if i % SAVE_EVERY == 0:
            save_json(TEMP_JSON, resultats)
            print(f"\n  💾 Sauvegarde ({len(resultats)} images) → {TEMP_JSON}\n")

    except Exception as e:
        erreurs.append({"image": img_path.name, "erreur": str(e)})
        print(f"  [{i:>4}/{total_r}] ✗ {img_path.name} — {e}")


# ══════════════════════════════════════════════════════════════
#  ÉTAPE 5 — Sauvegardes finales
# ══════════════════════════════════════════════════════════════
save_json(FINAL_JSON, resultats)

# CSV — fusionner anciennes + nouvelles lignes
fieldnames = ["image_id","nom_commercial","DCI","dosage_ocr",
              "texte_fr","texte_ar","nb_zones","langue","qualite"]
lignes_old = []
if os.path.exists(FINAL_CSV):
    with open(FINAL_CSV, encoding="utf-8-sig") as f:
        lignes_old = list(csv.DictReader(f))
ids_old = {r["image_id"] for r in lignes_old}

lignes_new = []
for nom, ann in resultats.items():
    if nom not in ids_old:
        lignes_new.append({
            "image_id":       nom,
            "nom_commercial": "",
            "DCI":            "",
            "dosage_ocr":     ann.get("dosage") or "",
            "texte_fr":       ann.get("texte_fr","")[:80],
            "texte_ar":       ann.get("texte_ar","")[:80],
            "nb_zones":       ann.get("nb_zones", 0),
            "langue":         ("AR+FR" if ann.get("texte_ar") and ann.get("texte_fr")
                               else "AR" if ann.get("texte_ar") else "FR"),
            "qualite":        "bonne" if ann.get("nb_zones",0) >= 3 else "faible",
        })

with open(FINAL_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(lignes_old + lignes_new)

save_json(ERRORS_JSON, erreurs)

# Nettoyer temp
if os.path.exists(TEMP_JSON):
    os.remove(TEMP_JSON)

duration = time.time() - t_start
print(f"\n{'═'*60}")
print(f"  ✅ Terminé en {duration/3600:.1f}h")
print(f"  📊 Total annoté  : {len(resultats)}/{len(toutes)} images")
print(f"  🆕 Cette session : {total_r - len(erreurs)} images")
print(f"  ❌ Erreurs       : {len(erreurs)}")
print(f"  ⚡ Vitesse moy.  : {duration/max(total_r,1):.1f}s / image")
print(f"{'═'*60}\n")
