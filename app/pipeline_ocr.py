# ══════════════════════════════════════════════════════════════
#  PHASE 2 — PIPELINE OCR COMPLET
#  Reconnaissance de boîtes de médicaments
#  Compatible : photo smartphone + image PC
#
#  Usage dans Colab :
#    from pipeline_ocr import PipelineOCR
#    pipeline = PipelineOCR()
#    result   = pipeline.traiter("photo_boite.jpg")
#    print(result)
# ══════════════════════════════════════════════════════════════

import os, re, json, time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import easyocr
except ImportError:
    raise ImportError("pip install easyocr")

CONFIG = {
    "max_size":       1024,
    "confidence_min": 0.35,
    "gpu":            True,
}


# ── Utilitaires ───────────────────────────────────────────────

def detect_language(text):
    arabic = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    ratio  = arabic / max(len(text), 1)
    if ratio > 0.4:  return "AR"
    if ratio > 0.1:  return "AR+FR"
    return "FR"

def extract_dosage(text):
    m = re.search(r"(\d+\.?\d*)\s*(mg|ml|g|mcg|UI|%)", text, re.IGNORECASE)
    return m.group(0).strip() if m else None

def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2)


# ══════════════════════════════════════════════════════════════
#  MODULE 1 — CHARGEMENT IMAGE
# ══════════════════════════════════════════════════════════════

class ImageLoader:
    FORMATS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.heic']

    @staticmethod
    def depuis_fichier(chemin):
        ext = Path(chemin).suffix.lower()
        if ext == '.heic':
            try:
                import pyheif
                from PIL import Image as PILImage
                heif = pyheif.read(chemin)
                pil  = PILImage.frombytes(heif.mode, heif.size, heif.data,
                                          "raw", heif.mode, heif.stride)
                return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except ImportError:
                raise ImportError("pip install pyheif Pillow")
        img = cv2.imread(str(chemin))
        if img is None:
            raise ValueError(f"Impossible de lire : {chemin}")
        return img

    @staticmethod
    def depuis_upload_colab():
        from google.colab import files
        import io
        from PIL import Image as PILImage
        print("Selectionne une image de boite de medicament...")
        uploaded = files.upload()
        if not uploaded:
            raise ValueError("Aucun fichier uploade.")
        nom     = list(uploaded.keys())[0]
        pil     = PILImage.open(io.BytesIO(uploaded[nom])).convert("RGB")
        img     = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        print(f"Image chargee : {nom} ({img.shape[1]}x{img.shape[0]}px)")
        return img, nom

    @staticmethod
    def depuis_url(url):
        import urllib.request, tempfile
        ext = Path(url).suffix or '.jpg'
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            urllib.request.urlretrieve(url, tmp.name)
            img = cv2.imread(tmp.name)
        os.unlink(tmp.name)
        if img is None:
            raise ValueError(f"Impossible de charger : {url}")
        return img


# ══════════════════════════════════════════════════════════════
#  MODULE 2 — PRETRAITEMENT
# ══════════════════════════════════════════════════════════════

class Preprocessor:
    def __init__(self, max_size=1024):
        self.max_size = max_size

    def traiter(self, img):
        infos = {"etapes": []}
        # Resize
        h, w  = img.shape[:2]
        ratio = 1.0
        if max(h, w) > self.max_size:
            ratio = self.max_size / max(h, w)
            img   = cv2.resize(img, (int(w*ratio), int(h*ratio)),
                               interpolation=cv2.INTER_AREA)
            infos["etapes"].append(f"resize_{ratio:.2f}x")
        infos["ratio_resize"] = round(ratio, 3)

        # Score flou avant
        score = blur_score(img)
        infos["blur_score_avant"] = score
        infos["niveau_flou"] = (
            "net"    if score >= 80 else
            "leger"  if score >= 50 else
            "modere" if score >= 25 else "fort"
        )

        # CLAHE contraste
        lab     = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l       = cv2.createCLAHE(2.0, (8,8)).apply(l)
        img     = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)
        infos["etapes"].append("clahe")

        # Correction flou
        img, methode = self._corriger_flou(img, score)
        infos["correction_flou"] = methode
        if methode != "aucune":
            infos["etapes"].append(f"deblur_{methode}")
            infos["blur_score_apres"] = blur_score(img)

        infos["dimensions_finales"] = {"largeur": img.shape[1], "hauteur": img.shape[0]}
        return img, infos

    def _corriger_flou(self, img, score):
        if score >= 80:
            return img, "aucune"
        elif score >= 50:
            b = cv2.GaussianBlur(img, (0,0), 2.0)
            return cv2.addWeighted(img, 1.4, b, -0.4, 0), "unsharp_mask"
        elif score >= 25:
            b = cv2.GaussianBlur(img, (0,0), 3.0)
            s = cv2.addWeighted(img, 1.8, b, -0.8, 0)
            k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
            return cv2.filter2D(s, -1, k), "unsharp+kernel"
        else:
            up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            k  = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
            s  = cv2.filter2D(up, -1, k)
            b  = cv2.GaussianBlur(s, (0,0), 2.0)
            s2 = cv2.addWeighted(s, 1.3, b, -0.3, 0)
            return cv2.resize(s2, (img.shape[1], img.shape[0])), "deep_sharpen"


# ══════════════════════════════════════════════════════════════
#  MODULE 3 — OCR MULTILINGUE
# ══════════════════════════════════════════════════════════════

class OCREngine:
    def __init__(self, gpu=True, confidence_min=0.35):
        print("Chargement du modele OCR...")
        self.reader         = easyocr.Reader(["ar","en"], gpu=gpu, verbose=False)
        self.confidence_min = confidence_min
        print("Modele OCR pret")

    def extraire(self, img):
        t0  = time.time()
        raw = self.reader.readtext(img)

        zones, textes_ar, textes_fr = [], [], []
        dosage, conf_total = None, []

        for (bbox, text, conf) in raw:
            if conf < self.confidence_min or not text.strip():
                continue
            text = self._nettoyer(text)
            if not text:
                continue

            lang = detect_language(text)
            d    = extract_dosage(text)
            if d and not dosage:
                dosage = d

            zones.append({
                "texte":     text,
                "langue":    lang,
                "confiance": round(float(conf), 3),
                "bbox":      [[int(p[0]), int(p[1])] for p in bbox],
                "dosage":    d,
            })
            conf_total.append(float(conf))
            (textes_ar if "AR" in lang else textes_fr).append(text)

        return {
            "nb_zones":       len(zones),
            "texte_fr":       " | ".join(textes_fr),
            "texte_ar":       " | ".join(textes_ar),
            "texte_complet":  " ".join(textes_fr + textes_ar),
            "dosage":         dosage,
            "confiance_moy":  round(sum(conf_total)/max(len(conf_total),1), 3),
            "zones":          zones,
            "temps_ocr_ms":   round((time.time()-t0)*1000),
        }

    def _nettoyer(self, text):
        import unicodedata
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[^\w\s\-\./+%\u0600-\u06FF]", "", text)
        return re.sub(r"\s+", " ", text).strip()


# ══════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════

class PipelineOCR:
    def __init__(self):
        self.preprocessor = Preprocessor(CONFIG["max_size"])
        self.ocr          = OCREngine(CONFIG["gpu"], CONFIG["confidence_min"])
        self.loader       = ImageLoader()

    def traiter(self, source):
        """
        source :
          - str/Path  : chemin fichier (PC ou Drive)
          - "upload"  : upload interactif Colab
          - "url:..."  : URL image
          - np.ndarray: image deja chargee
        """
        t_total = time.time()
        nom     = "inconnu"

        if isinstance(source, np.ndarray):
            img = source
        elif isinstance(source, str) and source == "upload":
            img, nom = self.loader.depuis_upload_colab()
        elif isinstance(source, str) and source.startswith("url:"):
            img = self.loader.depuis_url(source[4:])
            nom = source[4:].split("/")[-1]
        else:
            nom = Path(str(source)).name
            img = self.loader.depuis_fichier(str(source))

        h0, w0 = img.shape[:2]
        img_prep, infos_prep = self.preprocessor.traiter(img)
        ocr_res = self.ocr.extraire(img_prep)

        resultat = {
            "fichier":         nom,
            "dimensions_orig": {"largeur": w0, "hauteur": h0},
            "pretraitement":   infos_prep,
            "ocr": {
                "nb_zones":      ocr_res["nb_zones"],
                "texte_fr":      ocr_res["texte_fr"],
                "texte_ar":      ocr_res["texte_ar"],
                "texte_complet": ocr_res["texte_complet"],
                "dosage":        ocr_res["dosage"],
                "confiance_moy": ocr_res["confiance_moy"],
                "zones":         ocr_res["zones"],
                "temps_ms":      ocr_res["temps_ocr_ms"],
            },
            "temps_total_ms":  round((time.time()-t_total)*1000),
            "timestamp":       datetime.now().isoformat(),
            "statut":          "ok" if ocr_res["nb_zones"] > 0 else "vide",
        }
        self._afficher(resultat)
        return resultat

    def traiter_lot(self, dossier, sauvegarder=None):
        """Traite toutes les images d'un dossier."""
        imgs = []
        for ext in ['.jpg','.jpeg','.png','.webp','.bmp']:
            imgs.extend(Path(dossier).rglob(f"*{ext}"))
        imgs = sorted(set(imgs))

        print(f"\nTraitement en lot : {len(imgs)} images")
        resultats = []
        for i, p in enumerate(imgs, 1):
            try:
                r = self.traiter(str(p))
                resultats.append(r)
                print(f"  [{i:>4}/{len(imgs)}] {p.name:<30}"
                      f"| {r['ocr']['nb_zones']} zones | {r['temps_total_ms']}ms")
            except Exception as e:
                print(f"  [{i:>4}/{len(imgs)}] ERREUR {p.name} — {e}")

        if sauvegarder:
            with open(sauvegarder, 'w', encoding='utf-8') as f:
                json.dump(resultats, f, ensure_ascii=False, indent=2)
            print(f"Resultats sauvegardes : {sauvegarder}")
        return resultats

    def _afficher(self, r):
        o = r['ocr']
        p = r['pretraitement']
        print(f"\n{'='*55}")
        print(f"  Fichier    : {r['fichier']}")
        print(f"  Zones OCR  : {o['nb_zones']}")
        print(f"  Texte FR   : {o['texte_fr'][:60] or 'aucun'}")
        print(f"  Texte AR   : {o['texte_ar'][:60] or 'aucun'}")
        print(f"  Dosage     : {o['dosage'] or 'non detecte'}")
        print(f"  Confiance  : {o['confiance_moy']}")
        print(f"  Flou       : {p.get('niveau_flou','?')} (score: {p.get('blur_score_avant','?')})")
        print(f"  Temps      : {r['temps_total_ms']}ms")
        print(f"  Statut     : {r['statut']}")
        print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════
#  TEST RAPIDE
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from google.colab import drive
    drive.mount('/content/drive')

    BASE    = '/content/drive/MyDrive/MasterIT_Project2_ML_DL'
    DATASET = f'{BASE}/MasterIT_Project2_Dataset'

    pipeline = PipelineOCR()

    # Test avec la premiere image du dataset
    imgs = [f for f in os.listdir(DATASET) if f.endswith(('.jpg','.jpeg','.png'))]
    if imgs:
        print(f"\nTest avec : {imgs[0]}")
        res = pipeline.traiter(os.path.join(DATASET, imgs[0]))

    # Pour uploader ta propre photo :
    # res = pipeline.traiter("upload")

    print("Pipeline Phase 2 operationnel !")
    print("Prochaine etape : Phase 3 Matching")
