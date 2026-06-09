# ══════════════════════════════════════════════════════════════
#  PHASE 3 — MATCHING
#  Identifie un médicament depuis le texte extrait par l'OCR
#
#  Techniques combinées :
#    1. Levenshtein  → corrige les fautes OCR (ex: Spectiom → Spectrum)
#    2. TF-IDF       → similarité sur texte complet
#    3. Dosage boost → bonus si le dosage correspond
#
#  Usage dans Colab :
#    from matching import Matcher
#    matcher = Matcher(csv_path)
#    result  = matcher.identifier(ocr_result)
# ══════════════════════════════════════════════════════════════

import csv, re, os
import numpy as np
from pathlib import Path

# ── Dépendances ───────────────────────────────────────────────
try:
    from rapidfuzz import fuzz, process
except ImportError:
    raise ImportError("pip install rapidfuzz")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    raise ImportError("pip install scikit-learn")


# ══════════════════════════════════════════════════════════════
#  UTILITAIRES
# ══════════════════════════════════════════════════════════════

def normaliser(text: str) -> str:
    """Normalise un texte pour comparaison."""
    if not text:
        return ""
    text = text.lower().strip()
    # Supprimer caractères parasites
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extraire_dosage_norm(text: str) -> str:
    """Extrait et normalise un dosage depuis un texte."""
    if not text:
        return ""
    m = re.search(r"(\d+\.?\d*)\s*(mg|ml|g|mcg|ui|%)", text, re.IGNORECASE)
    if m:
        valeur = m.group(1)
        unite  = m.group(2).lower().replace("ui", "UI")
        return f"{valeur}{unite}"
    return ""


# ══════════════════════════════════════════════════════════════
#  CHARGEMENT DE LA BASE DE RÉFÉRENCE
# ══════════════════════════════════════════════════════════════

class BaseReference:
    def __init__(self, csv_path: str):
        self.medicaments = []
        self._charger(csv_path)
        print(f"✅ Base chargée : {len(self.medicaments)} médicaments")

    def _charger(self, csv_path: str):
        with open(csv_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Construire la liste des variantes pour ce médicament
                variantes = [v.strip() for v in
                             row.get("nom_variantes", "").split("|") if v.strip()]
                variantes.append(row["nom_commercial"])

                self.medicaments.append({
                    "id":             row["id"],
                    "categorie":      row.get("categorie", ""),
                    "nom_commercial": row["nom_commercial"],
                    "nom_ar":         row.get("nom_ar", ""),
                    "DCI":            row.get("DCI", ""),
                    "dosage":         row.get("dosage", ""),
                    "forme":          row.get("forme", ""),
                    "laboratoire":    row.get("laboratoire", ""),
                    "variantes":      variantes,
                    "prix":           row.get("prix", ""), 
                    # Texte complet normalisé pour TF-IDF
                    "texte_index":    normaliser(
                        " ".join([
                            row["nom_commercial"],
                            row.get("DCI", ""),
                            row.get("dosage", ""),
                            row.get("forme", ""),
                            row.get("laboratoire", ""),
                            row.get("nom_variantes", "").replace("|", " "),
                        ])
                    )
                })


# ══════════════════════════════════════════════════════════════
#  MOTEUR DE MATCHING
# ══════════════════════════════════════════════════════════════

class Matcher:

    SEUIL_CONFIANCE = 0.35   # en dessous → "non identifié"

    def __init__(self, csv_path: str):
        self.base = BaseReference(csv_path)
        self._construire_index_tfidf()

    def _construire_index_tfidf(self):
        """Construit l'index TF-IDF sur tous les médicaments."""
        corpus     = [m["texte_index"] for m in self.base.medicaments]
        self.tfidf = TfidfVectorizer(
            analyzer="char_wb",   # n-grammes de caractères → robuste aux fautes
            ngram_range=(2, 4),
            min_df=1
        )
        self.matrix = self.tfidf.fit_transform(corpus)
        print("✅ Index TF-IDF construit")

    # ── Méthode principale ────────────────────────────────────

    def identifier(self, ocr_result: dict) -> dict:
        """
        Identifie le médicament depuis un résultat OCR.
        
        ocr_result : dict sorti par pipeline_ocr.py
          {
            "ocr": {
              "texte_fr": "...",
              "texte_ar": "...",
              "texte_complet": "...",
              "dosage": "500mg"
            }
          }
        Retourne le médicament identifié avec son score.
        """
        # Extraire le texte OCR
        if "ocr" in ocr_result:
            texte_fr      = ocr_result["ocr"].get("texte_fr", "")
            texte_ar      = ocr_result["ocr"].get("texte_ar", "")
            texte_complet = ocr_result["ocr"].get("texte_complet", "")
            dosage_ocr    = ocr_result["ocr"].get("dosage", "") or ""
        else:
            # Compatibilité si on passe directement un dict texte
            texte_fr      = ocr_result.get("texte_fr", "")
            texte_ar      = ocr_result.get("texte_ar", "")
            texte_complet = ocr_result.get("texte_complet",
                                           f"{texte_fr} {texte_ar}")
            dosage_ocr    = ocr_result.get("dosage", "") or ""

        # Score par chaque méthode
        scores_lev   = self._score_levenshtein(texte_complet)
        scores_tfidf = self._score_tfidf(texte_complet)
        scores_ar    = self._score_arabe(texte_ar)

        # Fusion des scores (pondération)
        scores_finaux = []
        for i, med in enumerate(self.base.medicaments):
            score = (
                0.45 * scores_lev[i]   +   # Levenshtein (principal)
                0.35 * scores_tfidf[i] +   # TF-IDF (contexte)
                0.20 * scores_ar[i]        # Arabe (bonus)
            )
            # Bonus dosage (+15% si le dosage correspond)
            if dosage_ocr and med["dosage"]:
                dos_ref = extraire_dosage_norm(med["dosage"])
                dos_ocr = extraire_dosage_norm(dosage_ocr)
                if dos_ref and dos_ocr and dos_ref == dos_ocr:
                    score = min(1.0, score * 1.15)

            scores_finaux.append((score, i))

        # Trier par score décroissant
        scores_finaux.sort(key=lambda x: x[0], reverse=True)

        # Top 3 candidats
        top3 = []
        for score, idx in scores_finaux[:3]:
            med = self.base.medicaments[idx]
            top3.append({
                "nom_commercial": med["nom_commercial"],
                "DCI":            med["DCI"],
                "dosage":         med["dosage"],
                "forme":          med["forme"],
                "laboratoire":    med["laboratoire"],
                "categorie":      med["categorie"],
                "nom_ar":         med["nom_ar"],
                "prix":           med.get("prix", ""),
                "score":          round(score, 3),
            })

        meilleur = top3[0]
        identifie = meilleur["score"] >= self.SEUIL_CONFIANCE

        resultat = {
            "statut":          "identifie" if identifie else "non_identifie",
            "meilleur_match":  meilleur if identifie else None,
            "confiance":       meilleur["score"],
            "niveau_confiance": (
                "haute"  if meilleur["score"] >= 0.70 else
                "moyenne" if meilleur["score"] >= 0.45 else
                "faible"
            ),
            "top3":            top3,
            "texte_ocr_fr":    texte_fr,
            "texte_ocr_ar":    texte_ar,
            "dosage_ocr":      dosage_ocr,
        }

        self._afficher(resultat)
        return resultat

    # ── Méthodes de scoring ───────────────────────────────────

    def _score_levenshtein(self, texte: str) -> list:
        """
        Score Levenshtein entre le texte OCR et les variantes
        de chaque médicament. Robuste aux fautes d'OCR.
        """
        texte_norm = normaliser(texte)
        scores     = []

        for med in self.base.medicaments:
            best = 0.0
            for variante in med["variantes"]:
                # Similarité sur le token le plus ressemblant
                s = fuzz.partial_ratio(
                    normaliser(variante), texte_norm
                ) / 100.0
                if s > best:
                    best = s
            scores.append(best)

        return scores

    def _score_tfidf(self, texte: str) -> list:
        """
        Score TF-IDF cosinus entre le texte OCR et l'index.
        Capture la similarité globale du texte.
        """
        if not texte.strip():
            return [0.0] * len(self.base.medicaments)

        try:
            vec    = self.tfidf.transform([normaliser(texte)])
            scores = cosine_similarity(vec, self.matrix)[0]
            return scores.tolist()
        except Exception:
            return [0.0] * len(self.base.medicaments)

    def _score_arabe(self, texte_ar: str) -> list:
        """
        Score spécifique pour le texte arabe.
        Compare avec les noms arabes de la base.
        """
        if not texte_ar or not texte_ar.strip():
            return [0.0] * len(self.base.medicaments)

        scores = []
        for med in self.base.medicaments:
            if med["nom_ar"]:
                s = fuzz.partial_ratio(
                    normaliser(med["nom_ar"]),
                    normaliser(texte_ar)
                ) / 100.0
            else:
                s = 0.0
            scores.append(s)
        return scores

    # ── Affichage ─────────────────────────────────────────────

    def _afficher(self, r: dict):
        print(f"\n{'═'*55}")
        print(f"  RÉSULTAT MATCHING")
        print(f"{'─'*55}")
        print(f"  Statut     : {r['statut'].upper()}")
        print(f"  Confiance  : {r['confiance']} ({r['niveau_confiance']})")

        if r["meilleur_match"]:
            m = r["meilleur_match"]
            print(f"\n  ✅ Médicament identifié :")
            print(f"     Nom        : {m['nom_commercial']}")
            print(f"     DCI        : {m['DCI']}")
            print(f"     Dosage     : {m['dosage']}")
            print(f"     Forme      : {m['forme']}")
            print(f"     Labo       : {m['laboratoire']}")
            print(f"     Catégorie  : {m['categorie']}")
            print(f"     Nom AR     : {m['nom_ar']}")
        else:
            print(f"\n  ❌ Médicament non identifié")
            print(f"     (score trop faible : {r['confiance']})")

        print(f"\n  Top 3 candidats :")
        for i, c in enumerate(r["top3"], 1):
            print(f"     {i}. {c['nom_commercial']:<20} score: {c['score']}")
        print(f"{'═'*55}\n")

    # ── Évaluation en lot ─────────────────────────────────────

    def evaluer_lot(self, resultats_ocr: list) -> dict:
        """
        Évalue le matching sur un lot de résultats OCR.
        Retourne les métriques globales.
        """
        total      = len(resultats_ocr)
        identifies = 0
        haute_conf = 0

        for res in resultats_ocr:
            match = self.identifier(res)
            if match["statut"] == "identifie":
                identifies += 1
            if match["confiance"] >= 0.70:
                haute_conf += 1

        return {
            "total":             total,
            "identifies":        identifies,
            "taux_identification": round(identifies/max(total,1)*100, 1),
            "haute_confiance":   haute_conf,
            "taux_haute_conf":   round(haute_conf/max(total,1)*100, 1),
        }


# ══════════════════════════════════════════════════════════════
#  TEST RAPIDE
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    CSV_PATH = ('/content/drive/MyDrive/MasterIT_Project2_ML_DL'
                '/reference/medicaments.csv')

    matcher = Matcher(CSV_PATH)

    # ── Test 1 : texte bien reconnu ───────────────────────────
    print("\n TEST 1 — Texte bien reconnu")
    res1 = matcher.identifier({
        "ocr": {
            "texte_fr":      "Spectrum 500mg Ciprofloxacine Cooper Pharma",
            "texte_ar":      "سبيرفلوكساسين",
            "texte_complet": "Spectrum 500mg Ciprofloxacine Cooper Pharma",
            "dosage":        "500mg"
        }
    })

    # ── Test 2 : fautes OCR ───────────────────────────────────
    print("\n TEST 2 — Fautes OCR (Spectiom → Spectrum)")
    res2 = matcher.identifier({
        "ocr": {
            "texte_fr":      "Spectiom 50Omg Cooperr",
            "texte_ar":      "",
            "texte_complet": "Spectiom 50Omg Cooperr",
            "dosage":        "500mg"
        }
    })

    # ── Test 3 : texte partiel ────────────────────────────────
    print("\n TEST 3 — Texte partiel (juste Kintex)")
    res3 = matcher.identifier({
        "ocr": {
            "texte_fr":      "Kintex",
            "texte_ar":      "كينتيكس",
            "texte_complet": "Kintex كينتيكس",
            "dosage":        ""
        }
    })

    # ── Test 4 : médicament inconnu ──────────────────────────
    print("\n TEST 4 — Médicament hors base")
    res4 = matcher.identifier({
        "ocr": {
            "texte_fr":      "Dafalgan 500mg paracetamol",
            "texte_ar":      "",
            "texte_complet": "Dafalgan 500mg paracetamol",
            "dosage":        "500mg"
        }
    })
