"""
OCR Service - Text extraction from medication box images.
Supports Arabic, French, and Latin text using Tesseract + EasyOCR.
"""
import re
import logging
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    raw_text: str
    cleaned_text: str
    tokens: List[str]
    confidence: float
    engine: str
    language_detected: Optional[str] = None


class OCRService:
    """
    Multi-engine OCR service.

    Primary engine: Tesseract (fast, good for printed text)
    Fallback engine: EasyOCR (better for complex layouts)
    """

    def __init__(self):
        self._tesseract_available = self._check_tesseract()
        self._easyocr_reader = None  # lazy init

    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Extract text from a preprocessed image.
        Tries Tesseract first, then EasyOCR as fallback.
        """
        result = None

        if self._tesseract_available:
            result = self._tesseract_ocr(image)

        if result is None or len(result.raw_text.strip()) < 3:
            result = self._easyocr_extract(image)

        if result is None:
            result = OCRResult(
                raw_text="", cleaned_text="", tokens=[],
                confidence=0.0, engine="none"
            )

        return result

    def extract_text_multi(self, images: list) -> OCRResult:
        """
        Run OCR on multiple preprocessed versions of the same image
        and merge results for better coverage.
        """
        all_texts = []
        best_confidence = 0.0
        best_engine = "none"

        for name, img in images:
            try:
                res = self.extract_text(img)
                if res.raw_text.strip():
                    all_texts.append(res.raw_text)
                    if res.confidence > best_confidence:
                        best_confidence = res.confidence
                        best_engine = res.engine
            except Exception as e:
                logger.debug(f"OCR failed for {name} variant: {e}")

        # Merge unique tokens from all versions
        combined = " ".join(all_texts)
        cleaned = self._clean_text(combined)
        tokens = self._tokenize(cleaned)

        return OCRResult(
            raw_text=combined,
            cleaned_text=cleaned,
            tokens=tokens,
            confidence=best_confidence,
            engine=f"multi_{best_engine}"
        )

    # ─── Private ────────────────────────────────────────────────────────────

    def _check_tesseract(self) -> bool:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            logger.warning("Tesseract not available. Will use EasyOCR.")
            return False

    def _tesseract_ocr(self, image: np.ndarray) -> Optional[OCRResult]:
        try:
            import pytesseract
            from pytesseract import Output

            # Try French + Arabic + Latin
            lang_configs = [
                ("fra+ara", "--oem 3 --psm 3"),
                ("fra+eng", "--oem 3 --psm 3"),
                ("eng", "--oem 3 --psm 3"),
            ]

            best_text = ""
            best_conf = 0.0

            for langs, config in lang_configs:
                try:
                    data = pytesseract.image_to_data(
                        image, lang=langs, config=config,
                        output_type=Output.DICT
                    )
                    words = [w for w, c in zip(data["text"], data["conf"])
                             if w.strip() and int(c) > 20]
                    conf_vals = [int(c) for c in data["conf"]
                                 if str(c).lstrip("-").isdigit() and int(c) > 0]
                    avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0

                    text = " ".join(words)
                    if len(text) > len(best_text):
                        best_text = text
                        best_conf = avg_conf / 100
                except Exception:
                    continue

            if not best_text.strip():
                return None

            cleaned = self._clean_text(best_text)
            return OCRResult(
                raw_text=best_text,
                cleaned_text=cleaned,
                tokens=self._tokenize(cleaned),
                confidence=best_conf,
                engine="tesseract"
            )
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return None

    def _easyocr_extract(self, image: np.ndarray) -> Optional[OCRResult]:
        try:
            import easyocr
            if self._easyocr_reader is None:
                self._easyocr_reader = easyocr.Reader(
                    ["fr", "ar", "en"],
                    gpu=False,
                    verbose=False
                )

            results = self._easyocr_reader.readtext(image)
            texts = [text for (_, text, conf) in results if conf > 0.2]
            confs = [conf for (_, _, conf) in results if conf > 0.2]

            raw = " ".join(texts)
            cleaned = self._clean_text(raw)
            avg_conf = sum(confs) / len(confs) if confs else 0.0

            return OCRResult(
                raw_text=raw,
                cleaned_text=cleaned,
                tokens=self._tokenize(cleaned),
                confidence=avg_conf,
                engine="easyocr"
            )
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """Normalize OCR output."""
        if not text:
            return ""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove non-printable chars
        text = re.sub(r"[^\w\s\-./()°%+àâäéèêëîïôöùûüçæœ\u0600-\u06FF]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text).strip()
        return text.upper()

    def _tokenize(self, text: str) -> List[str]:
        """Split cleaned text into meaningful tokens."""
        if not text:
            return []
        tokens = re.split(r"[\s\-./()]+", text)
        # Filter out very short tokens (likely noise) and numbers-only
        tokens = [t for t in tokens if len(t) >= 2]
        return tokens
