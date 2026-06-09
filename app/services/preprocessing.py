"""
Image preprocessing pipeline for medication box OCR.
Handles noise reduction, contrast enhancement, deskewing, etc.
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocessing pipeline to improve OCR accuracy on medication box images.

    Steps:
    1. Resize if too large
    2. Convert to grayscale
    3. Denoise
    4. Contrast / brightness enhancement
    5. Adaptive thresholding (binarization)
    6. Deskew
    7. Morphological operations to clean up noise
    """

    MAX_DIMENSION = 3000

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Full preprocessing pipeline. Returns OpenCV image (numpy array)."""
        img = self._load_image(image_bytes)
        img = self._resize_if_needed(img)
        img_gray = self._to_grayscale(img)
        img_denoised = self._denoise(img_gray)
        img_enhanced = self._enhance_contrast(img_denoised)
        img_binary = self._binarize(img_enhanced)
        img_deskewed = self._deskew(img_binary)
        img_clean = self._morphological_cleanup(img_deskewed)
        return img_clean

    def preprocess_for_arabic(self, image_bytes: bytes) -> np.ndarray:
        """
        Specialized preprocessing for Arabic text on packaging.
        Arabic text benefits from higher resolution and different thresholding.
        """
        img = self._load_image(image_bytes)
        img = self._resize_if_needed(img, max_dim=4000)
        img_gray = self._to_grayscale(img)
        # Arabic text often needs less aggressive binarization
        img_enhanced = self._enhance_contrast(img_gray, clip_limit=3.0)
        img_binary = self._binarize(img_enhanced, method="otsu")
        return img_binary

    def preprocess_multiple_views(self, image_bytes: bytes) -> list:
        """
        Returns multiple preprocessed versions of the same image.
        Useful for trying different approaches and merging OCR results.
        """
        img = self._load_image(image_bytes)
        img = self._resize_if_needed(img)
        gray = self._to_grayscale(img)

        versions = []

        # Version 1: Standard pipeline
        v1 = self._binarize(self._enhance_contrast(self._denoise(gray)))
        versions.append(("standard", v1))

        # Version 2: Higher contrast
        v2 = self._binarize(self._enhance_contrast(gray, clip_limit=4.0))
        versions.append(("high_contrast", v2))

        # Version 3: Original grayscale (for some cases standard works better)
        versions.append(("grayscale", gray))

        # Version 4: Inverted (dark text on light bg vs light on dark)
        v4 = cv2.bitwise_not(v1)
        versions.append(("inverted", v4))

        return versions

    # ─── Private helpers ────────────────────────────────────────────────────

    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes to OpenCV format."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            # Try via PIL as fallback
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img

    def _resize_if_needed(self, img: np.ndarray, max_dim: int = None) -> np.ndarray:
        max_dim = max_dim or self.MAX_DIMENSION
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        # Upscale if too small (helps OCR)
        if max(h, w) < 800:
            scale = 800 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return img

    def _to_grayscale(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

    def _enhance_contrast(self, img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(img)

    def _binarize(self, img: np.ndarray, method: str = "adaptive") -> np.ndarray:
        if method == "otsu":
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            binary = cv2.adaptiveThreshold(
                img, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )
        return binary

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """Correct slight rotation of the image."""
        try:
            coords = np.column_stack(np.where(img > 0))
            if len(coords) == 0:
                return img
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            # Only deskew if angle is significant
            if abs(angle) > 0.5:
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
        except Exception as e:
            logger.debug(f"Deskew failed (non-critical): {e}")
        return img

    def _morphological_cleanup(self, img: np.ndarray) -> np.ndarray:
        """Remove small noise artifacts using morphological operations."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return img
