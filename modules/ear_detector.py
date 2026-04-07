"""
ear_detector.py - Ear detection module
Uses OpenCV Haar Cascade; falls back to full image ROI if cascade not found.
"""

import cv2
import numpy as np
import os
import logging
from modules.utils import preprocess_image, TARGET_SIZE

logger = logging.getLogger(__name__)

# ─── Cascade Paths ────────────────────────────────────────────────────────────
# OpenCV ships haarcascade_mcs_leftear.xml and haarcascade_mcs_rightear.xml
_CV2_DATA = cv2.data.haarcascades  # type: ignore

LEFT_EAR_CASCADE  = os.path.join(_CV2_DATA, "haarcascade_mcs_leftear.xml")
RIGHT_EAR_CASCADE = os.path.join(_CV2_DATA, "haarcascade_mcs_rightear.xml")


class EarDetector:
    """Detects and crops the ear region from an input image."""

    def __init__(self):
        self.left_cascade  = self._load_cascade(LEFT_EAR_CASCADE)
        self.right_cascade = self._load_cascade(RIGHT_EAR_CASCADE)

    # ── Private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_cascade(path: str):
        if os.path.isfile(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                logger.info(f"Loaded cascade: {os.path.basename(path)}")
                return cascade
        logger.warning(f"Cascade not found / empty: {path}")
        return None

    @staticmethod
    def _best_detection(detections: np.ndarray) -> tuple | None:
        """Return the largest bounding box."""
        if detections is None or len(detections) == 0:
            return None
        largest = max(detections, key=lambda r: r[2] * r[3])
        return tuple(largest)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, gray_img: np.ndarray) -> tuple[np.ndarray, tuple | None]:
        """
        Detect ear in *gray_img*.

        Returns
        -------
        roi : np.ndarray
            Preprocessed ear ROI ready for feature extraction.
        bbox : (x, y, w, h) | None
            Bounding box in the original image coordinate space, or None.
        """
        best_bbox = None

        for cascade in [self.left_cascade, self.right_cascade]:
            if cascade is None:
                continue
            detections = cascade.detectMultiScale(
                gray_img,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            bbox = self._best_detection(detections)
            if bbox is not None:
                # Pick the first successful detection
                best_bbox = bbox
                break

        if best_bbox is not None:
            x, y, w, h = best_bbox
            # Add small padding (10 %)
            pad_x = int(w * 0.10)
            pad_y = int(h * 0.10)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(gray_img.shape[1], x + w + pad_x)
            y2 = min(gray_img.shape[0], y + h + pad_y)
            roi = gray_img[y1:y2, x1:x2]
        else:
            # Fallback: use the whole image as ROI
            logger.debug("No ear detected; using full image as ROI.")
            roi = gray_img
            best_bbox = None

        roi = preprocess_image(roi, target_size=TARGET_SIZE)
        return roi, best_bbox

    def draw_detection(self, color_img: np.ndarray, bbox: tuple | None) -> np.ndarray:
        """Draw bounding box on a colour image for visualisation."""
        vis = color_img.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 150), 2)
            cv2.putText(vis, "Ear", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 150), 1)
        return vis
