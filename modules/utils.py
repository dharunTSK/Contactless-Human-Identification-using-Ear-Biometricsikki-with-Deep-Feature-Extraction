"""
utils.py - Utility functions for the Ear Biometrics System
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_SIZE = (128, 128)   # Resize all ear images to this
SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pgm')


# ─── Image Loading & Preprocessing ────────────────────────────────────────────

def load_image(path: str, grayscale: bool = True) -> np.ndarray | None:
    """Load an image from disk. Returns None on failure."""
    if not os.path.isfile(path):
        logger.warning(f"File not found: {path}")
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    if img is None:
        logger.warning(f"Could not read image: {path}")
    return img


def preprocess_image(img: np.ndarray, target_size=TARGET_SIZE) -> np.ndarray:
    """Resize, denoise, and normalise an ear image."""
    # Resize
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    # Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return img


def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV numpy array (grayscale)."""
    img = np.array(pil_img.convert('L'))
    return img


def cv_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convert OpenCV grayscale numpy array to PIL Image."""
    return Image.fromarray(cv_img)


def cv_to_pil_rgb(cv_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB Image."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ─── Dataset Utilities ────────────────────────────────────────────────────────

def list_image_files(folder: str) -> list[str]:
    """Return all supported image file paths inside a folder."""
    files = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(SUPPORTED_EXTS):
            files.append(os.path.join(folder, fname))
    return sorted(files)


def get_class_folders(dataset_root: str) -> list[str]:
    """Return sub-folder names (each represents one person/class)."""
    return sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ])


# ─── Augmentation ─────────────────────────────────────────────────────────────

def augment_image(img: np.ndarray) -> list[np.ndarray]:
    """
    Apply lightweight augmentations and return a list of variants
    (including the original).
    """
    variants = [img]

    # Horizontal flip
    variants.append(cv2.flip(img, 1))

    # Slight rotation ±10°
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        variants.append(rotated)

    # Brightness shift using PIL
    pil_img = cv_to_pil(img)
    for factor in [0.75, 1.25]:
        bright = ImageEnhance.Brightness(pil_img).enhance(factor)
        variants.append(pil_to_cv(bright))

    return variants


# ─── Scoring Helpers ──────────────────────────────────────────────────────────

def confidence_to_label(confidence: float) -> str:
    """Map a confidence score to a human-readable label."""
    if confidence >= 0.90:
        return "Very High"
    elif confidence >= 0.75:
        return "High"
    elif confidence >= 0.55:
        return "Moderate"
    else:
        return "Low"


def format_accuracy(value: float) -> str:
    return f"{value * 100:.2f}%"
