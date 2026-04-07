"""
feature_extractor.py - Hybrid LBP + HOG feature extraction
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from skimage.exposure import rescale_intensity
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts a hybrid feature vector from a preprocessed ear ROI.

    Feature vector = [ LBP histogram | HOG descriptor ]
    """

    def __init__(
        self,
        # LBP params
        lbp_radius: int = 3,
        lbp_n_points: int = 24,
        lbp_method: str = "uniform",
        # HOG params
        hog_orientations: int = 9,
        hog_pixels_per_cell: tuple = (8, 8),
        hog_cells_per_block: tuple = (2, 2),
    ):
        self.lbp_radius         = lbp_radius
        self.lbp_n_points       = lbp_n_points
        self.lbp_method         = lbp_method
        self.hog_orientations   = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block

        # Pre-compute number of LBP bins for 'uniform'
        self._lbp_bins = lbp_n_points + 2  # uniform encoding

    # ─── LBP ──────────────────────────────────────────────────────────────────

    def extract_lbp(self, img: np.ndarray) -> np.ndarray:
        """
        Compute LBP histogram over the whole image.
        Returns a normalised histogram vector.
        """
        lbp = local_binary_pattern(
            img,
            P=self.lbp_n_points,
            R=self.lbp_radius,
            method=self.lbp_method,
        )
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=self._lbp_bins,
            range=(0, self._lbp_bins),
        )
        # Normalize
        hist = hist.astype(float)
        norm = hist.sum()
        if norm > 0:
            hist /= norm
        return hist

    def extract_lbp_image(self, img: np.ndarray) -> np.ndarray:
        """Return the raw LBP image (for visualisation)."""
        lbp = local_binary_pattern(
            img, P=self.lbp_n_points, R=self.lbp_radius, method=self.lbp_method
        )
        # Scale to 0-255 for display
        lbp_vis = rescale_intensity(lbp, out_range=(0, 255)).astype(np.uint8)
        return lbp_vis

    # ─── HOG ──────────────────────────────────────────────────────────────────

    def extract_hog(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute HOG descriptor.
        Returns (feature_vector, hog_visualisation_image).
        """
        feat, hog_vis = hog(
            img,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            visualize=True,
            block_norm="L2-Hys",
        )
        hog_vis = rescale_intensity(hog_vis, out_range=(0, 255)).astype(np.uint8)
        return feat, hog_vis

    # ─── Hybrid ───────────────────────────────────────────────────────────────

    def extract(self, img: np.ndarray) -> np.ndarray:
        """
        Full hybrid feature vector: [LBP hist | HOG descriptor].
        img must be a preprocessed grayscale image (128×128).
        """
        lbp_feat           = self.extract_lbp(img)
        hog_feat, _        = self.extract_hog(img)
        feature_vector     = np.concatenate([lbp_feat, hog_feat])
        return feature_vector.astype(np.float32)

    def extract_with_visuals(self, img: np.ndarray) -> dict:
        """
        Extract features and also return visualisation images.

        Returns
        -------
        dict with keys:
            'feature_vector' : np.ndarray
            'lbp_image'      : np.ndarray (uint8)
            'hog_image'      : np.ndarray (uint8)
            'lbp_hist'       : np.ndarray
            'hog_feat'       : np.ndarray
        """
        lbp_hist  = self.extract_lbp(img)
        lbp_image = self.extract_lbp_image(img)
        hog_feat, hog_image = self.extract_hog(img)
        feature_vector = np.concatenate([lbp_hist, hog_feat]).astype(np.float32)

        return {
            "feature_vector": feature_vector,
            "lbp_image":      lbp_image,
            "hog_image":      hog_image,
            "lbp_hist":       lbp_hist,
            "hog_feat":       hog_feat,
        }

    # ─── Info ─────────────────────────────────────────────────────────────────

    @property
    def feature_length(self) -> int:
        """Total length of the hybrid feature vector."""
        # HOG: depends on image size; we compute dynamically
        dummy = np.zeros((128, 128), dtype=np.uint8)
        return len(self.extract(dummy))

    def __repr__(self):
        return (
            f"FeatureExtractor("
            f"lbp_radius={self.lbp_radius}, lbp_n_points={self.lbp_n_points}, "
            f"hog_orientations={self.hog_orientations}, "
            f"hog_pixels_per_cell={self.hog_pixels_per_cell})"
        )
