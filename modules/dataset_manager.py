"""
dataset_manager.py - Dataset loading, augmentation, train/test split
"""

import os
import numpy as np
import logging
from sklearn.model_selection import train_test_split

from modules.utils import (
    load_image, preprocess_image, augment_image,
    get_class_folders, list_image_files, TARGET_SIZE,
)
from modules.ear_detector import EarDetector
from modules.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Walks a dataset folder (sub-folder per person) and builds
    feature matrices ready for sklearn classifiers.

    Expected layout
    ---------------
    dataset_root/
        Person_01/  img1.jpg  img2.jpg ...
        Person_02/  img1.jpg ...
        ...
    """

    def __init__(
        self,
        dataset_root: str,
        detector: EarDetector,
        extractor: FeatureExtractor,
        augment: bool = True,
        test_size: float = 0.25,
        random_state: int = 42,
    ):
        self.dataset_root  = dataset_root
        self.detector      = detector
        self.extractor     = extractor
        self.augment       = augment
        self.test_size     = test_size
        self.random_state  = random_state

        self.class_names: list[str] = []
        self.X_train: np.ndarray | None = None
        self.X_test:  np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test:  np.ndarray | None = None

        self._total_images   = 0
        self._loaded_images  = 0

    # ─── Private ──────────────────────────────────────────────────────────────

    def _process_image(self, img_path: str) -> np.ndarray | None:
        """Load → detect ear → extract features for one image."""
        gray = load_image(img_path, grayscale=True)
        if gray is None:
            return None
        roi, _ = self.detector.detect(gray)
        feat   = self.extractor.extract(roi)
        return feat

    # ─── Public API ───────────────────────────────────────────────────────────

    def scan(self) -> dict:
        """Scan dataset root and return summary statistics."""
        folders = get_class_folders(self.dataset_root)
        stats   = {}
        for folder in folders:
            path   = os.path.join(self.dataset_root, folder)
            files  = list_image_files(path)
            stats[folder] = len(files)
        return stats

    def load(self, progress_callback=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Load and process the entire dataset.

        Parameters
        ----------
        progress_callback : callable(current, total) | None

        Returns
        -------
        X : (N, D) feature matrix
        y : (N,)  label array (integers)
        """
        folders = get_class_folders(self.dataset_root)
        if not folders:
            raise FileNotFoundError(
                f"No sub-folders found in dataset root: {self.dataset_root}"
            )

        self.class_names = folders
        X_list, y_list   = [], []

        # Count total for progress
        all_files = []
        for label_idx, folder in enumerate(folders):
            folder_path = os.path.join(self.dataset_root, folder)
            files       = list_image_files(folder_path)
            for f in files:
                all_files.append((f, label_idx))

        self._total_images  = len(all_files)
        self._loaded_images = 0

        for img_path, label_idx in all_files:
            gray = load_image(img_path, grayscale=True)
            if gray is None:
                continue

            roi, _ = self.detector.detect(gray)

            images_to_process = augment_image(roi) if self.augment else [roi]
            for variant in images_to_process:
                feat = self.extractor.extract(variant)
                X_list.append(feat)
                y_list.append(label_idx)

            self._loaded_images += 1
            if progress_callback:
                progress_callback(self._loaded_images, self._total_images)

        if not X_list:
            raise ValueError("No features extracted — check dataset path and images.")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        logger.info(f"Loaded {len(X)} samples, {len(folders)} classes.")
        return X, y

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified train / test split."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def progress(self) -> tuple[int, int]:
        return self._loaded_images, self._total_images
