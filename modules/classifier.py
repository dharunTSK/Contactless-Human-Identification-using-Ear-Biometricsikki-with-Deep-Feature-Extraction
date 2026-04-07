"""
classifier.py - SVM-based ear biometric classifier
"""

import os
import numpy as np
import joblib
import logging
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "ear_svm_model.pkl"
)


class EarClassifier:
    """
    Wraps a sklearn Pipeline (StandardScaler + SVM-RBF) for ear identification.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path  = model_path
        self.pipeline: Pipeline | None = None
        self.class_names: list[str]    = []
        self.trained: bool             = False

    # ─── Build ────────────────────────────────────────────────────────────────

    def _build_pipeline(self, C: float = 10.0, gamma: str | float = "scale") -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=C, gamma=gamma,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            )),
        ])

    # ─── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        class_names: list[str],
        grid_search: bool = True,
        progress_callback=None,
    ) -> dict:
        """
        Train SVM classifier.

        Parameters
        ----------
        X_train, y_train : training data
        class_names      : list of class label strings
        grid_search      : run GridSearchCV to tune C & gamma
        progress_callback: callable(message: str)

        Returns
        -------
        dict with 'best_params', 'cv_accuracy', 'train_accuracy'
        """
        self.class_names = class_names

        def _cb(msg):
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        _cb("Building SVM pipeline …")

        if grid_search and len(X_train) >= 20:
            _cb("Running Grid Search for optimal C & γ …")
            param_grid = {
                "svm__C":     [0.1, 1, 10, 100],
                "svm__gamma": ["scale", "auto", 0.001, 0.01],
            }
            base_pipe = self._build_pipeline()
            gs = GridSearchCV(
                base_pipe, param_grid,
                cv=min(5, len(np.unique(y_train))),
                scoring="accuracy",
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X_train, y_train)
            self.pipeline   = gs.best_estimator_
            best_params     = gs.best_params_
            cv_accuracy     = gs.best_score_
            _cb(f"Best params: {best_params}  CV accuracy: {cv_accuracy:.4f}")
        else:
            _cb("Training SVM with default params …")
            self.pipeline = self._build_pipeline()
            self.pipeline.fit(X_train, y_train)
            best_params   = {}
            cv_accuracy   = cross_val_score(
                self.pipeline, X_train, y_train,
                cv=min(5, len(np.unique(y_train))), scoring="accuracy",
            ).mean()

        train_preds    = self.pipeline.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_preds)
        self.trained   = True
        _cb(f"Training complete. Train accuracy: {train_accuracy:.4f}")

        return {
            "best_params":    best_params,
            "cv_accuracy":    cv_accuracy,
            "train_accuracy": train_accuracy,
        }

    # ─── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate on test data and return a metrics dict."""
        if not self.trained or self.pipeline is None:
            raise RuntimeError("Model is not trained yet.")

        y_pred   = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report   = classification_report(
            y_test, y_pred, target_names=self.class_names, output_dict=True
        )
        cm       = confusion_matrix(y_test, y_pred)

        return {
            "accuracy":  accuracy,
            "report":    report,
            "confusion_matrix": cm,
            "y_pred":    y_pred,
            "y_test":    y_test,
        }

    # ─── Prediction ───────────────────────────────────────────────────────────

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float]:
        """
        Predict identity of a single feature vector.

        Returns
        -------
        (label: str, confidence: float)
        """
        if not self.trained or self.pipeline is None:
            raise RuntimeError("Model is not trained yet. Please train first.")

        fv = feature_vector.reshape(1, -1)
        label_idx   = self.pipeline.predict(fv)[0]
        proba       = self.pipeline.predict_proba(fv)[0]
        confidence  = float(proba.max())

        label = (
            self.class_names[label_idx]
            if label_idx < len(self.class_names)
            else f"Person_{label_idx}"
        )
        return label, confidence

    def predict_top_k(self, feature_vector: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k predictions with probabilities."""
        if not self.trained or self.pipeline is None:
            raise RuntimeError("Model is not trained yet.")

        fv    = feature_vector.reshape(1, -1)
        proba = self.pipeline.predict_proba(fv)[0]
        top_k = np.argsort(proba)[::-1][:k]

        results = []
        for idx in top_k:
            name = (
                self.class_names[idx]
                if idx < len(self.class_names)
                else f"Person_{idx}"
            )
            results.append((name, float(proba[idx])))
        return results

    # ─── Persist ──────────────────────────────────────────────────────────────

    def save(self, path: str | None = None):
        """Save trained model to disk."""
        target = path or self.model_path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        payload = {
            "pipeline":    self.pipeline,
            "class_names": self.class_names,
        }
        joblib.dump(payload, target)
        logger.info(f"Model saved → {target}")

    def load(self, path: str | None = None) -> bool:
        """Load a previously saved model. Returns True on success."""
        target = path or self.model_path
        if not os.path.isfile(target):
            logger.warning(f"Model file not found: {target}")
            return False
        try:
            payload          = joblib.load(target)
            self.pipeline    = payload["pipeline"]
            self.class_names = payload["class_names"]
            self.trained     = True
            logger.info(f"Model loaded ← {target}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    # ─── Visualisation ────────────────────────────────────────────────────────

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str | None = None) -> plt.Figure:
        """Return a matplotlib Figure of the confusion matrix."""
        fig, ax = plt.subplots(figsize=(max(6, len(self.class_names)), max(5, len(self.class_names) - 1)))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")

        ax.set_title("Confusion Matrix", color="white", fontsize=14, pad=12)
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        plt.xticks(rotation=45, ha="right", color="white", fontsize=8)
        plt.yticks(color="white", fontsize=8)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        return fig

    def plot_accuracy_bar(self, metrics: dict, save_path: str | None = None) -> plt.Figure:
        """Bar chart of per-class precision, recall, f1."""
        report = metrics.get("report", {})
        classes = [c for c in report if c not in ("accuracy", "macro avg", "weighted avg")]
        precision = [report[c]["precision"] for c in classes]
        recall    = [report[c]["recall"]    for c in classes]
        f1        = [report[c]["f1-score"]  for c in classes]

        x = np.arange(len(classes))
        w = 0.25

        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.8), 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        ax.bar(x - w, precision, w, label="Precision", color="#4cc9f0")
        ax.bar(x,     recall,    w, label="Recall",    color="#f72585")
        ax.bar(x + w, f1,        w, label="F1-Score",  color="#7209b7")

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha="right", color="white", fontsize=8)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_ylim(0, 1.1)
        ax.tick_params(colors="white")
        ax.set_title("Per-Class Metrics", color="white", fontsize=14)
        ax.legend(facecolor="#1a1a2e", labelcolor="white")
        ax.spines[:].set_color("#333366")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
        return fig
