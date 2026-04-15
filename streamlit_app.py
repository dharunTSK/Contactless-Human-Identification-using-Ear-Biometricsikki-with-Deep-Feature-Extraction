"""
Streamlit web app for the Ear Biometrics identification system.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from modules.classifier import EarClassifier
from modules.dataset_manager import DatasetManager
from modules.ear_detector import EarDetector
from modules.feature_extractor import FeatureExtractor
from modules.utils import confidence_to_label, list_image_files


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = BASE_DIR / "dataset"
REPORTS_DIR = BASE_DIR / "reports"
SNAPSHOT_PATH = REPORTS_DIR / "latest_metrics.json"


def init_page() -> None:
    st.set_page_config(
        page_title="Ear Biometrics ID",
        page_icon="👂",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, #07111f 0%, #11203a 55%, #0ea5e9 100%);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }
        .hero h1, .hero p {
            color: #f8fafc;
            margin: 0;
        }
        .hero p {
            margin-top: 0.45rem;
            color: #cbd5e1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "detector" not in st.session_state:
        st.session_state.detector = EarDetector()
    if "extractor" not in st.session_state:
        st.session_state.extractor = FeatureExtractor()
    if "classifier" not in st.session_state:
        classifier = EarClassifier()
        st.session_state.classifier = classifier
        st.session_state.model_loaded = classifier.load()
    if "training_summary" not in st.session_state:
        st.session_state.training_summary = None
    if "evaluation" not in st.session_state:
        st.session_state.evaluation = None
    if "dataset_rows" not in st.session_state:
        st.session_state.dataset_rows = []
    if "training_log" not in st.session_state:
        st.session_state.training_log = []
    if "last_error" not in st.session_state:
        st.session_state.last_error = None

    snapshot = load_snapshot()
    if snapshot:
        if st.session_state.training_summary is None:
            st.session_state.training_summary = snapshot.get("training_summary")
        if st.session_state.evaluation is None:
            st.session_state.evaluation = snapshot.get("evaluation")
        if not st.session_state.dataset_rows:
            st.session_state.dataset_rows = snapshot.get("dataset_rows", [])


def load_snapshot() -> dict | None:
    if not SNAPSHOT_PATH.is_file():
        return None
    try:
        payload = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    evaluation = payload.get("evaluation")
    if evaluation and "confusion_matrix" in evaluation:
        evaluation["confusion_matrix"] = np.array(evaluation["confusion_matrix"])
    return payload


def save_snapshot(training_summary: dict, evaluation: dict, dataset_rows: list[dict]) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    payload = {
        "training_summary": to_builtin(training_summary),
        "evaluation": to_builtin(evaluation),
        "dataset_rows": to_builtin(dataset_rows),
    }
    SNAPSHOT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def scan_dataset_root(dataset_root: str) -> tuple[list[dict], int]:
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        return [], 0

    rows: list[dict] = []
    total_images = 0
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        count = len(list_image_files(str(folder)))
        total_images += count
        rows.append({"Identity": folder.name, "Images": count})
    return rows, total_images


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def analyse_rgb_image(rgb_image: np.ndarray) -> dict:
    detector: EarDetector = st.session_state.detector
    extractor: FeatureExtractor = st.session_state.extractor
    classifier: EarClassifier = st.session_state.classifier

    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    roi, bbox = detector.detect(gray_image)
    visuals = extractor.extract_with_visuals(roi)
    annotated = detector.draw_detection(bgr_image, bbox)

    result = {
        "annotated_rgb": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        "roi": roi,
        "lbp_image": visuals["lbp_image"],
        "hog_image": visuals["hog_image"],
        "bbox": bbox,
        "prediction": None,
        "top_k": [],
    }

    if classifier.trained:
        label, confidence = classifier.predict(visuals["feature_vector"])
        result["prediction"] = {
            "label": label,
            "confidence": confidence,
            "band": confidence_to_label(confidence),
        }
        result["top_k"] = classifier.predict_top_k(
            visuals["feature_vector"], k=min(3, len(classifier.class_names))
        )

    return result


def render_sidebar(dataset_root: str, dataset_rows: list[dict], total_images: int) -> None:
    classifier: EarClassifier = st.session_state.classifier
    summary = st.session_state.training_summary

    with st.sidebar:
        st.header("Control Panel")
        st.caption("Launch the biometric pipeline from the browser.")

        if classifier.trained:
            st.success("Model ready")
            st.write(f"`{len(classifier.class_names)}` identities loaded")
        else:
            st.warning("No trained model loaded")

        if st.button("Reload saved model", use_container_width=True):
            reloaded = classifier.load()
            st.session_state.model_loaded = reloaded
            if reloaded:
                st.success("Saved model loaded from `models/ear_svm_model.pkl`.")
            else:
                st.error("No saved model was found yet.")

        st.divider()
        st.write("**Dataset**")
        st.caption(dataset_root)
        st.metric("Identities", len(dataset_rows))
        st.metric("Images", total_images)

        if summary:
            st.divider()
            st.write("**Latest training**")
            st.metric("Test accuracy", f"{summary['test_accuracy'] * 100:.2f}%")
            st.metric("Feature dimension", summary["feature_dim"])


def render_dashboard(dataset_rows: list[dict], total_images: int) -> None:
    classifier: EarClassifier = st.session_state.classifier
    summary = st.session_state.training_summary

    st.markdown(
        """
        <div class="hero">
            <h1>Ear Biometrics Identification</h1>
            <p>Web-hosted ear detection, hybrid feature extraction, and SVM recognition powered by Streamlit.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Identities", len(dataset_rows))
    c2.metric("Dataset Images", total_images)
    c3.metric("Model Status", "Ready" if classifier.trained else "Not trained")
    accuracy_value = f"{summary['test_accuracy'] * 100:.2f}%" if summary else "—"
    c4.metric("Latest Accuracy", accuracy_value)

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Pipeline")
        st.markdown(
            """
            1. Detect the ear region with Haar cascades.
            2. Preprocess the crop to `128 x 128` with blur and CLAHE.
            3. Extract `LBP + HOG` hybrid features.
            4. Predict identity with an `RBF SVM` and confidence scores.
            """
        )
    with right:
        st.subheader("Dataset Summary")
        if dataset_rows:
            st.dataframe(dataset_rows, use_container_width=True)
        else:
            st.info("Point the app at a valid dataset folder to inspect its classes.")


def render_train_tab(dataset_root: str) -> None:
    classifier: EarClassifier = st.session_state.classifier

    st.subheader("Train Model")
    st.caption("One subfolder per identity. The trained model is saved to `models/ear_svm_model.pkl`.")

    cols = st.columns(3)
    augment = cols[0].checkbox("Use augmentation", value=True)
    grid_search = cols[1].checkbox("Use grid search", value=True)
    test_size = cols[2].slider("Test split", min_value=0.10, max_value=0.40, value=0.25, step=0.05)

    train_clicked = st.button("Start training", type="primary")
    if not train_clicked:
        if st.session_state.training_log:
            st.code("\n".join(st.session_state.training_log), language="text")
        return

    dataset_path = Path(dataset_root).expanduser()
    if not dataset_path.is_dir():
        st.error("Please provide a valid dataset directory before starting training.")
        return

    progress_bar = st.progress(0.0)
    status_box = st.empty()
    log_box = st.empty()
    logs: list[str] = []

    def push_log(message: str) -> None:
        logs.append(message)
        log_box.code("\n".join(logs), language="text")

    def update_progress(current: int, total: int) -> None:
        fraction = current / max(total, 1)
        progress_bar.progress(min(0.5, fraction * 0.5))
        status_box.info(f"Processing dataset image {current} of {total}")

    try:
        detector = st.session_state.detector
        extractor = st.session_state.extractor
        manager = DatasetManager(
            str(dataset_path),
            detector,
            extractor,
            augment=augment,
            test_size=float(test_size),
        )

        dataset_rows = [{"Identity": key, "Images": value} for key, value in manager.scan().items()]
        total_input_images = sum(row["Images"] for row in dataset_rows)
        push_log(f"Dataset root: {dataset_path}")
        push_log(f"Classes found: {len(dataset_rows)}")
        for row in dataset_rows:
            push_log(f"  {row['Identity']}: {row['Images']} images")

        status_box.info("Loading and preprocessing dataset...")
        X, y = manager.load(progress_callback=update_progress)
        push_log(f"Feature matrix shape: {X.shape}")
        progress_bar.progress(0.55)

        X_train, X_test, y_train, y_test = manager.split(X, y)
        push_log(f"Train samples: {len(X_train)}")
        push_log(f"Test samples: {len(X_test)}")
        status_box.info("Training SVM classifier...")

        training_result = classifier.train(
            X_train,
            y_train,
            manager.class_names,
            grid_search=grid_search,
            progress_callback=push_log,
        )

        progress_bar.progress(0.85)
        status_box.info("Evaluating model...")
        evaluation = classifier.evaluate(X_test, y_test)
        classifier.save()

        REPORTS_DIR.mkdir(exist_ok=True)
        cm_path = REPORTS_DIR / "confusion_matrix.png"
        bars_path = REPORTS_DIR / "per_class_metrics.png"
        fig_cm = classifier.plot_confusion_matrix(evaluation["confusion_matrix"], save_path=str(cm_path))
        plt.close(fig_cm)
        fig_bar = classifier.plot_accuracy_bar(evaluation, save_path=str(bars_path))
        plt.close(fig_bar)

        training_summary = {
            "dataset_root": str(dataset_path),
            "classes": len(manager.class_names),
            "input_images": total_input_images,
            "samples_after_augmentation": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "best_params": training_result["best_params"],
            "cv_accuracy": float(training_result["cv_accuracy"]),
            "train_accuracy": float(training_result["train_accuracy"]),
            "test_accuracy": float(evaluation["accuracy"]),
        }

        st.session_state.training_summary = training_summary
        st.session_state.evaluation = evaluation
        st.session_state.dataset_rows = dataset_rows
        st.session_state.training_log = logs
        st.session_state.last_error = None

        save_snapshot(training_summary, evaluation, dataset_rows)

        progress_bar.progress(1.0)
        status_box.success("Training complete. The saved model is ready for browser-based identification.")
        st.success(f"Test accuracy: {evaluation['accuracy'] * 100:.2f}%")
        st.rerun()
    except Exception as exc:
        st.session_state.training_log = logs
        st.session_state.last_error = str(exc)
        progress_bar.progress(0.0)
        status_box.error(f"Training failed: {exc}")


def render_prediction_result(result: dict, title: str) -> None:
    prediction = result["prediction"]

    if prediction:
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted identity", prediction["label"])
        c2.metric("Confidence", f"{prediction['confidence'] * 100:.2f}%")
        c3.metric("Confidence band", prediction["band"])
    else:
        st.info("No trained model is loaded yet. Detection and feature visualisations are shown below.")

    if result["bbox"] is None:
        st.caption("No ear bounding box was found, so the full image was used as the ROI fallback.")

    st.subheader(title)
    left, right = st.columns(2)
    with left:
        st.image(result["annotated_rgb"], caption="Detected ear region", use_container_width=True)
    with right:
        st.image(result["roi"], caption="Preprocessed ROI", use_container_width=True, clamp=True)

    left, right = st.columns(2)
    with left:
        st.image(result["lbp_image"], caption="LBP texture map", use_container_width=True, clamp=True)
    with right:
        st.image(result["hog_image"], caption="HOG structure map", use_container_width=True, clamp=True)

    if result["top_k"]:
        rows = []
        for index, (label, score) in enumerate(result["top_k"], start=1):
            rows.append({
                "Rank": index,
                "Identity": label,
                "Confidence": f"{score * 100:.2f}%",
            })
        st.write("**Top matches**")
        st.dataframe(rows, use_container_width=True)


def render_identify_tab() -> None:
    st.subheader("Identify Uploaded Image")
    uploaded = st.file_uploader(
        "Upload an ear image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=False,
    )
    if not uploaded:
        st.info("Upload an image to run the ear recognition pipeline.")
        return

    image = Image.open(uploaded)
    result = analyse_rgb_image(pil_to_rgb_array(image))
    render_prediction_result(result, "Detection and Feature Maps")


def render_camera_tab() -> None:
    st.subheader("Identify Camera Capture")
    captured = st.camera_input("Capture an ear image with your webcam")
    if not captured:
        st.info("Capture a frame to run browser-based identification.")
        return

    image = Image.open(captured)
    result = analyse_rgb_image(pil_to_rgb_array(image))
    render_prediction_result(result, "Camera Capture Analysis")


def render_reports_tab() -> None:
    classifier: EarClassifier = st.session_state.classifier
    evaluation = st.session_state.evaluation
    summary = st.session_state.training_summary

    st.subheader("Reports")
    if summary:
        col1, col2, col3 = st.columns(3)
        col1.metric("Cross-validation", f"{summary['cv_accuracy'] * 100:.2f}%")
        col2.metric("Training accuracy", f"{summary['train_accuracy'] * 100:.2f}%")
        col3.metric("Test accuracy", f"{summary['test_accuracy'] * 100:.2f}%")

    if evaluation is None:
        cm_path = REPORTS_DIR / "confusion_matrix.png"
        bars_path = REPORTS_DIR / "per_class_metrics.png"
        if cm_path.is_file() and bars_path.is_file():
            left, right = st.columns(2)
            with left:
                st.image(str(cm_path), caption="Confusion matrix", use_container_width=True)
            with right:
                st.image(str(bars_path), caption="Per-class metrics", use_container_width=True)
            st.info("Saved report images are available, but the underlying metrics were not loaded into this session.")
            return

        st.info("Train a model to generate evaluation reports.")
        return

    fig_cm = classifier.plot_confusion_matrix(evaluation["confusion_matrix"])
    st.pyplot(fig_cm, use_container_width=True)
    plt.close(fig_cm)

    fig_bar = classifier.plot_accuracy_bar(evaluation)
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    report_rows = []
    for class_name in classifier.class_names:
        metrics = evaluation["report"].get(class_name)
        if not metrics:
            continue
        report_rows.append({
            "Identity": class_name,
            "Precision": f"{metrics['precision'] * 100:.2f}%",
            "Recall": f"{metrics['recall'] * 100:.2f}%",
            "F1-score": f"{metrics['f1-score'] * 100:.2f}%",
            "Support": int(metrics["support"]),
        })

    if report_rows:
        st.write("**Per-class report**")
        st.dataframe(report_rows, use_container_width=True)


def main() -> None:
    init_page()
    init_state()

    st.title("Ear Biometrics Streamlit App")

    default_root = st.session_state.training_summary["dataset_root"] if st.session_state.training_summary else str(DEFAULT_DATASET_ROOT)
    dataset_root = st.text_input("Dataset folder", value=default_root)
    dataset_rows, total_images = scan_dataset_root(dataset_root)

    render_sidebar(dataset_root, dataset_rows, total_images)

    dashboard_tab, train_tab, identify_tab, camera_tab, reports_tab = st.tabs(
        ["Dashboard", "Train Model", "Identify Image", "Camera Capture", "Reports"]
    )

    with dashboard_tab:
        render_dashboard(dataset_rows, total_images)
    with train_tab:
        render_train_tab(dataset_root)
    with identify_tab:
        render_identify_tab()
    with camera_tab:
        render_camera_tab()
    with reports_tab:
        render_reports_tab()


if __name__ == "__main__":
    main()
