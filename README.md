# Contactless Human Identification using Ear Biometrics

Ear biometric identification system using Haar-cascade ear detection, hybrid `LBP + HOG` feature extraction, and an `SVM` classifier. The project now includes a Streamlit web app for browser-based training, image identification, camera capture, and reports.

## Features

- Streamlit web interface for training and inference
- Hybrid feature extraction with `LBP + HOG`
- `SVM` classifier with optional grid search
- Dataset browser with per-identity counts
- Uploaded-image and webcam capture identification
- Evaluation reports with confusion matrix and per-class metrics
- Docker support for always-on hosting

## Project Structure

```text
.
├── streamlit_app.py
├── main.py
├── modules/
├── models/
├── dataset/
├── .streamlit/config.toml
├── Dockerfile
└── requirements.txt
```

## Dataset Layout

Place images inside one folder per identity:

```text
dataset/
├── Person_01/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Person_02/
│   ├── img1.jpg
│   └── ...
└── ...
```

Minimum `5` images per identity is recommended.

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The desktop GUI is still available if you want it:

```bash
python main.py
```

## Permanent Hosting Options

### Option 1: Streamlit Community Cloud

Best when you want the simplest public deployment with a stable `streamlit.app` URL.

1. Push this repository to GitHub.
2. Go to `https://share.streamlit.io/`.
3. Create a new app from your repository.
4. Set the entrypoint file to `streamlit_app.py`.
5. Pick a custom subdomain if you want a cleaner public URL.

### Option 2: Always-on container hosting

Best when you need a true always-on app without Streamlit Community Cloud sleep behavior.

Build the image:

```bash
docker build -t ear-biometrics-streamlit .
```

Run the container:

```bash
docker run -p 8501:8501 ear-biometrics-streamlit
```

You can deploy the same container to Render, Railway, Fly.io, a VPS, or any Docker-capable host.

## Streamlit App Tabs

- `Dashboard`: pipeline summary and dataset overview
- `Train Model`: dataset loading, preprocessing, model training, and saving
- `Identify Image`: upload an image and predict identity
- `Camera Capture`: capture from the browser webcam and predict identity
- `Reports`: confusion matrix and per-class precision, recall, and F1

## Model Output

The trained classifier is saved to:

```text
models/ear_svm_model.pkl
```

Evaluation artifacts are saved to:

```text
reports/
├── confusion_matrix.png
├── per_class_metrics.png
└── latest_metrics.json
```

## References

1. Hurley et al. (2005), Force Field Feature Extraction for Ear Biometrics
2. Kumar and Wu (2012), Automated Human Identification using Ear Imaging
3. Emersic et al. (2017), AWE Database
4. Naveena and Hemantha Kumar (2020), HOG + SVM for Ear Classification
5. Wu et al. (2019), HOG + PCA + LBP for Ear Recognition
6. Oyebiyi et al. (2023), Systematic Literature Review on Ear Biometrics
