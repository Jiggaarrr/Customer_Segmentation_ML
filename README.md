# Customer Segmentation Dashboard

## Required folder structure

customer_segmentation/
├── app.py
├── Dockerfile
├── requirements.txt
├── data/
│   └── customer_segments.csv
└── models/
    ├── scaler.pkl
    ├── kmeans.pkl
    └── pca.pkl

## Run locally

pip install -r requirements.txt
streamlit run app.py

## Run with Docker

docker build -t customer-segmentation .
docker run -p 8501:8501 customer-segmentation

Then open:
http://localhost:8501

IMPORTANT:
The original notebook created `customer_segments.csv` and the three `.pkl` files.
The dataset file was not included in the uploaded files, so you must copy your
existing `customer_segments.csv` into the `data/` folder before running Docker.
