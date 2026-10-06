# Customer Segmentation Dashboard

A simple customer segmentation project that uses **K-Means clustering** to group customers based on their income, spending, purchase frequency, and average order value.

The project includes an interactive **Streamlit dashboard** where you can explore the customer segments, view their characteristics, visualize the clusters, and assign a new customer to a segment.

## What this project does

The project takes customer data and groups similar customers together using K-Means clustering.

The dashboard lets you:

- View the total number of customers
- See how many customer segments were created
- Check the average spending
- Compare the size of each cluster
- View the average characteristics of each customer segment
- Visualize the clusters using PCA
- Enter details for a new customer and predict their cluster

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Docker

## Project Structure

```text
customer-segmentation/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
│
├── data/
│   └── customer_segments.csv
│
└── models/
    ├── scaler.pkl
    ├── kmeans.pkl
    └── pca.pkl
```

## How it works

The application uses four customer features:

- **Income**
- **Spending**
- **Purchase Frequency**
- **Average Order Value**

The trained K-Means model assigns customers to different clusters based on these features.

Before prediction, the new customer's data is transformed using the saved scaler. The trained K-Means model then predicts the appropriate customer segment.

PCA is also used to reduce the customer data to two dimensions so that the clusters can be visualized on the dashboard.

## Run the project locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit app

```bash
streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

## Run with Docker

Build the Docker image:

```bash
docker build -t customer-segmentation .
```

Run the container:

```bash
docker run -p 8501:8501 customer-segmentation
```

Then open:

```text
http://localhost:8501
```

## Models

The `models` folder contains the trained components used by the application:

- `scaler.pkl` — used to scale customer features
- `kmeans.pkl` — trained K-Means clustering model
- `pca.pkl` — PCA model used for visualization

These files are loaded directly by the Streamlit application when it starts.

## Dataset

The processed customer dataset is stored in:

```text
data/customer_segments.csv
```

The application expects this file to be present because it loads the dataset directly when the dashboard starts.

## Why I built this

I built this project to understand how an unsupervised machine learning model can be turned into a usable application rather than keeping the work limited to a notebook.

It combines a basic machine learning workflow with model persistence, data visualization, a Streamlit interface, and Docker deployment.

## Future Improvements

Some possible improvements would be:

- Add more customer features
- Allow users to upload their own dataset
- Add automatic selection of the optimal number of clusters
- Add more detailed segment insights
- Deploy the application publicly
- Add model evaluation and monitoring

## Author

**Jigar Vishwakarma**

Computer Engineering Student | AI/ML Enthusiast
