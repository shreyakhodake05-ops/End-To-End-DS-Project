# End-To-End-DS-Project
Developed an end-to-end machine learning pipeline for Airbnb price prediction including data preprocessing, model training, and API deployment using FastAPI

## 🚀 Features
- Data Cleaning & Preprocessing
- Feature Engineering
- Model Training (Random Forest)
- Model Deployment using FastAPI
- API Testing using Swagger UI

## 📂 Dataset
- Airbnb listings dataset

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI

## ▶️ How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run API
uvicorn api:app --reload

### 3. Open Swagger UI
http://127.0.0.1:8000/docs

## 📊 Sample Output
```json
{
  "prediction": 31668.44
}
