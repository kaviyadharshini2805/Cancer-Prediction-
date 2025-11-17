# ❤️ KNN Cancer Prediction Web App

A machine learning–powered web application built using K-Nearest Neighbors (KNN) and Streamlit to predict whether a tumor is Benign (B) or Malignant (M) based on medical measurements.

## 📌 Features

🧪 Predict cancer diagnosis using a trained KNN Classifier

🔬 Uses the popular Breast Cancer Wisconsin Dataset

📊 Dataset cleaned & scaled using StandardScaler

🧠 Model trained using scikit-learn

🌐 Interactive UI built with Streamlit

💾 Model and scaler saved using Joblib

⚡ Tuned K-value for best accuracy

## 📁 Project Structure
KNN-Cancer-Prediction/

│

  ├── knn_train_model.py                 # Model training script (KNN)
  
  ├── streamlit_knn_cancer_prediction.py # Streamlit prediction web app
  
  ├── cancer_data.csv                    # Dataset
  
  ├── knn_cancer_model.pkl               # Saved KNN model
  
  ├── knn_scaler.pkl                     # Saved StandardScaler
  
  └── README.md                          # Documentation

## 🧠 Machine Learning Model

This project uses:

K-Nearest Neighbors Classifier (KNN)

StandardScaler for feature scaling

Train-test split: 80-20

Automatic K-value tuning (1 to 20)

Model evaluation includes:

✔ Accuracy Score

✔ Confusion Matrix

✔ Classification Report

# ▶️ How to Run the Project
## 1️⃣ Install Dependencies
pip install -r requirements.txt


Or install manually:

pip install pandas numpy scikit-learn streamlit joblib

## 2️⃣ Train the Model (optional but recommended)
python knn_train_model.py


This will generate:

knn_cancer_model.pkl

knn_scaler.pkl

## 3️⃣ Run the Streamlit Web App
streamlit run streamlit_knn_cancer_prediction.py


Your app will open in the browser automatically 🎉

## 📊 Dataset Information

The dataset contains medical tumor characteristics such as:

Radius

Texture

Perimeter

Area

Smoothness

Compactness

Concavity

Symmetry

Fractal Dimension

Worst / Standard Error values

Target variable:

diagnosis →

B = Benign

M = Malignant

## 🚀 Future Improvements

Add more ML models (Random Forest, SVM, Logistic Regression)

Add data visualizations inside Streamlit

Deploy on cloud:

Streamlit Community Cloud

Render

HuggingFace Spaces

Add K-value slider for interactive tuning
