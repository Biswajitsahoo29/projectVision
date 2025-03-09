# projectoverview

This project predicts heart disease risk (binary classification) using the Cleveland heart disease dataset. We train two models:

XGBoost (gradient boosting)
Deep Neural Network (DNN) using Keras
Then we combine these models in an ensemble (fusion) for improved performance. Finally, we deploy the solution via Streamlit in Google Colab, using pyngrok to generate a public URL for the app.

Key ML concepts include:
Data cleaning & preprocessing
Feature scaling (StandardScaler)
Model tuning (GridSearchCV)
Ensemble learning (weighted average)
Model evaluation (ROC-AUC, confusion matrix)
Streamlit deployment in Colab

# Dataset
Source: UCI Machine Learning Repository – Heart Disease
File Used: processed.cleveland.data
Features: 13 clinical features + 1 target column
Target: num (binarized: 0 = no disease, 1 = disease)

# Setup & Requirements
Python 3.7+
Packages (see requirements.txt):
streamlit
pyngrok
xgboost
scikit-learn
tensorflow
pandas
numpy
joblib
matplotlib
seaborn

# Colab or Local Environment:
If using Google Colab, install packages via !pip install ....
If local, use pip install -r requirements.txt.

# Running Locally (Colab or Local Python)
**Download/Clone the Repo**
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

**Install Dependencies**
pip install -r requirements.txt

**Run the Notebook**
Colab: Upload HeartDiseasePrediction.ipynb to your Google Drive and open it in Colab.
Local Jupyter:
jupyter notebook HeartDiseasePrediction.ipynb

**Follow the Steps in the Notebook**
Load and preprocess the dataset (processed.cleveland.data).
Train the XGBoost and DNN models.
Evaluate performance using ROC-AUC, confusion matrix, etc.
Ensemble the predictions.
Save the models (best_xgb_model.pkl, dnn_model.h5, scaler.pkl).

**Model Training & Evaluation**
Data Preprocessing:
Convert num > 0 → 1, else 0.
Handle missing values, scale features.
XGBoost:
Tuned via GridSearchCV (e.g., max_depth, n_estimators, learning_rate).
DNN:
Built with Keras (layers + dropout).
EarlyStopping for preventing overfitting.
Ensemble:
Weighted average of probabilities (e.g., 0.5 * xgb_prob + 0.5 * dnn_prob).
Metrics:
ROC-AUC for measuring performance.
Confusion Matrix for classification stats.
Classification Report for precision, recall, F1-score.

**Deployment with Streamlit (Colab + pyngrok)**
**install pyngrok and streamlit in colab**
# This is used to create a public URL for the Streamlit server
!pip install streamlit pyngrok
**Add ngrok AUTHTOKEN**
# Configure ngrok with your token to enable extended usage
!ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
 **Run the app**
from pyngrok import ngrok
import os

# Kill any process using port 8501 (Streamlit's default)
!fuser -k 8501/tcp

# Start ngrok tunnel on port 8501
public_url = ngrok.connect(8501)
print("Your public URL:", public_url)

# Launch the Streamlit app in the background
get_ipython().system_raw("streamlit run app.py &")
