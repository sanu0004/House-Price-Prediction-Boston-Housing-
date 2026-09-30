# 🏠 House Price Prediction (Boston Housing)

A deep learning regression model built with TensorFlow/Keras to predict
Boston house prices from neighborhood and property features.

## 📌 Overview
This project uses the Boston Housing dataset (506 samples, 13 input features)
to train a neural network that predicts the median home value (in $1000s).

## 📂 Project Structure
├── Predict_Price_of_the_House_Exercise.ipynb   # Main notebook
└── README.md

## 📊 Dataset
Loaded directly from `tensorflow.keras.datasets.boston_housing`.
Features: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT.
Target: median value of owner-occupied homes.

## 🔍 Workflow
1. Load the dataset and convert to a Pandas DataFrame
2. Standardize features using the training mean and standard deviation (test data uses training statistics to avoid leakage)
3. Build a Sequential neural network: Dense(64, ReLU) → Dense(64, ReLU) → Dense(1)
4. Compile with the Adam optimizer, MSE loss, and MAE metric
5. Train for 100 epochs with a 20% validation split
6. Evaluate on test data and compare predictions with actual prices

## 📈 Results
- Test Mean Absolute Error: about 2.92 (roughly $2,920 average error)

## 🛠️ Tech Stack
Python · TensorFlow / Keras · Pandas · NumPy · Jupyter / Google Colab

## ▶️ How to Run
1. Clone the repo
2. Install dependencies:
   pip install tensorflow pandas numpy jupyter
3. Run the notebook:
   jupyter notebook Predict_Price_of_the_House_Exercise.ipynb

## 🚀 Future Scope
- Plot training vs validation loss curves
- Add early stopping, dropout, or regularization
- Compare against Linear Regression, Random Forest, and XGBoost
- Add R² and RMSE metrics and hyperparameter tuning
