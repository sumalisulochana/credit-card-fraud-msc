# Credit Card Fraud Detection System

## Overview
This project focuses on identifying potential fraudulent customers using machine learning techniques. The dataset includes customer personal, financial, and demographic information.

## Objectives
- Analyze the dataset and understand patterns
- Build machine learning models for fraud detection
- Improve model performance using appropriate techniques
- Develop an interactive application for prediction

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Shiny for Python
- Git & GitHub

## Models Used
- Logistic Regression (with class balancing and threshold adjustment)
- Random Forest

## Key Features
- Data preprocessing (handling missing values, encoding, scaling)
- Exploratory Data Analysis (EDA)
- Model comparison and evaluation
- Model improvement using threshold adjustment
- Multi-page GUI application for fraud prediction

## Application
The application allows users to:
- View dataset overview
- Explore visualizations
- Input customer details
- Predict fraud risk using selected models

## How to Run the Project

Follow the steps below to run this project on your local machine.

### 1. Prerequisites
Make sure the following software is installed:

- Python (version 3.9 or above)
- Jupyter Notebook or VS Code
- Git (optional, for cloning the repository)

### 2. Install Required Libraries

Open terminal or command prompt and run:

pip install pandas numpy scikit-learn matplotlib seaborn shiny joblib

---

### 3. Run Data Processing

Open the notebook:

01_data_handling.ipynb

Run all cells to:
- Load dataset
- Clean data
- Handle missing values
- Perform encoding and scaling

---

### 4. Run Exploratory Data Analysis (EDA)

Open:

02_eda.ipynb

Run all cells to:
- Generate visualizations
- Analyze fraud distribution
- Understand data patterns

---

### 5. Run Model Building

Open:

03_model_building.ipynb

Run all cells to:
- Train machine learning models
- Evaluate performance
- Apply improvements (threshold adjustment)
- Save models (lr_model.pkl, rf_model.pkl, scaler.pkl)

---

### 6. Run the Application

In terminal, navigate to project folder and run:

shiny run --reload app.py

---

### 7. Use the Application

- Open the app in browser
- Go to Prediction tab
- Enter customer details
- Select model
- Click "Predict" to see fraud risk
