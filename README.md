# WHITE-Z4X
# Machine Learning Mini Projects – 2025

This repository contains 3 beginner-friendly machine learning projects I completed to enhance my understanding of applied ML techniques. Each project solves a real-world problem using Python and popular ML libraries.

---

## Projects Included

### 1. Daily Calorie Calculator

**Description**  
A machine learning-based model that predicts the number of calories a person needs based on personal attributes such as height, weight, age, gender, and exercise level.

**Technologies Used**
- Python (Jupyter Notebook)
- Pandas, NumPy
- Scikit-learn (Linear Regression)
- Matplotlib, Seaborn (for visualization)

**Features**
- Preprocessing of input data
- Regression model training
- Visualization of model accuracy

**Usage**
Open the notebook `daily_calorie_calculator.ipynb` in Jupyter or Google Colab and run the cells sequentially to see model training and predictions.

---

### 2. Depression Detection from Social Media

**Description**  
A sentiment classification model that detects signs of depression from social media posts using NLP and logistic regression.

**Technologies Used**
- Python (Jupyter Notebook)
- Pandas, NumPy
- TfidfVectorizer for text preprocessing
- Scikit-learn (Logistic Regression)
- Matplotlib, Seaborn (EDA)

**Features**
- Labelled dataset analysis
- Text vectorization
- Binary classification: Depressed vs. Not Depressed
- Evaluation metrics and confusion matrix

**Usage**
Run `Depression detection from social media.ipynb` in Jupyter or Colab to train and test the model.

---

### 3. Depression Detection Web App

**Description**  
A simple Flask-based web application that takes a social media text post as input and returns a depression level prediction using a pre-trained model.

**Technologies Used**
- Python
- Flask Web Framework
- Scikit-learn, TfidfVectorizer, Logistic Regression
- HTML (Jinja2 templates)

**Features**
- Interactive web form to input a text post
- Real-time prediction display
- Error handling for model training issues

**Usage**
1. Ensure `depression_social_media_dataset.csv` is placed in the same directory as `app.py`.
2. Run the app:

```bash
python app.py
