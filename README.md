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
- Tesseract ocr
**Features**
- Interactive web form to input a text post
- Real-time prediction display
- Error handling for model training issues

**Usage**
1. Ensure `depression_social_media_dataset.csv` is placed in the same directory as `app.py`.
2. Run the app:

**bash**
   python app.py

### Hindi-English Translator Web Application

A bilingual translation web application that translates text between Hindi and English using pre-trained transformer models and a basic dictionary-based lookup method. Built with Streamlit and powered by Hugging Face models and datasets.

---

##  Features

- Translate text between **Hindi ↔ English**
- Choose between:
  - **High-Accuracy Neural Translation** (Helsinki-NLP Transformer Models)
  - **Basic Lookup Translation** (Dictionary-based)
- View random dataset samples and translation outputs
- Simple and responsive user interface using **Streamlit**

---

##  Technologies Used

| Technology           | Purpose                                      |
|----------------------|----------------------------------------------|
| **Streamlit**        | Web UI Framework                             |
| **Hugging Face Transformers** | Pre-trained translation models         |
| **Hugging Face Datasets**     | Hindi-English sentence pairs          |
| **Pandas**           | Dataset handling and manipulation            |
| **PyTorch**          | Backend support for Transformers             |

---

##  Project Workflow & Implementation

### Step 1: Initial Model Training (Failed Attempt)
- Custom small dataset collected manually
- Custom transformer model trained
- **Issue:** Overfitting due to limited data size

### Step 2: Improved Dataset
- Switched to `cfilt/iitb-english-hindi` dataset (20,000+ samples)
- Hugging Face’s `Trainer` API used
- **Issue:** Training too slow and resource-intensive

### Step 3: Optimized Solution (Final Approach)
- Adopted **Helsinki-NLP's** pre-trained models:
  - `opus-mt-en-hi`
  - `opus-mt-hi-en`
- Loaded using Hugging Face’s `pipeline()` for fast inference

### Step 4: UI Development
- Built with **Streamlit**
- Features:
  - Dropdown for language selection
  - Radio buttons to choose translation mode
  - Text input/output areas
  - Validation messages

### Step 5: Dictionary Lookup Mode
- Created basic word-to-word dictionaries
- Implemented as fallback or educational demonstration mode

---

##  Fixes & Error Handling

| Issue                          | Resolution                                      |
|--------------------------------|-------------------------------------------------|
| `PyTorch` not installed        | Installed manually                             |
| Streamlit backend errors       | Fixed via cache clear and decorator correction |
| Model overfitting              | Controlled by data sampling                    |
| `label_visibility` error       | Resolved by updating Streamlit version         |
| Slow epoch training            | Replaced with pre-trained models               |

---

##  Getting Started

### Prerequisites

Install required packages:

***bash***
pip install streamlit pandas datasets transformers torch

### Run the Application

streamlit run TTNEW.py

Acknowledgments
  1.Helsinki-NLP

  2.Hugging Face Datasets

  3.Hugging Face Transformers

  4.CFILT IIT-Bombay (IITB)

  5.Scikit-learn

  6.Google Tesseract OCR

###Author
Zafar Farhaan Z.H
Email: zfarhaan68.com
GitHub: github.com/zafarfarhaan07
LinkedIn: https://in.linkedin.com/in/zafar-farhaan
