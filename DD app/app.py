import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
# Removed train_test_split and classification_report as they are not needed for prediction serving
from flask import Flask, render_template, request
import os # Import os module to construct path

app = Flask(__name__)

# --- Model Loading and Training ---
# Get the directory where the app.py script is located
base_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the CSV file relative to the script's location
csv_path = os.path.join(base_dir, "C:\\Users\\zafar\\Desktop\\ML projects\\DD app\\depression_social_media_dataset.csv")

model = None
training_error = None

try:
    # Load the dataset [cite: 1]
    print(f"Attempting to load dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print("Dataset loaded successfully.")

    # Prepare data (using the whole dataset for training as in the original script's loop context) [cite: 1]
    X = df['text']
    y = df['label']

    # Create and train the model pipeline [cite: 1]
    model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
    print("Training model...")
    model.fit(X, y)
    print("Model trained successfully.")

except FileNotFoundError:
    training_error = f"Error: Dataset file not found at {csv_path}. Please make sure 'depression_social_media_dataset.csv' is in the same directory as app.py."
    print(training_error)
except Exception as e:
    training_error = f"An error occurred during model training: {e}"
    print(training_error)
# --- End Model Loading and Training ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles both displaying the form and processing the prediction."""
    prediction_result = None
    user_text = ""

    if request.method == 'POST':
        user_text = request.form.get('social_media_post') # Get text from form
        if user_text and model:
            # Make prediction using the trained model [cite: 1]
            prediction = model.predict([user_text])[0]
            prediction_result = f"Predicted Depression Level: {prediction.upper()}" # [cite: 1]
        elif not model:
            prediction_result = training_error # Show training error if model failed to load/train

    # Render the template, passing any prediction or error message
    return render_template('index.html', prediction=prediction_result, error=training_error, submitted_text=user_text)

if __name__ == '__main__':
    app.run(debug=True) # debug=True helps during development