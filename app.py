import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
from flask import Flask, request, jsonify, render_template
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Paths for data and models
data_path = "C:\\Users\\yadav\\Desktop\\audioproject\\archive\\dataset"
model_path = "C:\\Users\\yadav\\Desktop\\audioproject\\model\\mode2l.pkl"
label_path = "C:\\Users\\yadav\\Desktop\\audioproject\\model\\label2_encoder.pkl"

def extract_features(file_path):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=48000, mono=True)

        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).mean(axis=1)  # Extract 40 MFCCs
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr).mean(axis=1)      # Chroma features
        mel = librosa.feature.melspectrogram(y=audio, sr=sr).mean(axis=1)      # Mel Spectrogram
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr).mean(axis=1)  # Spectral contrast

        
        return np.hstack([mfccs, chroma, mel, contrast])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


model = joblib.load(model_path)
label_encoder = joblib.load(label_path)

app = Flask(__name__)

def predict_emotion(file_path):
   
    try:
        # Extract features from the audio file
        features = extract_features(file_path)
        
        # Ensure features are valid
        if features is None:
            return "Error: Unable to extract features from the audio file."
        
        # Check feature size 
        expected_size = 187 
        if features.shape[0] < expected_size:
            features = np.pad(features, (0, expected_size - features.shape[0]), mode='constant')
        elif features.shape[0] > expected_size:
            features = features[:expected_size]

        # Reshape features for prediction
        features = features.reshape(1, -1)

        # Predict the emotion
        predicted_label = model.predict(features)[0]
        predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]

        return predicted_emotion
    except ValueError as ve:
        logging.error(f"ValueError during prediction: {ve}")
        return f"Error processing the file: {str(ve)}"
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return f"Unexpected error: {str(e)}"
    

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])

def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['audio']
    file_path = f"uploaded_files/{audio_file.filename}"
    audio_file.save(file_path)

    emotion = predict_emotion(file_path)
    return jsonify({'emotion': emotion})

if __name__ == "__main__": 
    app.run(debug=True)