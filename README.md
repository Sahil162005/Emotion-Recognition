# 🎙️ Emotion Recognition from Speech

This is a Flask-based web application that predicts human emotion from speech audio files. The system uses pre-trained machine learning models to classify emotions such as **Happy**, **Sad**, **Angry**, **Neutral**, and others, based on audio feature extraction (like MFCCs).

 ## 📁 Project Structure
emotion-recognition/
│
├── audio/ # Stores recorded/test audio files
│
├── model/ # Pre-trained model and encoders
│ ├── label_encoder.pkl # Encoder mapping emotion labels
│ └── label2_encoder.pkl # (Optional secondary encoder)
│
├── templates/ # HTML templates (for Flask)
│ ├── index.html # Main UI page
│ └── ab.c # [Can be removed if not used]
│
├── uploaded_files/ # Stores uploaded .wav files temporarily
│
├── app.py # Flask application entry point
├── README.md # Project documentation


## 🚀 Features

- 🎧 Upload `.wav` audio files via web UI
- 📊 Extracts MFCC features from audio
- 🤖 Predicts emotion using a trained model
- 💬 Displays predicted emotion on screen

---

## 🛠️ Tech Stack

- **Frontend**: HTML (Jinja2 templating)
- **Backend**: Python, Flask
- **Audio Processing**: Librosa, SoundFile
- **Modeling**: Scikit-learn / Keras
- **Others**: NumPy, Pandas, Pickle
  ---
 🧠 Model Details
The trained model uses MFCC features extracted from .wav files.

Encoders (label_encoder.pkl and label2_encoder.pkl) are used to transform labels during prediction.

You can update the model or add deep learning support in future iterations.
 🔮 Future Improvements
🎙️ Real-time microphone-based prediction

🌍 Support for multilingual datasets

🧠 Upgrade to CNN/LSTM or transformer-based models

📈 Visual feedback (waveform, spectrograms)
