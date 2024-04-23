import logging
import os

# Set TensorFlow log level to suppress unnecessary messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2', '3'}

# Configure logging to suppress other messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

import librosa
import numpy as np
from tensorflow.keras.models import load_model


def predict_genre(model, audio_file_path, genre_mapping):

    # Load audio file
    signal, sample_rate = librosa.load(audio_file_path, sr=22050)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T

    # MFCC'leri uygun boyuta getir
    mfcc = np.resize(mfcc, (130, 13, 1))

    # Reshape MFCC'leri uygun boyuta
    mfcc = mfcc[np.newaxis, ...]

    # Predict using the model
    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction, axis=1)

    # Map predicted index to genre label
    genre_label = genre_mapping[predicted_index[0]]
    # print("Raw prediction:", prediction)

    return genre_label

# Load your trained model
model_path = r"models\audio_lstm.h5"
model = load_model(model_path)

# Path to the audio file you want to predict
audio_file_path = r"sounds\rutvik.wav" # real


# Genre mapping (update this according to your dataset)
genre_mapping = {0: "spoof", 1: "real"}

# Make the prediction
predicted_genre = predict_genre(model, audio_file_path, genre_mapping)
# print("Predicted genre:", predicted_genre)


