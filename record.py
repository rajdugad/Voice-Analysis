import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import io
import librosa
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.io.wavfile import read
from lstm import predict_genre
from tensorflow.keras.models import load_model
from sound import record_and_save_audio
import plotly.express as px

def load_audio(audiopath, sampling_rate=22000):
    if isinstance(audiopath, str): # If the input is a file path
        if audiopath.endswith('.mp3'):
            audio, lsr = librosa.load(audiopath, sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        elif audiopath.endswith('.wav'):
            sr, audio = read(audiopath)
            if sr != sampling_rate:
                audio = librosa.resample(audio.astype(np.float32), sr, sampling_rate)
            audio = torch.FloatTensor(audio)
        else:
            assert False, f"Unsupported audio format provided: {audiopath[-4:]}"
    elif isinstance(audiopath, io.BytesIO): # If the input is file content
        audio, lsr = torchaudio.load(audiopath)
        audio = audio[0] # Remove any channel data
        if lsr != sampling_rate:
            audio = torchaudio.functional.resample(audio, lsr, sampling_rate)
    if torch.any(audio > 2) or not torch.any(audio < 0):
        print(f"Error with audio data. Max={audio.max()} min={audio.min()}")
    audio = torch.clamp(audio, -1, 1)
    return audio.unsqueeze(0)



#function for classifier
def classify_audio_clip(clip):
    """
    Returns whether or not the classifier thinks the given clip came from AI generation.
    :param clip: torch tensor containing audio waveform data (get it from load_audio)
    :return: The probability of the audio clip being AI-generated.
    
    """
    classifier = AudioMiniEncoderWithClassifierHead (2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    clip = clip.cpu().unsqueeze (0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]

def lstm_a(audio_file_path):
    # Load your trained model
    model_path = r"models\audio_lstm.h5"
    model = load_model(model_path)
    # Genre mapping (update this according to your dataset)
    genre_mapping = {0: "Spoof", 1: "Real"}
    # Make the prediction
    predicted_genre = predict_genre(model, audio_file_path, genre_mapping)
    return predicted_genre

def cnn_a(audio_file_path):
    # Load your trained model
    model_path = r"models\cnn_audio.h5"
    model = load_model(model_path)
    # Genre mapping (update this according to your dataset)
    genre_mapping = {0: "Spoof", 1: "Real"}
    # Make the prediction
    predicted_genre = predict_genre(model, audio_file_path, genre_mapping)
    return predicted_genre

st.set_page_config(layout="wide")

def main():
    st.title("AI-Generated Voice Detection")
    
    if st.button("Start Recording"):
        st.text("Recording...")
        record_and_save_audio(r"temp.wav")
        st.text("Recording finished")
        
        audio_path = r"temp.wav"
        audio_clip = load_audio(audio_path)
        result = classify_audio_clip(audio_clip)
        result = result.item()
        result_lstm = lstm_a(audio_path)
        result_cnn = cnn_a(audio_path)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("Your results are below")
            st.info(f"LSTM result: {result_lstm}")
            st.info(f"CNN result: {result_cnn}")
            st.info(f"Result Probability: {result}")
            st.success(f"The recorded audio is {result * 100:.2f}% likely to be AI Generated.")
            if result * 100 > 5:
                st.success("Spoof")
            else:
                st.success("Real")
            if result_lstm == "Spoof":
                st.success("LSTM: Spoof")
            else:
                st.success("LSTM: Real")
            if result_cnn == "Spoof":
                st.success("CNN: Spoof")
            else:
                st.success("CNN: Real")
                
        with col2:
            st.info("Your recorded audio is below")
            st.audio("temp.wav")
                   
        with col3:
            st.info("Disclaimer")
            st.warning("These classification or detection mechanisms are not always accurate. They should be considered as a strong signal and not the ultimate decision makers.")


if __name__ == "__main__":
    main()