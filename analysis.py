import streamlit as st 
import os
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import io
import librosa
import torch
import torch.nn.functional as F
import torchaudio
from scipy.io.wavfile import read
from lstm import predict_genre
from tensorflow.keras.models import load_model
import pandas as pd
import plotly.express as px

def load_audio(audiopath, sampling_rate=22000):
    if isinstance(audiopath, str): # If the input is a file path
        if audiopath.endswith('.mp3'):
            audio, lsr = librosa.load(audiopath, sr=sampling_rate)
            audio = torch.FloatTensor(audio)
        elif audiopath.endswith('.wav'):
            audio, sr = read(audiopath)
            if sr != sampling_rate:
                audio = librosa.resample(np.float32(audio), sr, sampling_rate)
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
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"]) # Accepts both .mp3 and .wav
    user_classification = st.selectbox("Select the classification for the audio:", ["", "Real", "Spoof"])
    if uploaded_file is not None and user_classification:
        if st.button("Analyze Audio"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("Your results are below")
                audio_clip = load_audio(uploaded_file)
                result = classify_audio_clip(audio_clip)
                result = result.item()
                path = "sounds/" + uploaded_file.name
                result_lstm = lstm_a(path)
                result_cnn = cnn_a(path)
                st.info(f"LSTM result: {result_lstm}")
                st.info(f"CNN result: {result_cnn}")
                st.info(f"Result Probability: {result}")
                st.success(f"The uploaded audio is {result * 100:.2f}% likely to be AI Generated.")
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
                    
                # Determine the result (Spoof or Real)
                if result * 100 > 5:
                    final_result = "Spoof"
                else:
                    final_result = "Real"
                
                # Create DataFrame with new data
                df = pd.DataFrame({
                    'Uploaded File Name': [uploaded_file.name],
                    'LSTM Result': [result_lstm],
                    'CNN Result': [result_cnn],
                    'TTS Result': [final_result],  # Append the final result to the 'Result' column
                    'User Classification': [user_classification]  # Add user's classification
                })
                
                # If file exists, append new data; otherwise, create a new file
                if os.path.exists('analysis_results.xlsx'):
                    existing_data = pd.read_excel('analysis_results.xlsx')
                    combined_data = pd.concat([existing_data, df], ignore_index=True)
                    combined_data.to_excel('analysis_results.xlsx', index=False)
                else:
                    df.to_excel('analysis_results.xlsx', index=False)
                
            with col2:
                st.info("Your uploaded audio is below")
                st.audio(uploaded_file)
                fig = px.line()

                audio_data = audio_clip.squeeze().numpy()
                fig.add_scatter(x=list(range(len(audio_data))), y=audio_data)
                fig.update_layout(
                    title="Waveform Plot",
                    xaxis_title="Time",
                    yaxis_title="Amplitude"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col3:
                st.info("Disclaimer")
                st.warning("These classification or detection mechanisms are not always accurate. They should be considered as a strong signal and not the ultimate decision makers.")

if __name__ == "__main__":
    main()
