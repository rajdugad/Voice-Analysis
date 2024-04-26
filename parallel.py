import streamlit as st 
import os
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import io
import librosa
import plotly.express as px
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.io.wavfile import read
from lstm import predict_genre
from tensorflow.keras.models import load_model
import time
import threading
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# Set TensorFlow log level to suppress unnecessary messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2', '3'}

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


def classify_audio_clip(audio_file_path, results_dict):
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    state_dict = torch.load('classifier.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(state_dict)
    
    # print("Loading audio...")
    # Load audio
    audio, _ = torchaudio.load(audio_file_path)
    audio = audio.unsqueeze(0).cpu()  # Make sure it's on CPU
    
    # print("Classifying audio...")
    # Classify
    start_time = time.time()  # Start measuring execution time
    results = F.softmax(classifier(audio), dim=-1)
    end_time = time.time()  # End measuring execution time
    execution_time = end_time - start_time  # Calculate execution time
    print(f"Classifier execution time: {execution_time}")
    # print(results[0][0].item())
    results_dict['classifier'] = (results[0][0].item(), execution_time)  # Changed index to get probability for 'Real'

def lstm_a(audio_file_path, results_dict):
    # Load your trained model
    model_path = r"models\audio_lstm.h5"
    model = load_model(model_path)
    # Genre mapping (update this according to your dataset)
    genre_mapping = {0: "Spoof", 1: "Real"}
    # Make the prediction
    start_time = time.time()  # Start measuring execution time
    predicted_genre = predict_genre(model, audio_file_path, genre_mapping)
    end_time = time.time()  # End measuring execution time
    execution_time = end_time - start_time  # Calculate execution time
    print(f"LSTM execution time: {execution_time}")
    results_dict['lstm'] = (predicted_genre, execution_time)

def cnn_a(audio_file_path, results_dict):
    # Load your trained model
    model_path = r"models\cnn_audio.h5"
    model = load_model(model_path)
    # Genre mapping (update this according to your dataset)
    genre_mapping = {0: "Spoof", 1: "Real"}
    # Make the prediction
    start_time = time.time()  # Start measuring execution time
    predicted_genre = predict_genre(model, audio_file_path, genre_mapping)
    end_time = time.time()  # End measuring execution time
    execution_time = end_time - start_time  # Calculate execution time
    print(f"CNN execution time: {execution_time}")
    results_dict['cnn'] = (predicted_genre, execution_time)

st.set_page_config(layout="wide")

def main():
    st.title("AI-Generated Voice Detection")
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"]) # Accepts both .mp3 and .wav
    if uploaded_file is not None:
        if st.button("Analyze Audio"):
            start_time = time.time()
            # Create a dictionary to store the results
            results_dict = {}
            audio_clip=load_audio(uploaded_file)
            # Perform classification tasks in parallel using threads
            classifier_thread = threading.Thread(target=classify_audio_clip, args=("sounds/" + uploaded_file.name, results_dict))
            lstm_thread = threading.Thread(target=lstm_a, args=("sounds/" + uploaded_file.name, results_dict))
            cnn_thread = threading.Thread(target=cnn_a, args=("sounds/" + uploaded_file.name, results_dict))
            
            # Start the threads
            classifier_thread.start()
            lstm_thread.start()
            cnn_thread.start()
            
            # Wait for all threads to complete
            classifier_thread.join()
            lstm_thread.join()
            cnn_thread.join()

            # Calculate total execution time
            total_execution_time = time.time() - start_time
            print(f"Total classification execution time: {total_execution_time} seconds")
    

            # Display the results using columns
            col1, col2, col3 = st.columns(3)

            with col1:
                st.info("Your results are below:")
                st.info(f"LSTM result: {results_dict.get('lstm', (None, None))[0]}")
                st.info(f"CNN result: {results_dict.get('cnn', (None, None))[0]}")
                st.info(f"Result Probability: {results_dict.get('classifier', (None, None))[0]}")
                st.success(f"The uploaded audio is {results_dict.get('classifier', (None, None))[0] * 100:.2f}% likely to be AI Generated.")
                if results_dict.get('classifier', (None, None))[0] * 100 > 5:
                    st.success("Spoof")
                else:
                    st.success("Real")
                if results_dict.get('lstm', (None, None))[0] == "Spoof":
                    st.success("LSTM: Spoof")
                else:
                    st.success("LSTM: Real")
                if results_dict.get('cnn', (None, None))[0] == "Spoof":
                    st.success("CNN: Spoof")
                else:
                    st.success("CNN: Real")

            with col2:
                st.info("Your uploaded audio is below:")
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
                st.info("Disclaimer: These classification or detection mechanisms are not always accurate. They should be considered as a strong signal and not the ultimate decision makers.")
                st.write(f"Total classification execution time: {total_execution_time} seconds")
            
if __name__ == "__main__":
    main()
