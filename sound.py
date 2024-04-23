import sounddevice as sd
import scipy.io.wavfile as wav

def record_and_save_audio(filename, duration=5, samplerate=22000):
    print("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished")
    
    print("Saving audio to", filename)
    wav.write(filename, samplerate, audio_data)

# # # Example usage:
# filename = "temp.wav"
# record_and_save_audio(filename)
