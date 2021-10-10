import wave
import pyaudio
import librosa
import numpy as np
from tensorflow.keras.models import load_model

class AudioHandler():
    def __init__(self):
        self.FORMAT = pyaudio.paInt32
        self.CHANNELS = 2
        self.RATE = 44100
        self.CHUNK = 1024
        self.record_seconds = 2
        self.audio = None
        self.stream = None
        self.frames = None

    # Starting the Stream
    def start(self):
        print("Recording...")
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.FORMAT,
                                      channels=self.CHANNELS,
                                      rate=self.RATE,
                                      input=True,
                                      output=False,
                                      frames_per_buffer=self.CHUNK)

        self.frames = []
        for i in range(0, int(self.RATE / self.CHUNK * self.record_seconds)):
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)
        print("Finished...")

    # Stopping the Stream
    def stop(self, i):
        self.stream.close()
        self.audio.terminate()

        waveFile = wave.open("file.wav", 'wb')
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        waveFile.setframerate(self.RATE)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()

def Predictor():
    x = []

    linear_data, _ = librosa.load("file.wav", sr=16000, mono=True)
    x.append(abs(librosa.stft(linear_data).mean(axis=1).T))
    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1], 1)   

    model = load_model('model.h5')
    y_pred = model.predict(x)
    print (y_pred)
    print(y_pred.argmax())

if __name__ == '__main__':
    
    val = 0
    while True:
        inp = input("Enter Y to Record, N to Exit: ")
        if inp == 'y' or inp == 'Y':
            audio = AudioHandler()
            audio.start()
            audio.stop(val)
            Predictor()
            val += 1
        else:
            break
        
    
    