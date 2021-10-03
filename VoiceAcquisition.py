'''语音采集使用.wav格式
    采样率：8k/16k
    位深:16bit
    单声道
'''
import sounddevice as sd
from scipy.io.wavfile import write
fs = 16000
seconds =3
myrecording =sd.rec(int(seconds * fs),samplerate=fs,channels=1)
sd.wait()
write("output.wav",fs,myrecording)


