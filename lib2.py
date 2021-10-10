import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
from matplotlib import pyplot as plt 



# 原始信号
sample_rate, signal = scipy.io.wavfile.read('/home/lichengjun/lza/dsp实验/Speech-Recognition-System/recordings/0_george_0.wav')  # File assumed to be in the same directory
plt.figure(1)
plt.plot(signal)
plt.savefig('./Signal-in-the-Time-Domain.jpg')


# 预加重
pre_emphasis = 0.97
emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
plt.figure(2)
plt.plot(emphasized_signal)
plt.savefig('./Signal-in-the-Time-Domain-after-Pre-Emphasis.jpg')


# 分帧
num_frames = 30
signal_length = len(emphasized_signal)
frame_step = int(numpy.ceil(signal_length/31))
frame_length = frame_step * 2
signal_length = len(emphasized_signal)


pad_signal_length = num_frames * frame_step + frame_length
z = numpy.zeros((pad_signal_length - signal_length))
pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(numpy.int32, copy=False)]


# 加窗
frames *= numpy.hamming(frame_length)



# FFT和功率谱
NFFT = 512
mag_frames = numpy.absolute(numpy.fft.rfft(frames))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum


# Mel频率倒谱系数
nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = numpy.dot(pow_frames, fbank.T)
filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * numpy.log10(filter_banks)  # dB
plt.plot(filter_banks)
plt.savefig('./Spectrogram-of-the-Signal.jpg')


# MFCC
num_ceps = 12
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

cep_lifter = 22
(nframes, ncoeff) = mfcc.shape
n = numpy.arange(ncoeff)
lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
mfcc *= lift  #*
plt.plot(filter_banks)
plt.savefig('./MFCCs.jpg')


# 均值归一化
filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
plt.plot(filter_banks)
plt.savefig('./Normalized-Filter-Banks.jpg')


mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
plt.plot(filter_banks)
plt.savefig('./Normalized-MFCCs.jpg')