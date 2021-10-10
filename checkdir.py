import os
from matplotlib.pyplot import draw_if_interactive
import numpy
from sklearn.preprocessing import normalize
test = numpy.random.rand(1000,10)
norm2 = normalize(test, axis=1)
path = os.getcwd()+"/recordings"
# print(path)
trainlabel = numpy.zeros((3000,1))
dic = os.listdir(path)
label = {}
dir = sorted(dic)
for i in dir:
    label[i] = i[0]
label = numpy.array([int(i) for i in list(label.values())])
numpy.savetxt("label.txt", label)
# import scipy.io.wavfile

# features = numpy.zeros((3000,90))
# # 原始信号
# for I,J in enumerate(dir):

#     A = []
#     B = []
#     C = []
#     sample_rate, signal = scipy.io.wavfile.read("recordings/"+ J)  # File assumed to be in the same directory

#     # 预加重
#     pre_emphasis = 0.97
#     emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

#     # 分帧
#     num_frames = 30
#     signal_length = len(emphasized_signal)
#     frame_step = int(numpy.ceil(signal_length/31))
#     frame_length = frame_step * 2
#     signal_length = len(emphasized_signal)


#     pad_signal_length = num_frames * frame_step + frame_length
#     z = numpy.zeros((pad_signal_length - signal_length))
#     pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

#     indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
#     frames = pad_signal[indices.astype(numpy.int32, copy=False)]


#     # 加窗
#     frames *= numpy.hamming(frame_length)

#     # 短时能量与短时平均振幅
#     short_power = numpy.sum(frames**2, axis=1)
#     short_power = short_power.reshape(-1)
#     short_power = short_power / numpy.linalg.norm(short_power)

#     short_amplitude = numpy.sum(numpy.abs(frames), axis=1)
#     short_amplitude = short_amplitude.reshape(-1)
#     short_amplitude = short_amplitude / numpy.linalg.norm(short_amplitude)
#     # 短时过零率
#     zero_count = numpy.zeros(0)
#     for i in range(frames.shape[0]):
#         count = 0
#         for j in range(1, frames.shape[1]):
#             count = count + 0.5*abs(numpy.sign(frames[i][j])-numpy.sign(frames[i][j-1]))
#         zero_count = numpy.append(zero_count, count/frames.shape[1])


#     # 双端点检测
#     sum = 0
#     short_powerAverage = 0
#     for en in short_power :
#         sum = sum + en
#     short_powerAverage = sum / len(short_power)

#     sum = 0
#     for en in short_power[:5] :
#         sum = sum + en
#     ML = sum / 5                        
#     MH = short_powerAverage / 4              #较高的能量阈值
#     ML = (ML + MH) / 4    #较低的能量阈值
#     sum = 0
#     for zcr in zero_count[:5] :
#         sum = float(sum) + zcr             
#     Zs = sum / 5                     #过零率阈值


#     # 首先利用较大能量阈值 MH 进行初步检测
#     flag = 0
#     for i in range(len(short_power)):
#         if len(A) == 0 and flag == 0 and short_power[i] > MH :
#             A.append(i)
#             flag = 1
#         elif flag == 0 and short_power[i] > MH and i - 21 > A[len(A) - 1]:
#             A.append(i)
#             flag = 1
#         elif flag == 0 and short_power[i] > MH and i - 21 <= A[len(A) - 1]:
#             A = A[:len(A) - 1]
#             flag = 1

#         if flag == 1 and short_power[i] < MH :
#             A.append(i)
#             flag = 0

#     # 利用较小能量阈值 ML 进行第二步能量检测
#     for j in range(len(A)) :
#         i = A[j]
#         if j % 2 == 1 :
#             while i < len(short_power) and short_power[i] > ML :
#                 i = i + 1
#             B.append(i)
#         else :
#             while i > 0 and short_power[i] > ML :
#                 i = i - 1
#             B.append(i)

#     # 利用过零率进行最后一步检测
#     for j in range(len(B)) :
#         i = B[j]
#         if j % 2 == 1 :
#             while i < len(zero_count) and zero_count[i] >= 3 * Zs :
#                 i = i + 1
#             C.append(i)
#         else :
#             while i > 0 and zero_count[i] >= 3 * Zs :
#                 i = i - 1
#             C.append(i)

#     for i in range(C[0]):
#         short_power[i] = 0
#         short_amplitude[i] = 0
#         zero_count[i] = 0
#     if len(C)>1:
#         for i in range(C[1], len(short_power)):
#             short_power[i] = 0
#             short_amplitude[i] = 0
#             zero_count[i] = 0

#     # print(short_amplitude,short_power,zero_count)
#     features[I] = numpy.concatenate((short_amplitude,short_power,zero_count),axis = 0)
#     print(features[I])
# numpy.savetxt("features.txt", features) 
