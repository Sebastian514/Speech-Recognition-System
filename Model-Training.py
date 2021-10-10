import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
# Importing Data
path = "/home/lichengjun/lza/dsp实验/Speech-Recognition-System/Spoken-Digit-Recognition/Dataset"
files = librosa.util.find_files(path, ext=['wav']) 
files = np.asarray(files)

# Converting to Values
data = []
x = []
y = []

for file in files: 
    # print(file)
    linear_data, _ = librosa.load(file, sr = 16000, mono = True)   
    data.append(linear_data)
    y.append(file[86])          #Depends on Path of files
print(y)
print("....................")
# Extracting Features
for i in range(len(data)):
    x.append(abs(librosa.stft(data[i]).mean(axis = 1).T))
    
x = np.array(x)
x = x.reshape(x.shape[0], x.shape[1], 1)

# Encoding Categorical Features
from sklearn.preprocessing import OneHotEncoder
y = np.array(y)
y = y.reshape(-1,1)
print(y)
enc = OneHotEncoder()
y = enc.fit_transform(y).toarray()

# Splitting Dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25)
# tensorflow.squeeze(y_train)
# tensorflow.squeeze(y_test)
# print(x_train.shape , x_test.shape, y_train.shape ,y_test.shape)
# Building CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Dropout

classifier = Sequential()
classifier.add(Convolution1D(filters = 128, kernel_size = 6, activation = 'relu', input_shape = (1025,1)))
classifier.add(Convolution1D(filters = 128, kernel_size = 6, activation = 'relu'))
classifier.add(MaxPooling1D(pool_size = 4))
classifier.add(Dropout(0.5))
classifier.add(Convolution1D(filters = 128, kernel_size = 6, activation = 'relu'))
classifier.add(Convolution1D(filters = 128, kernel_size = 6, activation = 'relu'))
classifier.add(MaxPooling1D(pool_size = 4))
classifier.add(Dropout(0.5))
classifier.add(Flatten())
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.summary()
# Fitting and Predicting
classifier.fit(x_train, y_train, epochs=30, batch_size=16, validation_split=0.1)
y_pred = classifier.predict(x_test)
_, accuracy = classifier.evaluate(x_test, y_test, batch_size=16)
y_pred = (y_pred > 0.5)
y_test = (y_test == 1)
print(accuracy)

# Saving model
classifier.save('model.h5')

