# Spoken Digit Recognition
Spoken Digit Recognition is a word recognition system that transcribes the individual spoken numbers from 0-9. Along with the recognition system, a real-time tester is implemented that recognizes the spoken digits on the go.

## _Dataset_
The [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) was used to train the model which had 3000 recordings from 6 speakers with english accents. The model was trained on 2250 recordings, and tested on 750 recordings.

## _Model and Training_

The model consists of:
- CNN Layers
- Fully Connected Layer
- Loss Function: Categorical Cross-Entropy
- Optimization Algorithm: Adam

Model is trained on 30 epochs using a batch size of 16

Accuracy: 86%

## _Usage_
- Clone the repository
- Run the Main File to train and test the model: ``` python Model-Training.py ```
- Run the Test File to test the model in real time: ``` python Real-Time-Tester.py ```
