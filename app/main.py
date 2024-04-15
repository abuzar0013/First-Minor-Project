# Import modelutil at the beginning of your script
import numpy as np
import tensorflow as tf
from typing import List
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
import cv2
import os
import modelutil

# Define vocabulary for character mapping
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Create mapping layers for characters to numbers and vice versa
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_data(path: str):
    # Load video frames and alignments
    file_name = os.path.splitext(os.path.basename(path))[0]
    video_path = os.path.join('/Users/abuzarakhtar/Documents/GitHub/LipNet/app/data/s1', f'{file_name}.mpg')
    alignment_path = os.path.join('/Users/abuzarakhtar/Documents/GitHub/LipNet/app/data/s2', f'{file_name}.align')
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)
    return frames, alignments

def load_video(path: str) -> List[float]:
    # Load video frames and preprocess them
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    # Standardize frames
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    scale = tf.cast((frames - mean), tf.float32) / std

    return scale

def load_alignments(path: str) -> List[str]:
    # Load alignment data from file
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = [line.split()[2] for line in lines if line.split()[2] != 'sil']

    # Map characters to numbers
    return char_to_num(tf.strings.unicode_split(tokens, input_encoding='UTF-8'))[1:]

# Example usage
test_path = '/Users/abuzarakhtar/Documents/GitHub/LipNet/app/data/s1/bwis9s.mpg'
frames, alignments = load_data(test_path)

# Display the first preprocessed frame
plt.imshow(frames[0].numpy().squeeze(), cmap='gray')
plt.show()

# Display the alignments
decoded_alignments = ["".join([num_to_char(x.numpy()).numpy().decode() for x in alignment]) for alignment in alignments]
print('actual text is' , decoded_alignments)

# Load the model
model = modelutil.load_model()

# Assuming frames is a single frame with shape (75, 46, 140, 1)
# Reshape and preprocess the frame
frame = tf.reshape(frames, (1, 75, 46, 140, 1))  # Add batch dimension

# Get the predictions for each timestep
predictions = model.predict(frame)

# Decode the predictions for each timestep
decoded_predictions = [[num_to_char(np.argmax(char)).numpy().decode('utf-8') for char in timestep] for timestep in predictions]

# Combine the predictions for each timestep into a single string
decoded_sentences = [''.join(timestep) for timestep in decoded_predictions]

# Print the decoded sentences
print('decoded text is', decoded_sentences)

