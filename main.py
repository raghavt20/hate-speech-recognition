import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Custom standardization function
def custom_standardisation(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

# Vectorization layer
max_features =  10000
sequence_length =  250
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardisation, max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length
)

# Load the dataset
df = pd.read_csv('hate_speech.csv')

# Preprocess the text data
df['tweet'] = df['tweet'].str.lower()
punctuations_list = string.punctuation

def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

df['tweet'] = df['tweet'].apply(lambda x: remove_punctuations(x))

# Balancing the dataset
class_2 = df[df['class'] ==  2]
class_1 = df[df['class'] ==  1].sample(n=3500)
class_0 = df[df['class'] ==  0]
balanced_df = pd.concat([class_0, class_0, class_0, class_1, class_2], axis=0)

# Splitting the dataset into training and validation sets
features = balanced_df['tweet']
target = balanced_df['class']
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=22)

# One-hot encoding the target variable
Y_train = pd.get_dummies(Y_train)
Y_val = pd.get_dummies(Y_val)

# Training the tokenizer
max_words =  5000
token = Tokenizer(num_words=max_words, lower=True, split=' ')
token.fit_on_texts(X_train)

# Generating token embeddings
training_seq = token.texts_to_sequences(X_train)
training_pad = pad_sequences(training_seq, maxlen=50, padding='post', truncating='post')

testing_seq = token.texts_to_sequences(X_val)
testing_pad = pad_sequences(testing_seq, maxlen=50, padding='post', truncating='post')

# Building the model with custom layer
embedding =  16
model = tf.keras.Sequential([
    layers.Embedding(max_features +  1, embedding),
    layers.Dropout(0.2),
    layers.GlobalAvgPool1D(),
    layers.Dropout(0.2),
    layers.Dense(1),
])
model.summary()

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
