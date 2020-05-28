# Iterates through hyperparameters to find most accurate model
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing import sequence
from keras import backend as K
import pickle
import tensorflow as tf
from keras.callbacks import TensorBoard
import time


# Load in dataset
X_train = pickle.load(open("50k_train.pickle", "rb"))
y_train = pickle.load(open("50k_label.pickle", "rb"))
labels_test = pickle.load(open("new_gen_label.pickle", "rb"))
test = pickle.load(open("new_gen_test.pickle", "rb"))


# define various hyperparameter combinations
dense_layers = [1,2,3]
layer_sizes = [32,64,100,128, 196]
embedding_sizes = [32, 64, 128, 196]
vocab_content = 62000
word_lim = 500


# iterate through the various hyperparameter combinations

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for embedding_size in embedding_sizes:
            NAME = "no_keras-{}-embed-{}-nodes-{}-dense-{}".format(embedding_size,layer_size, dense_layer, int(time.time()))
            print(NAME)
            model = Sequential()
            model.add(Embedding(vocab_content, embedding_size, input_length=word_lim))
            model.add(LSTM(layer_size))
            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('sigmoid'))
            model.add(Dense(1, activation = 'sigmoid'))
            
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )
            batch_size = 64
            num_epochs = 4
            
            X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
            X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]

            model.fit(X_train2, y_train2,
                      validation_data=(X_valid, y_valid),
                      batch_size=batch_size,
                      epochs=num_epochs,
                      callbacks=[tensorboard])
            
