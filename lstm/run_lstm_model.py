from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from keras.preprocessing import sequence
from keras import backend as K
import pickle
import tensorflow as tf
from keras.callbacks import TensorBoard
import csv


vocab_to_word_dict = pickle.load(open("vocab_to_word_dict", "rb"))

def create_ints(tokens):
    test_ints = []
    for line in tokens:
        try:
            test_ints.append(vocab_to_word_dict[line])
        except:
            test_ints.append(0)
    return test_ints


def split_word_vectors(text_in):
    length = 500
    chunks = []
    padded = []
    if len(text_in) > length:
        chunks = [text_in[x: x + 500] for x in range(0, len(text_in), 500)]
    else:
        chunks.append(text_in)
    chunks = sequence.pad_sequences(chunks, maxlen = 500)
    return chunks


def average_sentiment(text_in):
    loaded_model = tf.keras.models.load_model("no_keras_model.h5")
    unsplit_vector = create_ints(text_in)
    if len(unsplit_vector) == 1 and unsplit_vector[0] == 0:
        print('none')
        return ''
    vectors_in = split_word_vectors(unsplit_vector)
    polarities = loaded_model.predict(vectors_in)
    if len(polarities) > 1:
        sum_pol = 0
        for polarity in polarities:
            sum_pol += polarity[0]
        averaged_polarity = sum_pol / len(polarities)
        print(averaged_polarity)
        return averaged_polarity
    if len(polarities) == 1:
        print(polarities[0][0])
        return polarities[0][0]
    
    
def all_sentiments(file_in):
    loaded_model = tf.keras.models.load_model("no_keras_model.h5")
    counter = 1
    for book in file_in:
        unsplit_vector = create_ints(book)
        vectors_in = split_word_vectors(unsplit_vector)
        polarities = loaded_model.predict(vectors_in)
        if len(polarities) > 1:
            for polarity in polarities:
                with open('/home/txaa2019/no_keras_LSTM/all_polarities.csv', 'a') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([counter, polarity[0]])
                    print(polarity)
        if len(polarities) == 1:
            with open('/home/txaa2019/no_keras_LSTM/all_polarities.csv', 'a') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow([counter, polarities[0][0]])
                    print(polarities[0][0])
        with open('/home/txaa2019/no_keras_LSTM/all_polarities.csv', 'a') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(['\n', '\n'])
        counter += 1

        
def output_sentiment():
    counter = 1
    for book in book_set:
        with open('/home/txaa2019/no_keras_LSTM/average_polarities.csv', 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([counter, average_sentiment(book)])
        counter += 1
        
        
# Load in dataset
book_set = pickle.load(open("books_cleaned.pickle", "rb"))
all_sentiments(book_set)
