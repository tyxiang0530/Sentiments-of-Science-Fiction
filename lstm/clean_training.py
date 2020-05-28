import nltk
nltk.download()
from nltk.tokenize import word_tokenize
import string
import os
import pickle
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
import io
import re
import numpy as np
from collections import Counter
from keras.preprocessing import sequence


# preprocess_reviews removes non-alphabetical characters, stop words, and lemmatizes dataset
def preprocess_reviews(reviews):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(reviews)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining non-alphabetic words
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english')+list(punctuation))
    cleaned = [w for w in words if w not in stop_words]
    super_clean = [lemmatizer.lemmatize(word) for word in cleaned]
    return super_clean


# open our datasets
reviews_train = []
for line in open('/home/txaa2019/movie_training_dataset/movie_data/full_train.txt', 'r'):
    reviews_train.append(line.strip())
    
reviews_test = []
for line in open('/home/txaa2019/movie_training_dataset/movie_data/full_test.txt', 'r'):
    reviews_test.append(line.strip())
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")


# clean our reviews
cleaned_reviews_test = []
cleaned_reviews_train = []
for review in reviews_train:
    cleaned_reviews_train.append(preprocess_reviews(review))
for review in reviews_test:
    cleaned_reviews_test.append(preprocess_reviews(review))
print(cleaned_reviews_test[0])
print(cleaned_reviews_train[0])


# begin creating dictionary that matches words to integers
counts = Counter()
for line in cleaned_reviews_train:
    nums = Counter(line)
    counts = counts + nums
for line in cleaned_reviews_test:
    nums = Counter(line)
    counts = counts + nums
    

# removes all words that appear in our dataset less than two times (this removes very obscure words/mispellings)
counts = Counter(el for el in counts.elements() if counts[el] >= 2)

# save our dictionary for later use
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
pickle_out = open("vocab_to_word_dict", "wb")
pickle.dump(vocab_to_int, pickle_out)
pickle_out.close()

# convert our datasets to integers
print(len(cleaned_reviews_train))
train_ints = []
test_ints = []
for review in cleaned_reviews_train:
    temp = []
    for line in review:
        try:
            temp.append(vocab_to_int[line])
        except:
            temp.append(0)
    train_ints.append(temp)
print(train_ints[2])
for review in cleaned_reviews_test:
    temp = []
    for line in review:
        try:
            temp.append(vocab_to_int[line])
        except: temp.append(0)
    test_ints.append(temp)

# encode our labels: 1 for positive, 0 for negative
with open('/home/txaa2019/movie_training_dataset/movie_data/labels.txt', 'r') as f:
    labels = f.read()
labels_split = labels.split('\n')
print(len(labels_split))
encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])
print(labels_split[12499])

# pad our reviews so that they are all the same length and can be fed into deep learning model
seq_length = 500
train_ints = sequence.pad_sequences(train_ints, maxlen = seq_length)
test_ints = sequence.pad_sequences(test_ints, maxlen = seq_length)
all_train = np.concatenate([train_ints, test_ints])
all_label = np.concatenate([encoded_labels, encoded_labels])
print(len(all_train))
print(len(all_label))

# shuffle all datasets
from sklearn.utils import shuffle

all_train, all_label = shuffle(all_train, all_label, random_state=0)
print(len(all_train))
print(len(all_label))

# save our dataset
import pickle

pickle_out = open("50k_train.pickle", "wb")
pickle.dump(all_train, pickle_out)
pickle_out.close()

pickle_out = open("50k_label.pickle", "wb")
pickle.dump(all_label, pickle_out)
pickle_out.close()
