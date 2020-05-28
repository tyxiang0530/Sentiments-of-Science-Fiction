from nltk.tokenize import word_tokenize
import string
import os
import pickle
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer
import io


# load text
def build_text_corpus(folder_location):
    array_all_texts_initial = []
    for filename in os.listdir(folder_location):
        if filename.endswith(".txt"):
            location = folder_location + filename
            file_in = open(location, encoding = 'utf-8')
            text = file_in.read()
            file_in.close()
            array_all_texts_initial.append(text)
        else:
            continue

    with open('first_all_text_uncleaned.pickle', 'wb') as pickled_location:
        pickle.dump(array_all_texts_initial, pickled_location)


more_stop_words = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                   'august', 'september', 'october', 'november', 'december', 'today',
                   'tomorrow', 'yesterday', 'said', 'say', 'two', 'one', 'three', 'four', 'five',
                   'six', 'seven', 'eight', 'nine', 'ten', 'monday', 'tuesday', 'wednesday',
                   'thursday', 'friday', 'saturday', 'sunday', 'page']


# pre-stop word processes
def process_workflow(text_in):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text_in)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining non-alphabetic words
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english')+list(punctuation)+more_stop_words)
    cleaned = [w for w in words if w not in stop_words]
    super_clean = [lemmatizer.lemmatize(word) for word in cleaned]
    with io.open("D:\\Pomona\\Independent Research\\dataset\\all_book_cleaned.txt", 'a', encoding='utf8') as f:
        for sub in super_clean:
            try:
                f.write(sub + ' ')
            except:
                continue
        f.write('\n')
        f.close()


all_text_location = "text\\input\\location"
pickled_location = "pickle\\output\\location\\first_all_text_uncleaned.pickle"
# build_text_corpus(all_text_location)
books_in = open(pickled_location, "rb")
book_array = pickle.load(books_in)
books_in.close()
for book in book_array:
    process_workflow(book)
