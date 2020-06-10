'''
Processing Workflow for the text files
Consists of tokenizations, stop word removal, non-alphabetic symbol removal, lemmatization
'''


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
    all_text_name = []

    for filename in os.listdir(folder_location):
        if filename.endswith(".txt"):
            location = folder_location + filename
            file_in = open(location, encoding = 'utf-8')
            text = file_in.read()
            file_in.close()
            array_all_texts_initial.append(text)
            all_text_name.append(filename)
        else:
            continue

    return array_all_texts_initial, all_text_name


more_stop_words = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                   'august', 'september', 'october', 'november', 'december', 'today',
                   'tomorrow', 'yesterday', 'said', 'say', 'two', 'one', 'three', 'four', 'five',
                   'six', 'seven', 'eight', 'nine', 'ten', 'monday', 'tuesday', 'wednesday',
                   'thursday', 'friday', 'saturday', 'sunday']


# pre-stop word processes
def process_workflow(folder_location, out_location):
    texts, text_names = build_text_corpus(folder_location)
    lemmatizer = WordNetLemmatizer()

    for i in range(len(texts)):
        try:
            tokens = word_tokenize(texts[i])
            # convert to lower case
            tokens = [w.lower() for w in tokens]
            # remove punctuation
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            # remove remaining non-alphabetic words
            words = [word for word in stripped if word.isalpha()]
            stop_words = set(stopwords.words('english') + list(punctuation) + more_stop_words)
            cleaned = [w for w in words if w not in stop_words]
            super_clean = [lemmatizer.lemmatize(word) for word in cleaned]

            with open(out_location + text_names[i], "a") as f:
                for word in super_clean:
                    f.write(word + ' ')
            print('cleaning...')
        except:
            continue


out_location = "your\\text\\output\\location\\"
folder_location = "your\\text\folder\location\\"
process_workflow(folder_location, out_location)
