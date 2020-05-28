import collections
import pickle


def find_stop_word(text_in):
    word_counts = collections.Counter(word for words in text_in for word in words)
    return word_counts.most_common(100)


books_in = open("path\\to\\pickled\\file", "rb")
book_array = pickle.load(books_in)
books_in.close()
print(find_stop_word(book_array))
