import VoteClassifier as judge
import io


def analysis(data):
    book_ana = data
    sentiment_value, confidence = judge.sentiment_vote(book_ana)

    if confidence * 100 >= 80:
        output = open("D:\\Pomona\\Independent Research\\sentiment_analysis.py\\book-out.txt", "a")
        output.write(sentiment_value + str(confidence) + data[:3])
        output.write('\n')
        output.close()
    else:
        output = open("D:\\Pomona\\Independent Research\\sentiment_analysis.py\\book-out.txt", "a")
        output.write("inconclusive" + str(confidence))
        output.write('\n')
    return True


with io.open("D:\\Pomona\\Independent Research\\dataset\\all_book_cleaned.txt", 'r', encoding = 'utf8') as f:
    text = f.read()
    text_array = text.split('\n')
    f.close()
    for book in text_array:
        analysis(book)

