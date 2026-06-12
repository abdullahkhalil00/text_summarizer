import nltk
import heapq
import math
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def filehandling():
    print("Choose 1 for direct text input, 2 for .txt file, 3 for .pdf file")
    choice = input("Enter your choice: ")

    if choice == '1':
        text = input("Enter your text: ")
        summary = file_nltk(text)
        print("Summary: ", summary)

    elif choice == '2':
        file_path = input("Enter the path of the .txt file: ")
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                summary = file_nltk(content)
                print("Summary: ", summary)
        except FileNotFoundError:
            print("File not found. Please check the path and try again.")

    elif choice == '3':
        file_path = input("Enter the path of the .pdf file: ")
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text
                summary = file_nltk(content)
                print("Summary:")
                print(summary)
        except FileNotFoundError:
            print("File not found. Please check the path and try again.")
        except ImportError:
            print("PyPDF2 library is not installed. Please install it to read PDF files.")
    else:
        print("Invalid choice. Please choose 1, 2, or 3.")


def file_nltk(text):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Sentences pehle nikaal lo — baad mein generate_summary ko denge
    sentences = nltk.sent_tokenize(text)

    # Tokenization, stopword removal, lemmatization
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered]

    # Dono pass karo — dobara tokenize nahi hoga
    summary = generate_summary(lemmatized_words, sentences)
    return summary


def generate_summary(lemmatized_words, sentences):
    frequency = count_frequency(lemmatized_words)
    if not frequency:
        return ""

    max_freq = max(frequency.values())
    for word in frequency:
        frequency[word] = frequency[word] / max_freq

    # Sentences pehle se aa rahi hain — yahan sirf scoring ho rahi hai
    sentence_scores = {}
    for sentence in sentences:
        for word in sentence.lower().split():
            if word in frequency:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = frequency[word]
                else:
                    sentence_scores[sentence] += frequency[word]

    summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary


def count_frequency(lemmatized_words):
    from collections import Counter
    return Counter(lemmatized_words)
import math
from nltk.tokenize import word_tokenize

def TF_IDF(lemmatized_words, sentences):
    frequency = count_frequency(lemmatized_words)
    tf_idf_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in frequency:
                tf = frequency[word] / len(lemmatized_words)
                df = sum(1 for s in sentences if word in s.lower())
                idf = math.log(len(sentences) / (1 + df))
                tf_idf_score = tf * idf
                if sentence not in tf_idf_scores:
                    tf_idf_scores[sentence] = tf_idf_score
                else:
                    tf_idf_scores[sentence] += tf_idf_score
    summary_sentences = heapq.nlargest(5, tf_idf_scores, key=tf_idf_scores.get)
    summary = ' '.join(summary_sentences)
    return summary


filehandling()