import streamlit as st
import PyPDF2
import nltk
import heapq
import re
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(text):
    text = re.sub(r'[^\w\s\.\!\?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def file_nltk(text):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) == 0:
        return [], []
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered]
    return lemmatized_words, sentences

def count_frequency(words):
    return Counter(words)

def generate_summary_tf(lemmatized_words, sentences, summary_length):
    frequency = count_frequency(lemmatized_words)
    if not frequency:
        return ""
    max_freq = max(frequency.values())
    for word in frequency:
        frequency[word] = frequency[word] / max_freq
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in frequency:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + frequency[word]
    summary_size = min(summary_length, len(sentences))
    summary_sentences = heapq.nlargest(summary_size, sentence_scores, key=sentence_scores.get)
    return " ".join(summary_sentences)

def generate_summary_tfidf(lemmatized_words, sentences, summary_length):
    frequency = count_frequency(lemmatized_words)
    tfidf_scores = {}
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        for word in words:
            if word in frequency:
                tf = frequency[word] / len(lemmatized_words)
                df = sum(1 for s in sentences if word in s.lower())
                idf = math.log(len(sentences) / (1 + df))
                score = tf * idf
                tfidf_scores[sentence] = tfidf_scores.get(sentence, 0) + score
    summary_size = min(summary_length, len(sentences))
    summary_sentences = heapq.nlargest(summary_size, tfidf_scores, key=tfidf_scores.get)
    return " ".join(summary_sentences)

def show_summary(text, summary1, summary2):
    col1, col2, col3 = st.columns(3)
    col1.metric("Original words", len(text.split()))
    col2.metric("TF Summary words", len(summary1.split()))
    col3.metric("TF-IDF Summary words", len(summary2.split()))
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Original Text")
        st.write(text)
    with col5:
        st.subheader("TF Summary")
        st.write(summary1)
    st.subheader("TF-IDF Summary")
    st.write(summary2)

st.title("Text Summarizer")

choice = st.radio("Choose input type", ["Direct Text", "TXT File", "PDF File"])

summary_input = st.text_input("Choose summary length (number of sentences)", "3")

if summary_input.isdigit():
    summary_length = int(summary_input)
else:
    st.warning("Invalid input. Using default value 3.")
    summary_length = 3

if choice == "Direct Text":
    text = st.text_area("Enter your text", height=200)
    if st.button("Summarize"):
        text = clean_text(text)
        lemmatized_words, sentences = file_nltk(text)
        if len(sentences) < summary_length:
            st.warning("Text is too short for selected summary length.")
        summary1 = generate_summary_tf(lemmatized_words, sentences, summary_length)
        summary2 = generate_summary_tfidf(lemmatized_words, sentences, summary_length)
        show_summary(text, summary1, summary2)

elif choice == "TXT File":
    file = st.file_uploader("Upload .txt file", type=["txt"])
    if file is not None:
        content = file.read().decode("utf-8")
        content = clean_text(content)
        lemmatized_words, sentences = file_nltk(content)
        summary1 = generate_summary_tf(lemmatized_words, sentences, summary_length)
        summary2 = generate_summary_tfidf(lemmatized_words, sentences, summary_length)
        show_summary(content, summary1, summary2)

elif choice == "PDF File":
    file = st.file_uploader("Upload .pdf file", type=["pdf"])
    if file is not None:
        reader = PyPDF2.PdfReader(file)
        content = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                content += page_text
        content = clean_text(content)
        lemmatized_words, sentences = file_nltk(content)
        summary1 = generate_summary_tf(lemmatized_words, sentences, summary_length)
        summary2 = generate_summary_tfidf(lemmatized_words, sentences, summary_length)
        show_summary(content, summary1, summary2)