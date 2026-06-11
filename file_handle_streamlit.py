import streamlit as st
import PyPDF2
import nltk
import heapq
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


def clean_text(text):
    text = re.sub(r'[^\w\s\.\!\?]', '', text)  # special chars hatao, punctuation rakho
    text = re.sub(r'\s+', ' ', text)            # multiple spaces ek karo
    return text.strip()


def file_nltk(text):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) == 0:   
        return ""
    if len(sentences) <= 2:   
        return " ".join(sentences)

    # edge cases pehle check karo
    if len(sentences) == 0:
        return ""
    if len(sentences) <= 2:
        return " ".join(sentences)

    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered]

    return generate_summary(lemmatized_words, sentences)


def generate_summary(lemmatized_words, sentences):
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

    summary_size = max(1, len(sentences) // 3)
    summary_sentences = heapq.nlargest(summary_size, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)


def count_frequency(lemmatized_words):
    return Counter(lemmatized_words)


def show_summary(text):
    text = clean_text(text)
    if not text.strip():
        st.warning("Text empty hai, kuch likho ya upload karo.")
        return

    with st.spinner("Summary ban rahi hai..."):
        summary = file_nltk(text)
    col1, col2 = st.columns(2)
    col1.metric("Original words", len(text.split()))
    col2.metric("Summary words", len(summary.split()))
    tex1 , text2 = st.columns(2)
    with tex1:
        st.subheader("Original Text")
        st.write(text)
    with text2:
        st.subheader("Summary")
        st.write(summary)
    

    


# --- UI ---
st.title("Text Summarizer")

choice = st.radio("Choose input type", ["Direct Text", "TXT File", "PDF File"])

if choice == "Direct Text":
    text = st.text_area("Enter your text", height=200)
    if st.button("Summarize"):
        show_summary(text)

elif choice == "TXT File":
    file = st.file_uploader("Upload .txt file", type=["txt"])
    if file is not None:
        content = file.read().decode("utf-8")
        show_summary(content)

elif choice == "PDF File":
    file = st.file_uploader("Upload .pdf file", type=["pdf"])
    if file is not None:
        reader = PyPDF2.PdfReader(file)
        content = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                content += page_text
        show_summary(content)