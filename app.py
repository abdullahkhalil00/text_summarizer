import streamlit as st
import PyPDF2
import nltk
import heapq
import re
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

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

def generate_summary_tf(lemmatized_words, sentences, summary_length):
    freq = {}
    for w in lemmatized_words:
        freq[w] = freq.get(w, 0) + 1

    max_freq = max(freq.values()) if freq else 1
    for w in freq:
        freq[w] = freq[w] / max_freq

    scores = {}
    for s in sentences:
        for w in word_tokenize(s.lower()):
            if w in freq:
                scores[s] = scores.get(s, 0) + freq[w]

    top_n = min(summary_length, len(sentences))
    top_sentences = heapq.nlargest(top_n, scores, key=scores.get)
    return " ".join(top_sentences)

def generate_summary_tfidf(sentences, summary_length):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = tfidf_matrix.sum(axis=1).A1
    ranked_sentences = [sentences[i] for i in sentence_scores.argsort()[::-1]]
    top_n = min(summary_length, len(sentences))
    return " ".join(ranked_sentences[:top_n])

def create_txt(original_text, tf_summary, tfidf_summary):
    """Create a TXT file with original text, TF summary, and TF-IDF summary."""
    content = (
        "======Original Text\n"
        + original_text + "\n\n"
        "======TF Summary\n"
        + tf_summary + "\n\n"
        "======TF-IDF Summary\n"
        + tfidf_summary + "\n"
    )
    return content.encode("utf-8")

def create_pdf(original_text, tf_summary, tfidf_summary):
    """Create a PDF with original text, TF summary, and TF-IDF summary."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    page_width, page_height = letter
    margin = 50
    max_width = 85  # characters per line
    y = page_height - 60

    def draw_section_header(title):
        nonlocal y
        if y < 100:
            c.showPage()
            y = page_height - 60
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, y, title)
        y -= 6
        c.setLineWidth(0.8)
        c.line(margin, y, page_width - margin, y)
        y -= 18

    def draw_paragraph(text):
        nonlocal y
        c.setFont("Helvetica", 10)
        words = text.split()
        line = ""
        for word in words:
            if len(line + word) < max_width:
                line += word + " "
            else:
                if y < 60:
                    c.showPage()
                    y = page_height - 60
                    c.setFont("Helvetica", 10)
                c.drawString(margin, y, line.strip())
                y -= 16
                line = word + " "
        if line.strip():
            if y < 60:
                c.showPage()
                y = page_height - 60
                c.setFont("Helvetica", 10)
            c.drawString(margin, y, line.strip())
            y -= 16
        y -= 10  # spacing after paragraph

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Text Summarizer - Output Report")
    y -= 30

    # Section 1: Original Text
    draw_section_header("Original Text")
    draw_paragraph(original_text)

    # Section 2: TF Summary
    draw_section_header("TF Summary")
    draw_paragraph(tf_summary)

    # Section 3: TF-IDF Summary
    draw_section_header("TF-IDF Summary")
    draw_paragraph(tfidf_summary)

    c.save()
    buffer.seek(0)
    return buffer

def show_summary(text, s1, s2):
    c1, c2, c3 = st.columns(3)
    c1.metric("Original words", len(text.split()))
    c2.metric("TF Summary words", len(s1.split()))
    c3.metric("TF-IDF Summary words", len(s2.split()))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Text")
        st.write(text)

    with col2:
        st.subheader("TF Summary")
        st.write(s1)

    st.subheader("TF-IDF Summary")
    st.write(s2)

    st.markdown("---")
    st.subheader("Download Summary")

    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        txt_data = create_txt(text, s1, s2)
        st.download_button(
            label="📄 Download Summary (.txt)",
            data=txt_data,
            file_name="summary.txt",
            mime="text/plain",
            use_container_width=True
        )

    with dl_col2:
        pdf_data = create_pdf(text, s1, s2)
        st.download_button(
            label="📑 Download Summary (.pdf)",
            data=pdf_data,
            file_name="summary.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ── App ──────────────────────────────────────────────────────────
st.title("Text Summarizer")

choice = st.radio("Choose input type", ["Direct Text", "TXT File", "PDF File"])
summary_len = st.text_input("Summary length (sentences)", "3")
summary_len = int(summary_len) if summary_len.isdigit() else 3

if choice == "Direct Text":
    text = st.text_area("Enter text")
    if st.button("Summarize"):
        text = clean_text(text)
        lemmatized_words, sentences = file_nltk(text)
        s1 = generate_summary_tf(lemmatized_words, sentences, summary_len)
        s2 = generate_summary_tfidf(sentences, summary_len)
        show_summary(text, s1, s2)

elif choice == "TXT File":
    file = st.file_uploader("Upload TXT", type=["txt"])
    if file:
        text = file.read().decode("utf-8")
        text = clean_text(text)
        lemmatized_words, sentences = file_nltk(text)
        s1 = generate_summary_tf(lemmatized_words, sentences, summary_len)
        s2 = generate_summary_tfidf(sentences, summary_len)
        show_summary(text, s1, s2)

elif choice == "PDF File":
    file = st.file_uploader("Upload PDF", type=["pdf"])
    if file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        text = clean_text(text)
        lemmatized_words, sentences = file_nltk(text)
        s1 = generate_summary_tf(lemmatized_words, sentences, summary_len)
        s2 = generate_summary_tfidf(sentences, summary_len)
        show_summary(text, s1, s2)