# 📝 Text Summarizer

> An NLP-powered document summarization app — paste text, upload a TXT or PDF, and get instant summaries using two different algorithms.

🔗 **Live App:** [https://textsummarizer-cquhjmw7kqfvjappgkkbjs3.streamlit.app/](https://textsummarizer-cquhjmw7kqfvjappgkkbjs3.streamlit.app/)

---

## ✨ Features

- **3 Input Types** — Direct text input, TXT file upload, PDF file upload
- **2 Summary Algorithms** — TF (Term Frequency) and TF-IDF
- **Word Count Metrics** — Compare original vs summary word counts
- **Download Output** — Save your summary as `.txt` or `.pdf`
- **Adjustable Length** — Choose how many sentences the summary contains

---

## ⚙️ Technologies Used

### 🖥️ Framework
| Library | Purpose |
|---------|---------|
| `streamlit` | Web UI framework for building interactive Python apps |
| `Python 3.8+` | Core backend language |

### 🧠 NLP & Text Processing
| Library | Purpose |
|---------|---------|
| `nltk` | Natural Language Toolkit — core NLP operations |
| `nltk.sent_tokenize` | Splits text into individual sentences |
| `nltk.word_tokenize` | Splits sentences into individual words |
| `nltk.corpus.stopwords` | Filters out common words (the, is, in, etc.) |
| `nltk.stem.WordNetLemmatizer` | Reduces words to their root form (running → run) |

### 🤖 Machine Learning
| Library | Purpose |
|---------|---------|
| `sklearn.feature_extraction.text.TfidfVectorizer` | Computes TF-IDF scores for sentence ranking |
| `heapq` | Efficiently extracts the top-N highest scoring sentences |

### 📄 File Handling
| Library | Purpose |
|---------|---------|
| `PyPDF2` | Reads and extracts text from PDF files |
| `reportlab` | Generates PDF output files programmatically |
| `io.BytesIO` | In-memory binary buffer for PDF generation |
| `re` | Regular expressions for text cleaning (special chars, extra spaces) |

---

## 🔄 How It Works

### Algorithm 1 — TF (Term Frequency)
```
Input Text
    ↓
Clean Text   (regex — remove special characters & extra spaces)
    ↓
Tokenize     (split into sentences + words via NLTK)
    ↓
Filter       (remove stopwords)
    ↓
Lemmatize    (WordNetLemmatizer — reduce to root form)
    ↓
Score        (word frequency / max_frequency)
    ↓
Rank         (sentence score = sum of its word scores)
    ↓
Top-N        (heapq.nlargest)
    ↓
Summary ✅
```

### Algorithm 2 — TF-IDF
```
Input Sentences
    ↓
Vectorize    (TfidfVectorizer — convert sentences to numeric matrix)
    ↓
Score        (row-wise sum of TF-IDF matrix)
    ↓
Sort         (descending order by score)
    ↓
Top-N        (select first N sentences)
    ↓
Summary ✅
```

> **What's the difference?** TF ranks sentences purely by word frequency. TF-IDF gives higher weight to words that are rare but meaningful — making its summaries generally more informative.

---

## 💻 Local Setup

### Prerequisites
- Python 3.8 or higher
- `pip` package manager

### 1. Clone the Repository
```bash
git clone https://github.com/abdullahkhalil00/text_summarizer
cd text-summarizer
```



### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

The app will open in your browser at:
```
http://localhost:8501
```

---

## 📋 requirements.txt

```
streamlit
PyPDF2
nltk
scikit-learn
reportlab
```

---

## 📁 Project Structure

```
text-summarizer/
│
├── app.py               ← Main Streamlit application
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## 🌐 Deployment

This app is deployed on **Streamlit Community Cloud** — free hosting with auto-deploy on every push to the repository.

🔗 [https://textsummarizer-cquhjmw7kqfvjappgkkbjs3.streamlit.app/](https://textsummarizer-cquhjmw7kqfvjappgkkbjs3.streamlit.app/)

---

*Made with Abdullah using Python · NLTK · Scikit-learn · Streamlit*