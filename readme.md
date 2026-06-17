# 📝 Text Summarizer

> An NLP-powered document summarization app — paste text, upload TXT/PDF files (single or multiple), and get instant summaries in **three different algorithms**, in **any language**.

🔗 **Live App:** [https://textsummarizer-cquhjmw7kqfvjappgkkbjs3.streamlit.app/](https://textsummarizer-cquhjmw7kqfvjappgkkbjs3.streamlit.app/)

---

## ✨ Features

- **4 Input Types** — Direct text input, TXT file upload, PDF file upload, Multiple Documents at once
- **3 Summary Algorithms** — TF (Term Frequency), TF-IDF, and Abstractive (Transformer-generated, not just extracted sentences)
- **Multilingual** — automatically detects the input language, translates internally to English for summarization, then translates every summary back into the original language
- **Multi-Document Summarization** — upload several files together and get both per-document summaries and one combined, deduplicated summary across all of them (using diversity-aware sentence selection so the same point isn't repeated from multiple files)
- **Word Count Metrics** — compare original vs. summary word counts at a glance
- **Download Output** — save any summary (single or multi-document) as `.txt` or `.pdf`
- **Adjustable Controls** — extractive summary length (sentences), abstractive summary length (tokens), beam search width, and a diversity slider for multi-document summaries
- **Lightweight model options** — choose between `flan-t5-small` (recommended) or `t5-small` for abstractive summarization, or disable abstractive summarization entirely to save memory

---

## ⚙️ Technologies Used

### 🖥️ Framework
| Library | Purpose |
|---------|---------|
| `streamlit` | Web UI framework for building interactive Python apps |
| `Python 3.11+` | Core backend language |

### 🧠 NLP & Text Processing
| Component | Purpose |
|-----------|---------|
| Custom regex sentence splitter | Splits text into sentences — replaces NLTK's `punkt` tokenizer (no corpus download, no `LookupError` risk on fresh deploys) |
| Custom regex word tokenizer | Splits sentences into words |
| Hardcoded English stopword set | Filters out common words (the, is, in, etc.) without a runtime download |
| `unicodedata` | Cleans control/junk characters from extracted text while preserving every language's punctuation (Urdu ۔, Hindi ।, etc.) |

> 🧩 NLTK was removed entirely. Sentence/word splitting always runs on already-translated English text in this app, so a small regex-based tokenizer gives equivalent results with far less RAM, no startup downloads, and no corpus-related crashes.

### 🌍 Language Detection & Translation
| Library | Purpose |
|---------|---------|
| `langdetect` | Detects the language of the input text |
| `deep-translator` (`GoogleTranslator`) | Translates text to English for summarization, then translates the summaries back to the original language |

### 🤖 Machine Learning
| Library | Purpose |
|---------|---------|
| `sklearn.feature_extraction.text.TfidfVectorizer` | Computes TF-IDF scores for sentence ranking |
| `sklearn.metrics.pairwise.cosine_similarity` | Powers Maximal Marginal Relevance (MMR) — picks relevant *and* non-redundant sentences for multi-document summaries |
| `heapq` | Efficiently extracts the top-N highest scoring sentences |
| `transformers` + `torch` | Runs the abstractive summarization model (`google/flan-t5-small` by default) |

### 📄 File Handling
| Library | Purpose |
|---------|---------|
| `pypdf` | Reads and extracts text from PDF files (actively-maintained successor to `PyPDF2`) |
| `reportlab` | Generates PDF output files programmatically |
| `io.BytesIO` | In-memory binary buffer for PDF generation |
| `re` | Regular expressions for text cleaning and tokenization |

---

## 🔄 How It Works

### Overall Pipeline
```
Input (text / TXT / PDF / multiple files)
    ↓
Clean Text       (strip junk characters, keep all language punctuation)
    ↓
Detect Language  (langdetect)
    ↓
Translate → EN   (deep-translator, skipped if already English)
    ↓
Summarize        (TF · TF-IDF · Abstractive — all three, in parallel)
    ↓
Translate Back   (each summary → original language)
    ↓
Display + Download (.txt / .pdf)
```

### Algorithm 1 — TF (Term Frequency)
```
English Text
    ↓
Tokenize     (regex sentence + word splitter)
    ↓
Filter       (remove stopwords)
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
English Sentences
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

### Algorithm 3 — Abstractive (Transformers)
```
English Text
    ↓
Chunk        (split long text into ~380-word pieces if needed)
    ↓
Generate     (flan-t5-small / t5-small via model.generate(), beam search)
    ↓
Re-summarize (if multiple chunks: summarize the chunk-summaries together)
    ↓
Summary ✅   (a genuinely new sentence, not lifted from the source)
```

> **What's the difference?** TF ranks sentences purely by word frequency. TF-IDF gives higher weight to words that are rare but meaningful, making it generally more informative than TF. Both are *extractive* — they pick existing sentences. Abstractive summarization *generates* new sentences that paraphrase the source, similar to how a person would summarize it.

### Multi-Document Mode
Each uploaded file is summarized individually (its own language detected, translated, summarized, translated back). Then every document's English sentences are pooled together, exact duplicates are dropped, and a **Maximal Marginal Relevance (MMR)** selection balances relevance against similarity to sentences already chosen — so the combined summary doesn't just repeat the same point because three files happened to mention it. The combined abstractive summary runs the same chunk-and-regenerate process over the pooled text. The final combined summary is translated into whichever output language you choose.

---

## 🚀 Performance Notes (Streamlit Community Cloud)

This app is tuned to run comfortably on free-tier hosting:
- **No NLTK** — avoids multi-megabyte corpus downloads and the network calls/crashes that can come with them on a fresh deploy.
- **Lightweight abstractive model** — `flan-t5-small` (~80M params, ~300MB) instead of heavier options like `bart-large-cnn` (~400M params, ~1.6GB).
- **Lazy model loading** — the Transformer model only loads the first time you actually click "Summarize" with abstractive summarization enabled, and is cached afterward (`st.cache_resource`) so it never reloads.
- **CPU-only PyTorch** — `requirements.txt` pulls the CPU-only build, skipping unused CUDA libraries.
- **Adjustable beam width** — defaults to 2 beams instead of 4, with a 1–4 slider so you can trade speed for quality.
- **Optional abstractive toggle** — turn it off entirely if you only need TF/TF-IDF and want to skip loading any model.

---

## 💻 Local Setup

### Prerequisites
- Python 3.11 or higher
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

> ℹ️ The first time you run an abstractive summary, the app will download the selected model (a few hundred MB) — this happens once and is cached for all future runs.

---

## 📋 requirements.txt

```
streamlit
pypdf
scikit-learn
reportlab
langdetect
deep-translator

# CPU-only PyTorch build — smaller install, no unused CUDA libraries
--extra-index-url https://download.pytorch.org/whl/cpu
torch

transformers
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

*Made with Abdullah using Python · Transformers · Scikit-learn · Streamlit*