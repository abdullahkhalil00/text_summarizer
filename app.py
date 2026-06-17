"""
Multilingual Text Summarizer (TF / TF-IDF / Abstractive) — Streamlit Cloud build.

Optimized for low RAM/CPU usage on free hosting tiers:
  - No NLTK (removes ~50MB+ of downloaded corpora and a network call at startup).
  - Abstractive model defaults to google/flan-t5-small (~300MB) instead of
    facebook/bart-large-cnn (~1.6GB) — see ABSTRACTIVE_MODELS below for why.
  - The Transformer model is loaded lazily (only when the user actually clicks
    "Summarize" with abstractive summarization enabled) and cached process-wide
    via st.cache_resource so it is downloaded/loaded only once.
  - Beam search width is user-adjustable and defaults to 2 instead of 4.
"""

import os

# Must be set before transformers/torch do any work — quiets a noisy fork
# warning from the tokenizers library and skips a telemetry network call,
# both of which slightly slow down cold starts on Streamlit Cloud.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import re
import heapq
import unicodedata
from io import BytesIO

import streamlit as st
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# A single shared thread is enough for one small CPU model serving one
# request at a time; avoids torch over-subscribing the limited vCPUs that
# free hosting tiers give you.
torch.set_num_threads(1)


LANG_NAMES = {
    "en": "English", "ur": "Urdu", "hi": "Hindi", "ar": "Arabic",
    "fr": "French", "es": "Spanish", "de": "German", "zh-cn": "Chinese",
    "ru": "Russian", "pt": "Portuguese", "tr": "Turkish", "fa": "Persian",
}

# Both options are encoder-decoder T5 models, which are far lighter than
# BART-large (400M params / ~1.6GB) while still producing coherent abstractive
# summaries. flan-t5-small is instruction-tuned, so it follows the
# "summarize: " prompt more reliably than the older t5-small.
ABSTRACTIVE_MODELS = {
    "google/flan-t5-small": "FLAN-T5 Small — recommended (~80M params, ~300MB, instruction-tuned)",
    "t5-small": "T5 Small — lightest option (~60M params, ~240MB, slightly lower quality)",
}


# ── Lightweight text utilities (replaces NLTK) ───────────────────
# NLTK's punkt/punkt_tab/wordnet downloads add network calls, disk usage,
# and a class of "LookupError: resource not found" failures on fresh
# Streamlit Cloud instances. Sentence/word splitting only ever runs on text
# that has already been translated to English in this app, so a small regex
# splitter plus a hardcoded stopword set gives equivalent results for the
# TF/TF-IDF scoring this app needs, with no extra downloads and lower RAM.
_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more",
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've",
    "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
})

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
_WORD_RE = re.compile(r"[a-zA-Z']+")


def simple_sent_tokenize(text):
    """Lightweight sentence splitter (replacement for nltk.sent_tokenize)."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def simple_word_tokenize(text):
    """Lightweight word tokenizer (replacement for nltk.word_tokenize)."""
    return _WORD_RE.findall(text.lower())


# ── Text utilities ───────────────────────────────────────────────
def clean_text(text):
    """
    Strip control / non-printable junk characters (common artifacts from PDF
    text extraction) while preserving all language punctuation — including
    Urdu/Arabic full stops (۔) and commas (،), Hindi danda (।), etc. Keeping
    these matters because translation and sentence-splitting both rely on
    the original sentence-ending marks.
    """
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def detect_language(text):
    """Detect language of the given text. Falls back to English on failure."""
    try:
        sample = text[:1000] if len(text) > 1000 else text
        return detect(sample)
    except LangDetectException:
        return "en"


@st.cache_data(show_spinner=False, max_entries=300)
def translate_text(text, source, target, chunk_size=4000):
    """
    Translate text between languages, chunking long text to respect API
    limits. Cached so re-running the same input (e.g. after tweaking a
    summary-length slider) doesn't re-hit the translation API.
    """
    if not text.strip() or source == target:
        return text
    try:
        if len(text) <= chunk_size:
            return GoogleTranslator(source=source, target=target).translate(text)

        sentences = simple_sent_tokenize(text)
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) + 1 <= chunk_size:
                current += s + " "
            else:
                chunks.append(current.strip())
                current = s + " "
        if current.strip():
            chunks.append(current.strip())

        translator = GoogleTranslator(source=source, target=target)
        return " ".join(translator.translate(c) for c in chunks)
    except Exception:
        # Fail soft: keep the pipeline running with the untranslated text
        # rather than crashing the whole summary.
        return text


def extract_text_from_uploaded(file):
    """Extract raw text from an uploaded TXT or PDF file object."""
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    return file.read().decode("utf-8", errors="ignore")


def tokenize_for_scoring(text):
    """
    Split text into sentences and a stopword-filtered, lowercased word list
    used for TF/TF-IDF scoring. No stemming/lemmatization — intentionally
    simple, since frequency-based extractive scoring doesn't need it and it
    avoids pulling in NLTK's WordNet corpus.
    """
    sentences = simple_sent_tokenize(text)
    if not sentences:
        return [], []
    words = [w for w in simple_word_tokenize(text) if w not in _STOPWORDS and len(w) > 1]
    return words, sentences


# ── Single-document extractive scoring (TF / TF-IDF) ─────────────
def generate_summary_tf(words, sentences, summary_length):
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    max_freq = max(freq.values()) if freq else 1
    for w in freq:
        freq[w] /= max_freq

    scores = {}
    for s in sentences:
        for w in simple_word_tokenize(s):
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


# ── Abstractive scoring (Transformers, lightweight T5 family) ────
@st.cache_resource(show_spinner="Loading summarization model (first run only)...")
def get_abstractive_summarizer(model_name):
    """
    Loads the tokenizer + seq2seq model directly (no pipeline()) so this
    works across Transformers versions, including v5, which removed the
    old pipeline("summarization", ...) task registry entirely.

    low_cpu_mem_usage=True avoids allocating a duplicate full-precision copy
    of the weights while loading, which matters on memory-capped free tiers.
    Cached process-wide via st.cache_resource: downloaded/loaded exactly once.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,  # CPU inference: fp16/bf16 give no speedup here
    )
    model.to("cpu")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return {"tokenizer": tokenizer, "model": model}


def _chunk_by_words(text, max_words=380):
    """
    Rough chunking so each piece stays within the model's effective context
    budget. 380 words leaves headroom under T5's ~512-token comfort zone
    once the "summarize: " prompt and subword tokenization are accounted for.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _run_seq2seq(text, summarizer, max_new_tokens, min_new_tokens, num_beams):
    tokenizer, model = summarizer["tokenizer"], summarizer["model"]
    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=512)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,       # 1-2 beams keeps CPU/RAM use low; 4 was overkill
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def generate_summary_abstractive(text, summarizer, max_length=100, min_length=20, num_beams=2):
    """
    Abstractive summary using a lightweight T5-family model. Long inputs are
    chunked, each chunk summarized, then those chunk summaries are
    summarized again (hierarchical) to land near the requested length.
    Pass summarizer=None to skip abstractive summarization entirely (used
    when the user disables it to save memory).
    """
    if summarizer is None:
        return ""
    text = text.strip()
    if not text:
        return ""
    if len(text.split()) < 40:
        return text  # too short to meaningfully compress

    chunks = _chunk_by_words(text)
    safe_min = max(5, min_length)

    try:
        if len(chunks) == 1:
            return _run_seq2seq(chunks[0], summarizer, max_length, safe_min, num_beams)

        chunk_summaries = [
            _run_seq2seq(c, summarizer, max_length, max(5, safe_min // 2), num_beams)
            for c in chunks
        ]
        combined = " ".join(chunk_summaries)
        if len(combined.split()) > max_length:
            return _run_seq2seq(combined, summarizer, max_length, safe_min, num_beams)
        return combined
    except Exception as e:
        return f"(Abstractive summarization failed: {e})"


# ── Multi-document extractive scoring (relevance + diversity / MMR) ─
def select_diverse_sentences(sentences, tfidf_matrix, base_scores, top_n, diversity=0.3):
    """
    Maximal Marginal Relevance style selection: balances how relevant a
    sentence is against how similar it is to sentences already picked, so
    the combined multi-document summary doesn't repeat near-duplicate
    sentences pulled from different files.
    """
    n = len(sentences)
    top_n = min(top_n, n)
    if n == 0:
        return []

    sim_matrix = cosine_similarity(tfidf_matrix)
    selected, candidates = [], list(range(n))

    while len(selected) < top_n and candidates:
        if not selected:
            idx = max(candidates, key=lambda i: base_scores[i])
        else:
            def mmr_score(i):
                redundancy = max(sim_matrix[i][j] for j in selected)
                return (1 - diversity) * base_scores[i] - diversity * redundancy
            idx = max(candidates, key=mmr_score)
        selected.append(idx)
        candidates.remove(idx)

    selected.sort()
    return [sentences[i] for i in selected]


def generate_combined_summary_tf(words, sentences, summary_length, diversity=0.3):
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    max_freq = max(freq.values()) if freq else 1
    for w in freq:
        freq[w] /= max_freq

    scores = [sum(freq.get(w, 0) for w in simple_word_tokenize(s)) for s in sentences]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return " ".join(select_diverse_sentences(sentences, tfidf_matrix, scores, summary_length, diversity))


def generate_combined_summary_tfidf(sentences, summary_length, diversity=0.3):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    base_scores = tfidf_matrix.sum(axis=1).A1
    return " ".join(select_diverse_sentences(sentences, tfidf_matrix, base_scores, summary_length, diversity))


# ── Pipelines ─────────────────────────────────────────────────
def summarize_pipeline(raw_text, summary_length, summarizer, abs_max_len, abs_min_len, num_beams):
    """
    Single-document pipeline:
    1. Detect language
    2. Translate to English if needed
    3. Generate TF, TF-IDF, and Abstractive summaries in English
    4. Translate all three summaries back to the original language
    Returns: cleaned_original_text, tf_summary, tfidf_summary, abstractive_summary, detected_lang_code
    """
    cleaned = clean_text(raw_text)
    lang = detect_language(cleaned)

    text_en = translate_text(cleaned, source=lang, target="en") if lang != "en" else cleaned

    words, sentences = tokenize_for_scoring(text_en)
    if not sentences:
        return cleaned, "", "", "", lang

    tf_summary_en = generate_summary_tf(words, sentences, summary_length)
    tfidf_summary_en = generate_summary_tfidf(sentences, summary_length)
    abstractive_summary_en = generate_summary_abstractive(text_en, summarizer, abs_max_len, abs_min_len, num_beams)

    if lang != "en":
        tf_summary = translate_text(tf_summary_en, source="en", target=lang)
        tfidf_summary = translate_text(tfidf_summary_en, source="en", target=lang)
        abstractive_summary = (
            translate_text(abstractive_summary_en, source="en", target=lang) if abstractive_summary_en else ""
        )
    else:
        tf_summary, tfidf_summary, abstractive_summary = tf_summary_en, tfidf_summary_en, abstractive_summary_en

    return cleaned, tf_summary, tfidf_summary, abstractive_summary, lang


def summarize_multi_documents(docs, individual_len, combined_len, target_lang, summarizer,
                               diversity, abs_max_len, abs_min_len, num_beams,
                               combined_abs_max_len, combined_abs_min_len):
    """
    Multi-document pipeline. docs: list of {"name": str, "raw_text": str}

    Per document: detect its own language, translate to English, run TF /
    TF-IDF / Abstractive summaries, translate all three back to THAT
    document's own language.

    Combined: pool every document's English sentences, dedupe, run
    diversity-aware (MMR) TF and TF-IDF summaries over the pool, and run the
    abstractive model over the pooled English text (chunked + hierarchically
    re-summarized). The combined output is translated into target_lang.

    Returns: doc_results, combined_tf_summary, combined_tfidf_summary, combined_abstractive_summary
    """
    doc_results = []
    all_en_sentences = []
    all_en_words = []

    for doc in docs:
        cleaned = clean_text(doc["raw_text"])
        lang = detect_language(cleaned)
        text_en = translate_text(cleaned, source=lang, target="en") if lang != "en" else cleaned

        words, sentences = tokenize_for_scoring(text_en)
        if sentences:
            tf_en = generate_summary_tf(words, sentences, individual_len)
            tfidf_en = generate_summary_tfidf(sentences, individual_len)
            abstractive_en = generate_summary_abstractive(text_en, summarizer, abs_max_len, abs_min_len, num_beams)
            all_en_sentences.extend(sentences)
            all_en_words.extend(words)
        else:
            tf_en, tfidf_en, abstractive_en = "", "", ""

        if lang != "en":
            tf_summary = translate_text(tf_en, source="en", target=lang)
            tfidf_summary = translate_text(tfidf_en, source="en", target=lang)
            abstractive_summary = translate_text(abstractive_en, source="en", target=lang) if abstractive_en else ""
        else:
            tf_summary, tfidf_summary, abstractive_summary = tf_en, tfidf_en, abstractive_en

        doc_results.append({
            "name": doc["name"],
            "lang": lang,
            "cleaned_text": cleaned,
            "tf_summary": tf_summary,
            "tfidf_summary": tfidf_summary,
            "abstractive_summary": abstractive_summary,
        })

    if not all_en_sentences:
        return doc_results, "", "", ""

    seen, unique_sentences = set(), []
    for s in all_en_sentences:
        key = s.strip().lower()
        if key not in seen:
            seen.add(key)
            unique_sentences.append(s)

    combined_tf_en = generate_combined_summary_tf(all_en_words, unique_sentences, combined_len, diversity)
    combined_tfidf_en = generate_combined_summary_tfidf(unique_sentences, combined_len, diversity)
    combined_text_en = " ".join(unique_sentences)
    combined_abstractive_en = generate_summary_abstractive(
        combined_text_en, summarizer, combined_abs_max_len, combined_abs_min_len, num_beams
    )

    if target_lang != "en":
        combined_tf_summary = translate_text(combined_tf_en, source="en", target=target_lang)
        combined_tfidf_summary = translate_text(combined_tfidf_en, source="en", target=target_lang)
        combined_abstractive_summary = (
            translate_text(combined_abstractive_en, source="en", target=target_lang)
            if combined_abstractive_en else ""
        )
    else:
        combined_tf_summary = combined_tf_en
        combined_tfidf_summary = combined_tfidf_en
        combined_abstractive_summary = combined_abstractive_en

    return doc_results, combined_tf_summary, combined_tfidf_summary, combined_abstractive_summary


# ── Export helpers (TXT / PDF) ───────────────────────────────────
def create_txt_report(sections):
    """sections: list of (title, content) tuples."""
    parts = [f"======{title}\n{content or '(disabled / empty)'}\n" for title, content in sections]
    return "\n".join(parts).encode("utf-8")


def create_txt(original_text, tf_summary, tfidf_summary, abstractive_summary):
    return create_txt_report([
        ("Original Text", original_text),
        ("TF Summary", tf_summary),
        ("TF-IDF Summary", tfidf_summary),
        ("Abstractive Summary (Transformers)", abstractive_summary),
    ])


def create_pdf_report(sections, doc_title="Text Summarizer - Output Report"):
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
        words = (text or "(disabled / empty)").split()
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
        y -= 10

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, doc_title)
    y -= 30

    for title, content in sections:
        draw_section_header(title)
        draw_paragraph(content)

    c.save()
    buffer.seek(0)
    return buffer


def create_pdf(original_text, tf_summary, tfidf_summary, abstractive_summary):
    return create_pdf_report([
        ("Original Text", original_text),
        ("TF Summary", tf_summary),
        ("TF-IDF Summary", tfidf_summary),
        ("Abstractive Summary (Transformers)", abstractive_summary),
    ])


# ── Display helpers ──────────────────────────────────────────────
def show_summary(text, s1, s2, s3, lang="en"):
    lang_label = LANG_NAMES.get(lang, lang)
    st.info(f"Detected language: **{lang_label}** ({lang})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original words", len(text.split()))
    c2.metric("TF words", len(s1.split()))
    c3.metric("TF-IDF words", len(s2.split()))
    c4.metric("Abstractive words", len(s3.split()) if s3 else 0)

    st.subheader("Original Text")
    st.write(text)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"TF Summary ({lang_label})")
        st.write(s1)
    with col2:
        st.subheader(f"TF-IDF Summary ({lang_label})")
        st.write(s2)

    st.subheader(f"Abstractive Summary — Transformers ({lang_label})")
    st.write(s3 if s3 else "_Abstractive summarization was disabled for this run._")

    st.markdown("---")
    st.subheader("Download Summary")

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        txt_data = create_txt(text, s1, s2, s3)
        st.download_button(
            "📄 Download Summary (.txt)", data=txt_data,
            file_name="summary.txt", mime="text/plain", use_container_width=True
        )
    with dl_col2:
        pdf_data = create_pdf(text, s1, s2, s3)
        st.download_button(
            "📑 Download Summary (.pdf)", data=pdf_data,
            file_name="summary.pdf", mime="application/pdf", use_container_width=True
        )


def show_multi_summary(doc_results, combined_tf, combined_tfidf, combined_abstractive, target_lang):
    st.subheader(f"Processed {len(doc_results)} document(s)")

    for doc in doc_results:
        lang_label = LANG_NAMES.get(doc["lang"], doc["lang"])
        with st.expander(f"📄 {doc['name']}  —  {lang_label}"):
            st.write("**Original text**")
            st.write(doc["cleaned_text"])
            st.write("**TF Summary**")
            st.write(doc["tf_summary"])
            st.write("**TF-IDF Summary**")
            st.write(doc["tfidf_summary"])
            st.write("**Abstractive Summary (Transformers)**")
            st.write(doc["abstractive_summary"] or "_Disabled for this run._")

    st.markdown("---")
    target_label = LANG_NAMES.get(target_lang, target_lang)
    st.subheader(f"Combined Multi-Document Summary ({target_label})")
    st.caption("Built from every document's sentences together, with near-duplicate sentences across files filtered out.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Combined TF Summary**")
        st.write(combined_tf)
    with c2:
        st.write("**Combined TF-IDF Summary (diversity-aware)**")
        st.write(combined_tfidf)
    with c3:
        st.write("**Combined Abstractive Summary**")
        st.write(combined_abstractive or "_Disabled for this run._")

    st.markdown("---")
    st.subheader("Download Combined Report")

    sections = [
        ("Combined TF Summary", combined_tf),
        ("Combined TF-IDF Summary", combined_tfidf),
        ("Combined Abstractive Summary (Transformers)", combined_abstractive),
    ]
    for doc in doc_results:
        sections.append((f"{doc['name']} - Original Text", doc["cleaned_text"]))
        sections.append((f"{doc['name']} - TF Summary", doc["tf_summary"]))
        sections.append((f"{doc['name']} - TF-IDF Summary", doc["tfidf_summary"]))
        sections.append((f"{doc['name']} - Abstractive Summary", doc["abstractive_summary"]))

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        txt_data = create_txt_report(sections)
        st.download_button(
            "📄 Download Report (.txt)", data=txt_data,
            file_name="multi_doc_summary.txt", mime="text/plain", use_container_width=True
        )
    with dl_col2:
        pdf_data = create_pdf_report(sections, doc_title="Multi-Document Summary Report")
        st.download_button(
            "📑 Download Report (.pdf)", data=pdf_data,
            file_name="multi_doc_summary.pdf", mime="application/pdf", use_container_width=True
        )


# ── App ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Text Summarizer", layout="wide")
st.title("Text Summarizer (Multilingual + Multi-Document + Abstractive)")

# Sidebar only collects settings here — it does NOT load the model. The model
# is loaded lazily, inside the button handlers below, so opening the app (or
# just browsing TF/TF-IDF results) never pays the Transformer load cost.
with st.sidebar:
    st.header("Abstractive summarization")
    enable_abstractive = st.checkbox(
        "Enable abstractive (Transformer) summary",
        value=True,
        help="Turn off to save memory — the model is never downloaded or loaded."
    )
    if enable_abstractive:
        model_name = st.selectbox(
            "Model",
            options=list(ABSTRACTIVE_MODELS.keys()),
            format_func=lambda k: ABSTRACTIVE_MODELS[k],
            index=0,
        )
        abs_max_len = st.slider("Summary length (max new tokens)", 30, 200, 100, step=10)
        abs_min_len = max(10, abs_max_len // 4)
        num_beams = st.slider(
            "Beam search width", 1, 4, 2,
            help="1 = fastest/lightest (greedy). 2 is a good speed/quality balance on free CPU tiers."
        )
    else:
        model_name, abs_max_len, abs_min_len, num_beams = None, 0, 0, 1

choice = st.radio("Choose input type", ["Direct Text", "TXT File", "PDF File", "Multiple Documents"])

if choice != "Multiple Documents":
    summary_len = st.text_input("Extractive summary length (sentences)", "3")
    summary_len = int(summary_len) if summary_len.isdigit() else 3

if choice == "Direct Text":
    text = st.text_area("Enter text")
    if st.button("Summarize"):
        summarizer = get_abstractive_summarizer(model_name) if enable_abstractive else None
        with st.spinner("Detecting language, translating, and summarizing..."):
            cleaned, s1, s2, s3, lang = summarize_pipeline(
                text, summary_len, summarizer, abs_max_len, abs_min_len, num_beams
            )
        show_summary(cleaned, s1, s2, s3, lang)

elif choice == "TXT File":
    file = st.file_uploader("Upload TXT", type=["txt"])
    if file:
        raw_text = file.read().decode("utf-8", errors="ignore")
        if st.button("Summarize"):
            summarizer = get_abstractive_summarizer(model_name) if enable_abstractive else None
            with st.spinner("Detecting language, translating, and summarizing..."):
                cleaned, s1, s2, s3, lang = summarize_pipeline(
                    raw_text, summary_len, summarizer, abs_max_len, abs_min_len, num_beams
                )
            show_summary(cleaned, s1, s2, s3, lang)

elif choice == "PDF File":
    file = st.file_uploader("Upload PDF", type=["pdf"])
    if file:
        if st.button("Summarize"):
            reader = PdfReader(file)
            raw_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    raw_text += page_text
            summarizer = get_abstractive_summarizer(model_name) if enable_abstractive else None
            with st.spinner("Detecting language, translating, and summarizing..."):
                cleaned, s1, s2, s3, lang = summarize_pipeline(
                    raw_text, summary_len, summarizer, abs_max_len, abs_min_len, num_beams
                )
            show_summary(cleaned, s1, s2, s3, lang)

elif choice == "Multiple Documents":
    files = st.file_uploader(
        "Upload multiple TXT/PDF files", type=["txt", "pdf"], accept_multiple_files=True
    )

    col_a, col_b = st.columns(2)
    with col_a:
        indiv_len = st.text_input("Per-document summary length (sentences)", "2")
        indiv_len = int(indiv_len) if indiv_len.isdigit() else 2
    with col_b:
        combined_len = st.text_input("Combined summary length (sentences)", "5")
        combined_len = int(combined_len) if combined_len.isdigit() else 5

    target_lang = st.selectbox(
        "Output language for combined summary",
        options=list(LANG_NAMES.keys()),
        format_func=lambda code: LANG_NAMES[code],
        index=0,
    )
    diversity = st.slider(
        "Diversity (reduce repeated content across documents)",
        min_value=0.0, max_value=0.9, value=0.3, step=0.1
    )

    if files and st.button("Summarize Documents"):
        docs = [{"name": f.name, "raw_text": extract_text_from_uploaded(f)} for f in files]
        summarizer = get_abstractive_summarizer(model_name) if enable_abstractive else None
        with st.spinner(f"Detecting languages, translating, and summarizing {len(docs)} document(s)..."):
            doc_results, combined_tf, combined_tfidf, combined_abstractive = summarize_multi_documents(
                docs, indiv_len, combined_len, target_lang, summarizer,
                diversity, abs_max_len, abs_min_len, num_beams,
                combined_abs_max_len=min(300, abs_max_len + 50) if enable_abstractive else 0,
                combined_abs_min_len=max(20, abs_min_len + 20) if enable_abstractive else 0,
            )
        show_multi_summary(doc_results, combined_tf, combined_tfidf, combined_abstractive, target_lang)