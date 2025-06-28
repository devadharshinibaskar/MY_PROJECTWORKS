import os
import nltk
import streamlit as st
from numpy.f2py.crackfortran import quiet
from transformers import pipeline
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import PyPDF2
import ssl
import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# Check if SSL certificate can be verified
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Download NLTK data
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab',quiet=True)
        nltk.download('stopwords', quiet=True)
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"An error occurred while downloading NLTK data: {e}")


# Set NLTK data path (if needed)
nltk.data.path.append(os.path.join(os.path.expanduser("~"), "nltk_data"))


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def preprocess_text(text):
    """Preprocess the extracted text by removing newlines and extra spaces."""
    return ' '.join(text.split())


def summarize_text(text, max_length=150, min_length=50):
    """Summarize the given text using a pre-trained model."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    max_chunk_length = 1024
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]

    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    return ' '.join(summaries)


def extract_key_takeaways(text, num_takeaways=5):
    """Extract key takeaways from the text."""
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]

    word_freq = Counter(words)
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_freq[word]
                else:
                    sentence_scores[sentence] += word_freq[word]

    key_takeaways = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_takeaways]
    return key_takeaways


def main():
    download_nltk_data()  # Ensure NLTK data is downloaded
    st.title("AI/ML Paper Summarizer")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Processing the PDF..."):
            raw_text = extract_text_from_pdf(uploaded_file)
            processed_text = preprocess_text(raw_text)

            summary = summarize_text(processed_text)
            key_takeaways = extract_key_takeaways(processed_text)

        st.subheader("Summary")
        st.write(summary)

        st.subheader("Key Takeaways")
        for i, takeaway in enumerate(key_takeaways, 1):
            st.write(f"{i}. {takeaway}")

        if st.checkbox("Show full text"):
            st.subheader("Full Text")
            st.text_area("", processed_text, height=300)


if __name__ == "__main__":
    main()
