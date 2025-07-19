import os
import docx 
import nltk
nltk.data.path.append('/home/codespace/nltk_data')
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy

nlp = spacy.load("en_core_web_sm")

def get_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def extract_text(filepath):
    ext = filepath.split('.')[-1].lower()
    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == 'docx':
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    elif ext == 'pdf':
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        return ""

def chunk_text(text, chunk_size=5):
    sentences = get_sentences(text)
    return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

def find_best_match(chunks, query):
    vectorizer = TfidfVectorizer().fit_transform([query] + chunks)
    similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    best_index = similarities.argmax()
    return chunks[best_index], similarities[best_index]
