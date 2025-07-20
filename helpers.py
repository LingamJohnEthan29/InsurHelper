import os
import docx 
import nltk
#nltk.data.path.append('/home/codespace/nltk_data')
import PyPDF2
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss 
#import spacy
from sentence_transformers import SentenceTransformer,util
#nlp = spacy.load("en_core_web_sm")
import regex as re


model = SentenceTransformer("all-MiniLM-L6-v2")

def get_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

def simple_sentence_tokenize(text):
    # Splits on .!? followed by a space or end of line
    return re.findall(r'[^.!?]+[.!?]', text)

def extract_best_sentence(text, query, model):
    sentences = simple_sentence_tokenize(text)
    query_emb = model.encode(query, convert_to_tensor=True)

    best_score = -1
    best_sentence = ''
    for sent in sentences:
        sent_emb = model.encode(sent, convert_to_tensor=True)
        score = util.cos_sim(query_emb, sent_emb).item()
        if score > best_score:
            best_score = score
            best_sentence = sent

    return best_sentence



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

def chunk_text(text):
    lines = text.split('\n')
    current_heading = None
    chunks = []
    current_chunk = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect new heading
        if re.match(r'^(table of benefits|exclusions?)', line.lower()):
            # Save the previous chunk with its heading
            if current_chunk:
                chunks.append((current_heading, ' '.join(current_chunk)))
                current_chunk = []
            current_heading = line
        else:
            current_chunk.append(line)

    # Append the last chunk
    if current_chunk:
        chunks.append((current_heading, ' '.join(current_chunk)))

    return chunks


def find_best_match(weighted_chunks, query):
    query_emb = model.encode(query, convert_to_tensor=True)

    best_score = -1
    best_chunk = ''
    for chunk, weight in weighted_chunks:
        chunk_emb = model.encode(chunk, convert_to_tensor=True)
        score = util.cos_sim(query_emb, chunk_emb).item() * weight
        if score > best_score:
            best_score = score
            best_chunk = chunk

    return best_chunk, best_score