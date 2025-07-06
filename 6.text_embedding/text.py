import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import os
import requests
import zipfile

# Sample corpus
corpus = [
    "I love natural language processing",
    "Deep learning is revolutionizing AI",
    "Word embeddings capture semantic meaning",
    "BERT provides contextual embeddings",
    "TF-IDF is a simple but effective technique"
]

# ----------------------------- #
# 1. TF-IDF Embeddings
# ----------------------------- #
print("\n🔹 TF-IDF Embeddings")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus).toarray()

print("TF-IDF vector shape:", tfidf_matrix.shape)
print("TF-IDF for sentence 1:", tfidf_matrix[0])

# ----------------------------- #
# 2. Word2Vec Embeddings (Average)
# ----------------------------- #
print("\n🔹 Word2Vec Embeddings (average of word vectors)")
tokenized_corpus = [sent.lower().split() for sent in corpus]
word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

def get_avg_w2v_vector(sentence):
    words = sentence.lower().split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(100)

w2v_vectors = np.array([get_avg_w2v_vector(sent) for sent in corpus])
print("Word2Vec vector shape:", w2v_vectors.shape)
print("Word2Vec for sentence 1:", w2v_vectors[0][:5])

# ----------------------------- #
# 3. GloVe Embeddings (Average)
# ----------------------------- #
print("\n🔹 GloVe Embeddings (average of word vectors)")
glove_path = "glove.6B.100d.txt"
if not os.path.exists(glove_path):
    print("Downloading GloVe...")
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    r = requests.get(url)
    with open("glove.6B.zip", "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile("glove.6B.zip", "r") as zip_ref:
        zip_ref.extractall()

# Load GloVe vectors
glove_vectors = {}
with open(glove_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vector = np.array(parts[1:], dtype=np.float32)
        glove_vectors[word] = vector

def get_avg_glove_vector(sentence):
    words = sentence.lower().split()
    vectors = [glove_vectors[word] for word in words if word in glove_vectors]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

glove_embeddings = np.array([get_avg_glove_vector(sent) for sent in corpus])
print("GloVe vector shape:", glove_embeddings.shape)
print("GloVe for sentence 1:", glove_embeddings[0][:5])

# ----------------------------- #
# 4. Sentence-BERT Embeddings
# ----------------------------- #
print("\n🔹 Sentence-BERT Embeddings")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_embeddings = sbert_model.encode(corpus)

print("SBERT vector shape:", sbert_embeddings.shape)
print("SBERT for sentence 1:", sbert_embeddings[0][:5])
