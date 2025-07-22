from flask import Flask, request, render_template, jsonify
from transformers import pipeline
from retriever import ContextRetriever
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)


def remove_duplicates(docs):
    seen = set()
    unique_docs = []
    for doc in docs:
        if doc not in seen:
            unique_docs.append(doc)
            seen.add(doc)
    return unique_docs

def filter_similar_docs(docs, threshold=0.9):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs, convert_to_numpy=True)

    keep = []
    for i, emb in enumerate(embeddings):
        # Check similarity with already kept docs
        if all(cosine_similarity([emb], [embeddings[j]])[0][0] < threshold for j in keep):
            keep.append(i)
    filtered_docs = [docs[i] for i in keep]
    return filtered_docs
    



# Load documents
with open("data/docs.txt", "r") as f:
    documents = [line.strip() for line in f.readlines() if line.strip()]
    
documents = remove_duplicates(documents)
documents = filter_similar_docs(documents, threshold=0.9)

print(f"Loaded {len(documents)} unique documents after deduplication.")    

retriever = ContextRetriever(docs=documents)

# Sentiment analyzer (optional, you can still keep this)
sentiment_analyzer = pipeline("sentiment-analysis")

# Generative text model (use a small/flan-t5 for demo)
generator = pipeline("text2text-generation", model="google/flan-t5-small")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    user_input = data.get("message", "").strip()
    category = data.get("category", "General")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    query_with_category = f"[{category}] {user_input}"
    retrieved_context = retriever.retrieve(query_with_category)
    context_text = " ".join(retrieved_context)

    # Combine context + user input for generation prompt
    prompt = f"Context: {context_text}\nUser question: {user_input}\nAnswer:"

    # Generate response
    generated = generator(prompt, max_length=150, do_sample=True, temperature=0.7)
    generated_text = generated[0]['generated_text']

    # (Optional) Sentiment on combined input
    sentiment = sentiment_analyzer(f"{context_text} {user_input}")[0]
    label = sentiment["label"]
    score = round(sentiment["score"], 2)

    return jsonify({
        "sentiment": label,
        "confidence": score,
        "category": category,
        "context_used": retrieved_context,
        "response": generated_text
    })

if __name__ == "__main__":
    app.run(debug=True)
