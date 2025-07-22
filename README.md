# chatbot
Notes:

torch is required as backend for transformers and sentence-transformers.

faiss-cpu for similarity search.

scikit-learn for cosine similarity calculation in deduplication.


# Context-Aware Sentiment Chatbot with Deduplication

This is a simple Flask-based chatbot application that:
- Uses a **retriever** with FAISS and Sentence Transformers to fetch relevant context based on user queries and selected categories.
- Performs **sentiment analysis** on combined user input and retrieved context.
- Removes **duplicate and near-duplicate documents** before indexing to improve retrieval quality.
- Optionally can be extended to use generative models for richer replies.

## Features

- Dynamic context retrieval based on user query + category.
- Duplicate and near-duplicate detection and removal in documents.
- Sentiment analysis with Hugging Face Transformers.
- Simple web UI with category selection and chat interface.

## Installation

1. Clone the repo:


Install dependencies:

pip install -r requirements.txt
Prepare your documents in data/docs.txt, one context line per entry.

Running the app

python app.py
Open your browser at http://127.0.0.1:5000

Project Structure

your_project/
├── app.py
├── retriever.py
├── data/
│   └── docs.txt
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
Notes
You can adjust the duplicate detection threshold in app.py (filter_similar_docs function).

For better responses, integrate a generative model (like flan-t5 or OpenAI GPT) in the /analyze route.

For large datasets, consider persisting the FAISS index to avoid rebuilding every run.

License
MIT License

Enjoy building your smart context-aware chatbot! 🚀


If you want, I can help generate a full ZIP with everything or add generative reply support next. Just let me know!

