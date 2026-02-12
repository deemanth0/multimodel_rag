# Advanced RAG UI App

Modern Streamlit UI for your notebook workflow:
- Upload `.txt` or `.pdf`
- Build TF-IDF embeddings + FAISS vector index
- Ask grounded questions using Groq LLM responses
- Inspect retrieved chunks in the chat UI

## Files

- `main.py`: reusable RAG engine (ingestion, chunking, retrieval, answer generation)
- `app.py`: modern Streamlit UI
- `requirements.txt`: Python dependencies

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Usage

1. Enter your `GROQ_API_KEY` in the sidebar form.
2. Upload a `.txt` or `.pdf` document.
3. Click **Build Index**.
4. Ask questions in the chat box.

## Notes

- Answers are constrained to retrieved context.
- If context is insufficient, the app returns a fallback response.
- This project does not commit API keys in source code.
