# Immersive Experiences Chatbot

A Streamlit prototype for questions about immersive experiences, AR, VR, creator workflows, and the BLCKUNICRN platform. It integrates document retrieval with an external language-model response; it does not train a language model.

## Implemented workflow

1. `embedding_generator.py` extracts PDF text with PyMuPDF, splits it into token chunks, requests OpenAI embeddings, and writes source/chunk metadata to Pinecone.
2. `streamlit_app.py` queries Pinecone through LangChain's `RetrievalQA` chain.
3. The QA result and a platform overview become context for a DeepSeek chat request.
4. Streamlit displays chat messages for the current session. If retrieval is unavailable, the app can fall back to its built-in overview.

The ingestion helper uses `text-embedding-ada-002` and a 1,536-dimensional index. The application and ingestion script currently share the hard-coded index name `blckunicrn`.

## Setup

`runtime.txt` targets Python 3.10:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Provide these configuration names locally, through environment variables or Streamlit secrets:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `DEEPSEEK_API_KEY`

The `.env.example` file contains no values. Keep the populated `.env` and Streamlit secrets out of source control. An absent Streamlit secrets file falls back to environment configuration.

## Documents and launch

The source PDFs are not included. Supply documents you can use, retain their source attribution, and update the `SOURCES` and `IDs` lists in **`embedding_generator.py`**.

This explicit command creates embeddings and writes to Pinecone:

```sh
python embedding_generator.py
```

Importing the helper does not read `.env`, open PDFs, or contact services. The script uses the Pinecone v3 `list_indexes().names()` interface and creates its index only when absent.

With the services configured and the index prepared:

```sh
streamlit run streamlit_app.py
```

This entry point checks required keys and stops with an explanatory message if they are missing. `app.py` remains an alternate UI implementation.

## Offline checks

The checks use only the Python standard library and replace external SDKs at the import boundary:

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
```

They cover environment fallback, safe helper imports, missing-key handling, chunk boundaries, the index-exists/create paths, vector/chunk validation, and stable source metadata. They never call OpenAI, Pinecone, DeepSeek, or Streamlit services and do not read a secrets file.

## Limitations

The dependency installation and live service workflow have not been verified by these offline checks. The document corpus and retrieval/answer benchmarks are not included. Answers may be unsupported, and the final UI does not display source citations. Chat history is displayed, but the outgoing DeepSeek request includes only the current constructed prompt rather than the full conversation.
