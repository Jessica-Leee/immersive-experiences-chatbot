# Chatbot for Immersive Experiences

A Streamlit prototype for questions about immersive experiences, AR, VR, creator workflows, and the BLCKUNICRN platform. It combines a document-retrieval workflow with an external language-model response.

## Implemented workflow

1. `embedding_generator.py` extracts PDF text with PyMuPDF, splits it into token-sized chunks, requests OpenAI embeddings, and prepares Pinecone records containing source and chunk metadata.
2. `streamlit_app.py` queries a Pinecone index through LangChain's `RetrievalQA` chain.
3. The QA result is added to a platform overview and the current question, then sent to the DeepSeek chat API for the displayed answer.
4. Streamlit keeps and displays chat messages for the current session. If retrieval initialization fails or returns no result, the app can fall back to its built-in platform overview.

This is an integration of Streamlit, OpenAI embeddings, Pinecone, LangChain, and DeepSeek. It does not train a language model.

## Setup and launch

`runtime.txt` targets Python 3.10. Create an environment and install the listed dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Configure these names through environment variables or Streamlit secrets:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `DEEPSEEK_API_KEY`

A local `.env` file is also supported when `python-dotenv` is installed. Keep configuration values out of source control. The app currently uses the hard-coded Pinecone index name `blckunicrn`; it does not read `PINECONE_ENV` or `PINECONE_INDEX_NAME`.

With the services configured and the index prepared:

```sh
streamlit run streamlit_app.py
```

`streamlit_app.py` checks for missing API keys and stops with an explanatory message. `app.py` is an alternate implementation; the command above uses the entry point with that guard.

## Data preparation and limitations

The ingestion file is **`embedding_generator.py`**. Its `SOURCES` and `IDs` lists select local PDFs; the source PDFs are not included in this repository. Supply documents you can use and retain their source attribution.

The ingestion helper currently references `Pinecone` and `ServerlessSpec` without importing those symbols. It also directly imports `tiktoken`, which is not explicitly listed in `requirements.txt`. Resolve those issues and check SDK compatibility before running ingestion. The script makes embedding requests and writes to Pinecone; it is not an offline preparation step.

The repository has no retrieval benchmark or answer-quality evaluation. Answers are not guaranteed to be supported by the documents, and the final response does not display source citations. Chat history is shown in the interface, but the outgoing DeepSeek request contains the current constructed prompt rather than the full conversation.
