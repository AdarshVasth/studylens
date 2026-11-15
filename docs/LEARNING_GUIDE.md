# Understand StudyLens

## The problem

Searching a long PDF by hand is slow. StudyLens breaks notes into passages and ranks the passages most relevant to a question. An optional language model converts those passages into an explanation.

## Five concepts to explain

1. **Chunking:** split each page into 140-word windows with 25-word overlap. This preserves nearby context without losing the original page number.
2. **TF-IDF:** words that are frequent in one passage but uncommon across the collection carry more weight. Cosine similarity ranks lexical matches.
3. **Embeddings:** a neural encoder maps text into vectors; similar meanings can have nearby vectors even when words differ.
4. **RRF:** combine search rankings rather than comparing incompatible raw scores. Each retrieved rank contributes `1 / (60 + rank)`.
5. **RAG:** retrieve evidence, place it in the language model context, generate an answer, and show sources. It reduces unsupported output but does not eliminate it.

## Trace one request

- `app/static/app.js` sends the question to `/api/ask`.
- `app/main.py` validates the input and calls `library.search`.
- `app/core.py` loads stored chunks and computes ranked matches.
- With generation off, the API returns extracted passages.
- With generation on, the API sends only the question and retrieved passages to the configured Ollama endpoint, then validates the answer structure and source IDs.
- The frontend inserts results using `textContent`, keeping uploaded text and generated text from being interpreted as HTML.

## Honest interview walkthrough

Say: “This project implements a local RAG pipeline with a TF-IDF baseline, optional dense retrieval, rank fusion, source references, and local generation. I can measure retrieval separately from answer generation.”

Then demonstrate an unknown topic: retrieval should return no passage in the lexical baseline. Show how keyword overlap can still return an irrelevant passage and explain why a threshold alone does not prove relevance. Show a malformed citation response in the unit test and explain the fallback.

## Experiments to make it yours

- Write 30 questions from different subjects before tuning retrieval. Reserve some for the final evaluation.
- Compare lexical and hybrid hit@1/hit@3 on the same held-out questions.
- Compare chunk sizes and record latency, memory use, and retrieval outcomes.
- Run the real local model. Check each answer for factual support, citation correctness, and unsupported statements separately.
- Add OCR or a reranker, then measure whether it helps.

Do not claim to have trained the embedding or language model. They are pretrained models integrated into an application.
