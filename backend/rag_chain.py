"""
Improved Retrieval-Augmented Generation (RAG) chain that:
  • Uses smaller, strategic text chunks for more precise retrieval
  • Implements similarity search with limited results
  • Handles off-topic questions properly
  • Supports multiple document formats including PDFs
"""

from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseRetriever, Document
from langchain.memory import ConversationSummaryMemory

import os
import time
import logging
import sqlite3
from dotenv import load_dotenv
from SQLite import init_db, DB_PATH, SIMILARITY_THRESHOLD
from typing import Optional, List, Tuple
import hashlib
import json
from pathlib import Path
import numpy as np
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  Load environment variables (.env should contain MISTRAL_API_KEY=sk‑***)
load_dotenv()

# Database setup (SQLite)
init_db()

# Directory for FAISS vectorstores (one per document)
VECTORSTORE_DIR = Path(__file__).resolve().parent / "vectorstores"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)



# ────────────────────────────────────────────────────────────────────────────
#  Document loading helper functions
# ────────────────────────────────────────────────────────────────────────────
def load_document(document_path: str):
    """Load a document based on its file extension."""
    logger.info(f"Attempting to load document: {document_path}")
    
    if not os.path.exists(document_path):
        logger.error(f"❌ {document_path} file not found. Please create this file before continuing.")
        raise FileNotFoundError(f"{document_path} file not found in the current directory")
    
    file_extension = Path(document_path).suffix.lower()
    
    if file_extension == '.pdf':
        logger.info(f"Loading PDF document: {document_path}")
        loader = PyPDFLoader(document_path)
        docs = loader.load()
        logger.info(f"✅ Successfully loaded PDF: {document_path} with {len(docs)} pages")
        return docs
    elif file_extension in ['.txt', '.md']:
        logger.info(f"Loading text document: {document_path}")
        loader = TextLoader(document_path)
        docs = loader.load()
        logger.info(f"✅ Successfully loaded text file: {document_path}")
        return docs
    else:
        logger.error(f"❌ Unsupported file type: {file_extension}")
        raise ValueError(f"Unsupported file type: {file_extension}. Please use .txt, .md, or .pdf files")

# ────────────────────────────────────────────────────────────────────────────
#  SQLite-based retriever using cosine similarity
# ────────────────────────────────────────────────────────────────────────────
def embed_and_store(document_path: str, chunks, embeddings_model) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    size_bytes = os.path.getsize(document_path)
    c.execute("INSERT INTO documents (name, path, size_bytes) VALUES (?, ?, ?)",
              (Path(document_path).name, str(document_path), size_bytes))
    doc_id = c.lastrowid

    for idx, ch in enumerate(chunks):
        emb = embeddings_model.embed_query(ch.page_content)
        emb_bytes = np.array(emb, dtype=np.float32).tobytes()
        c.execute("INSERT INTO chunks (doc_id, chunk_index, text, page, embedding) VALUES (?, ?, ?, ?, ?)",
                  (doc_id, idx, ch.page_content, ch.metadata.get("page", None), emb_bytes))
    conn.commit()
    conn.close()

    # Build FAISS vectorstore for this document to accelerate retrieval on large PDFs
    try:
        faiss_index = FAISS.from_documents(chunks, embeddings_model)
        index_dir = VECTORSTORE_DIR / f"doc_{doc_id}"
        # ensure index path exists
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss_index.save_local(str(index_dir))
        logger.info(f"Saved FAISS index for doc_id={doc_id} at {index_dir}")
    except Exception as e:
        logger.warning(f"Failed to build/save FAISS index for doc_id={doc_id}: {e}")

    return doc_id

def retrieve_top_k(query: str, k: int, embeddings_model, doc_id: Optional[int] = None) -> List[Tuple[str, int, float]]:
    # Deterministic grounding: cache retrieval results per normalized question hash
    def normalize_q(q: str) -> str:
        # Lowercase, strip whitespace, collapse internal whitespace
        s = " ".join(q.strip().lower().split())
        return s

    qnorm = normalize_q(query)
    qhash = hashlib.sha256(qnorm.encode('utf-8')).hexdigest()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check cache first
    c.execute("SELECT result FROM retrieval_cache WHERE qhash=? AND doc_id IS ? AND k=?", (qhash, doc_id, k))
    row = c.fetchone()
    if row:
        try:
            serialized = row[0]
            parsed = json.loads(serialized)
            # parsed expected to be list of [text, page, score]
            logger.info(f"Retrieval cache HIT for qhash={qhash} doc_id={doc_id} k={k}")
            conn.close()
            return [(t, p, float(s)) for t, p, s in parsed]
        except Exception:
            # Fallthrough to compute if cache deserialization fails
            logger.warning('Failed to deserialize retrieval cache; recomputing')

    # No cache hit — prefer FAISS vectorstore if available for this document
    topk: List[Tuple[str, int, float]] = []
    if doc_id is not None:
        index_dir = VECTORSTORE_DIR / f"doc_{doc_id}"
        if index_dir.exists():
            try:
                faiss_index = FAISS.load_local(str(index_dir), embeddings_model)
                docs_and_scores = faiss_index.similarity_search_with_score(query, k=k)
                for doc, score in docs_and_scores:
                    page = doc.metadata.get('page') if getattr(doc, 'metadata', None) else None
                    topk.append((doc.page_content, page, float(score)))
            except Exception as e:
                logger.warning(f"Failed to use FAISS index for doc_id={doc_id}: {e}. Falling back to SQLite scoring.")

    # If FAISS not used or produced no results, fallback to SQLite scoring
    if not topk:
        query_emb = np.array(embeddings_model.embed_query(query), dtype=np.float32)
        if doc_id:
            c.execute("SELECT id, text, page, embedding FROM chunks WHERE doc_id=?", (doc_id,))
        else:
            c.execute("SELECT id, text, page, embedding FROM chunks")
        rows = c.fetchall()

        scored = []
        for chunk_id, text, page, emb_bytes in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            # avoid zero division
            denom = (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            sim = 0.0
            if denom != 0:
                sim = float(np.dot(query_emb, emb) / denom)
            scored.append((text, page, sim))

        scored.sort(key=lambda x: x[2], reverse=True)
        topk = scored[:k]

    # Store in cache for future deterministic grounding
    try:
        serialized = json.dumps([[t, p, s] for t, p, s in topk])
        c.execute("INSERT OR REPLACE INTO retrieval_cache (qhash, doc_id, k, query, result) VALUES (?, ?, ?, ?, ?)",
                  (qhash, doc_id, k, qnorm, serialized))
        conn.commit()
        logger.info(f"Stored retrieval cache for qhash={qhash} doc_id={doc_id} k={k} entries={len(topk)}")
    except Exception:
        logger.warning('Failed to write to retrieval cache')

    conn.close()
    return topk

# ────────────────────────────────────────────────────────────────────────────
#  Custom BaseRetriever implementation
# ────────────────────────────────────────────────────────────────────────────
class SQLiteRetriever(BaseRetriever):
    embeddings_model: MistralAIEmbeddings
    doc_id: int
    threshold: float = SIMILARITY_THRESHOLD
    
    def __init__(self, embeddings_model: MistralAIEmbeddings, doc_id: int, threshold: float = SIMILARITY_THRESHOLD):
        super().__init__(embeddings_model=embeddings_model, doc_id=doc_id, threshold=threshold)
        # No need to set attributes manually as they are handled by Pydantic

    def _get_relevant_documents(self, query: str) -> List[Document]:
        logger.info(f"SQLiteRetriever invoked for doc_id={self.doc_id} query='{query[:80]}'")
        top_chunks = retrieve_top_k(query, 3, self.embeddings_model, self.doc_id)
        filtered = [(text, page, score) for text, page, score in top_chunks if score >= self.threshold]
        return [Document(page_content=text, metadata={"page": page, "score": score}) for text, page, score in filtered]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

# ────────────────────────────────────────────────────────────────────────────
#  Improved RAG chain builder
# ────────────────────────────────────────────────────────────────────────────
def get_rag_chain(document_path="essay.txt"):
    """Return an improved ConversationalRetrievalChain with better chunking and retrieval."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError(
            "❌  MISTRAL_API_KEY not found.  Add it to your .env file."
        )

    # 1️⃣  Load document and split into strategic chunks
    try:
        # Use the document loader helper function
        docs = load_document(document_path)
    except Exception as e:
        logger.error(f"❌ Error loading document: {str(e)}")
        raise ValueError(f"Failed to load {document_path}: {str(e)}")
    
    # Use smaller chunks with some overlap for better context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,        # Smaller chunks (about 1-2 paragraphs)
        chunk_overlap=100,      # Some overlap to maintain context between chunks
        separators=["\n## ", "\n\n", "\n", " ", ""]  # Split on section markers first
    )
    
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Document split into {len(chunks)} chunks")

    # 2️⃣  Embed chunks with Mistral and create optimized retriever
    embeddings = MistralAIEmbeddings(
        model="mistral-embed", 
        mistral_api_key=api_key
    )
    
    doc_id = embed_and_store(document_path, chunks, embeddings)

    # 3️⃣  Set up language models
       # Higher temperature for more creative summaries
    summary_llm = ChatMistralAI(
        mistral_api_key=api_key, 
        temperature=0.3,
        model="mistral-large-latest"
    )

    # Lower temperature for factual answers
    qa_llm = ChatMistralAI(
        mistral_api_key=api_key, 
        temperature=0.1,
        model="mistral-large-latest"
    )
    
    logger.info("Created Mistral models for summaries and QA")

    # 4️⃣  Conversation‑summary memory - using default prompt
    memory = ConversationSummaryMemory(
        llm=summary_llm,
        memory_key="chat_history",
        return_messages=True,
        verbose=True,
        output_key="answer"  # Explicitly tell memory which output key to store
    )
    
    logger.info("Created conversation summary memory")

    # 5️⃣ Prompt enforcing inline citations + JSON sources
    qa_prompt = ChatPromptTemplate.from_template(
        """You are an assistant that answers user clinical questions strictly using only the provided context snippets.
<context>
{context}
</context>

Chat History:
{chat_history}

Question: {question}

Guidelines:

1. If the question is a greeting (like "hello", "hi", etc.), respond with a friendly greeting.
2. If the question is small talk or casual conversation, respond naturally without requiring info from the context and with empty sources.
3. If the answer is IN THE CONTEXT, Always return JSON with exactly these keys: answer, sources.
4. "answer": The concise answer text. Include inline citations like [doc:ID, p:PAGE] directly in the answer body.
5. "sources": an array of JSON objects, each {{"doc_id": <int>, "page": <int>, "quote": "<exact quote>"}}.
6. refuse if retrieval score is low.(e.g., low similarity).
7. Never invent information not present in the context or chat history.
8. If retrieval scores are too low, answer="I don't have information about that in my knowledge base." with empty sources.

Return ONLY a valid JSON object, same as exactly example output. no extra text.
"""
    )

    retriever = SQLiteRetriever(embeddings, doc_id, SIMILARITY_THRESHOLD)

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=qa_llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=True,
        return_source_documents=True,
        return_generated_question=False,
        rephrase_question=False,
    )
    return rag_chain



# ────────────────────────────────────────────────────────────────────────────
#  Safe wrapper to retry on rate‑limit errors
# ────────────────────────────────────────────────────────────────────────────
def invoke_with_retry(chain, input_data: dict, max_retries: int = 3) -> Optional[dict]:
    """
    Call chain.invoke(input_data) with automatic exponential-backoff retries
    when the Mistral API returns a rate-limit error.
    """
    for attempt in range(max_retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {str(e)}")
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s ...
                logger.info(f"Rate limit hit, waiting {wait}s before retry")
                time.sleep(wait)
                continue
            raise  # re‑raise anything else
    return None 