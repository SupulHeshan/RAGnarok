from pathlib import Path
import sqlite3

# ────────────────────────────────────────────────────────────────────────────
# Database setup (SQLite)
# ────────────────────────────────────────────────────────────────────────────

DB_PATH = str(Path(__file__).resolve().parent / "documents.db")
SIMILARITY_THRESHOLD = 0.5  # configurable threshold for refusal

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Ensure fresh schema by dropping both tables before recreation
    c.execute("DROP TABLE IF EXISTS chunks")
    c.execute("DROP TABLE IF EXISTS documents")
    c.execute("DROP TABLE IF EXISTS retrieval_cache")
    c.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            path TEXT,
            size_bytes INTEGER,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER,
            chunk_index INTEGER,
            text TEXT,
            page INTEGER,
            embedding BLOB,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        )
    """)
    # Cache retrieval results for deterministic grounding: keyed by (qhash, doc_id, k)
    c.execute("""
        CREATE TABLE retrieval_cache (
            qhash TEXT,
            doc_id INTEGER,
            k INTEGER,
            query TEXT,
            result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (qhash, doc_id, k)
        )
    """)
    conn.commit()
    conn.close()