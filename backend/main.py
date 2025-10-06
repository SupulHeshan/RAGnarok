# main.py ────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from rag_chain import get_rag_chain, invoke_with_retry, DB_PATH
import logging
import os
import shutil
from typing import List
from datetime import datetime
from pathlib import Path
import json, re
import sqlite3

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="RAGnarok API",
    description="A Retrieval-Augmented Generation chatbot API with memory",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
active_document = "essay.txt"  # Default document
document_metadata = {}  # Store metadata for each document

# Build the RAG chain once at startup
rag_chain = get_rag_chain(document_path=active_document)

# Request/Response Models
class Query(BaseModel):
    input: str = Field(..., min_length=1, max_length=1000)

class DocumentInfo(BaseModel):
    filename: str
    size: int
    active: bool
    last_modified: datetime
    upload_date: datetime

class ErrorResponse(BaseModel):
    detail: str
    code: str
    timestamp: datetime = Field(default_factory=datetime.now)

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail="Internal server error",
                code="INTERNAL_ERROR"
            ).dict()
        )

# Health check with detailed status
@app.get("/health")
async def health_check():
    try:
        # Check if RAG chain is properly initialized
        chain_status = "ok" if rag_chain is not None else "error"
        
        # Check if active document exists
        doc_status = "ok" if os.path.exists(active_document) else "error"
        
        # Check available memory
        memory_status = "ok" if hasattr(rag_chain, 'memory') else "error"
        
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "rag_chain": chain_status,
                "active_document": doc_status,
                "memory": memory_status
            },
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Upload document with metadata
@app.post("/files", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload a document to be used for RAG"""
    global rag_chain, active_document, document_metadata
    
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )
    
    # Validate file type
    if not file.filename.endswith((".txt", ".md", ".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .md, and .pdf files are supported"
        )
    
    # Save the uploaded file with metadata
    file_location = UPLOAD_DIR / file.filename
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Store metadata
        document_metadata[file.filename] = {
            "upload_date": datetime.now(),
            "last_modified": datetime.now(),
            "size": file.size
        }
        
        # Update RAG chain in background
        if background_tasks:
            background_tasks.add_task(update_rag_chain, file_location)
        else:
            await update_rag_chain(file_location)
        
        return DocumentInfo(
            filename=file.filename,
            size=file.size,
            active=True,
            last_modified=document_metadata[file.filename]["last_modified"],
            upload_date=document_metadata[file.filename]["upload_date"]
        )
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )

async def update_rag_chain(file_location: Path):
    """Update RAG chain with new document"""
    global rag_chain, active_document
    try:
        active_document = str(file_location)
        rag_chain = get_rag_chain(document_path=active_document)
        logger.info(f"Successfully updated RAG chain with {file_location}")
        # Clear retrieval cache after re-embedding a document to avoid stale results
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute("DELETE FROM retrieval_cache")
            conn.commit()
            conn.close()
            logger.info('Cleared retrieval cache after updating RAG chain')
        except Exception as e:
            logger.warning(f'Failed to clear retrieval cache: {e}')
    except Exception as e:
        logger.error(f"Failed to update RAG chain: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize RAG chain: {str(e)}"
        )

# List available documents with metadata
@app.get("/docs", response_model=List[DocumentInfo])
async def list_documents():
    """List all available documents with metadata"""
    documents = []
    
    for filename in os.listdir(UPLOAD_DIR):
        if filename.endswith((".txt", ".md", ".pdf")):
            file_path = UPLOAD_DIR / filename
            size = os.path.getsize(file_path)
            is_active = (filename == active_document)
            
            # Get or create metadata
            if filename not in document_metadata:
                document_metadata[filename] = {
                    "upload_date": datetime.fromtimestamp(os.path.getctime(file_path)),
                    "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)),
                    "size": size
                }
            
            documents.append(
                DocumentInfo(
                    filename=filename,
                    size=size,
                    active=is_active,
                    last_modified=document_metadata[filename]["last_modified"],
                    upload_date=document_metadata[filename]["upload_date"]
                )
            )
    
    return sorted(documents, key=lambda x: x.last_modified, reverse=True)

# Get conversation history with improved error handling
@app.get("/history")
async def get_history():
    """Return the current state of the conversation memory"""
    try:
        chain_dict = vars(rag_chain)
        memory_dict = {}
        history = []
        summary = ""
        
        # Enhanced debug information
        chain_attributes = []
        memory_attributes = []
        
        if hasattr(rag_chain, 'memory') and rag_chain.memory is not None:
            memory = rag_chain.memory
            memory_dict = vars(memory)
            memory_attributes = list(memory_dict.keys())
            
            logger.info(f"Memory attributes: {memory_attributes}")
            
            # Get chat history
            if hasattr(memory, 'chat_memory') and memory.chat_memory is not None:
                if hasattr(memory.chat_memory, 'messages'):
                    raw_messages = memory.chat_memory.messages
                    for msg in raw_messages:
                        history.append({
                            "role": getattr(msg, "type", str(type(msg))),
                            "content": getattr(msg, "content", str(msg)),
                            "timestamp": datetime.now().isoformat()
                        })
            
            # Get summary
            if hasattr(memory, 'buffer'):
                summary = memory.buffer
            elif hasattr(memory, 'moving_summary_buffer'):  # Fallback for backward compatibility
                summary = memory.moving_summary_buffer
        
        # Get detailed chain attributes
        if chain_dict:
            chain_attributes = list(chain_dict.keys())
            logger.info(f"Chain attributes: {chain_attributes}")
        
        # Create a more detailed response
        return {
            "history": history,
            "summary": summary,
            "memory_attributes": memory_attributes,
            "chain_attributes": chain_attributes,
            "active_document": active_document,
            "timestamp": datetime.now().isoformat(),
            "debug": {
                "chain_type": type(rag_chain).__name__,
                "memory_type": type(rag_chain.memory).__name__ if hasattr(rag_chain, 'memory') else None,
                "has_retriever": hasattr(rag_chain, 'retriever'),
                "has_memory": hasattr(rag_chain, 'memory'),
                "history_count": len(history)
            }
        }
    except Exception as e:
        logger.error(f"Error in get_history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conversation history: {str(e)}"
        )

# Main Q&A route with improved error handling
@app.post("/Chat")
async def ask_question(query: Query):
    try:
        log_memory_state("Before processing")
        logger.info(f"Invoking RAG chain with question: {query.input}")
        result = invoke_with_retry(rag_chain, {"question": query.input})
        logger.info("Checking if memory summary was updated")
        log_memory_state("After processing")

        if result is None:
            raise HTTPException(status_code=503, detail="Service temporarily unavailable due to rate limiting")

        def extract_json_from_text(text: str):
            """Robustly extract a JSON object/array from a model string.
            - strip common code fences
            - find first brace/bracket and match to the closing one
            - try a few heuristics (remove trailing commas) if parse fails
            """
            if not isinstance(text, str):
                raise ValueError("LLM returned non-string answer")

            s = text.strip()
            # remove leading/trailing code fences like ```json or ```
            s = re.sub(r"^```(?:json|\w+)?\s*", "", s, flags=re.I)
            s = re.sub(r"\s*```$", "", s)

            # try direct load first
            try:
                return json.loads(s)
            except Exception:
                pass

            # find first JSON boundary - look for { or [
            start = None
            for i, ch in enumerate(s):
                if ch in '{[':
                    start = i
                    break
            if start is None:
                raise ValueError('No JSON object found in LLM output')

            # find matching bracket using stack
            stack = []
            end = None
            pairs = {'{': '}', '[': ']'}
            open_ch = s[start]
            close_ch = pairs[open_ch]
            for i in range(start, len(s)):
                ch = s[i]
                if ch == open_ch:
                    stack.append(ch)
                elif ch == close_ch:
                    stack.pop()
                    if not stack:
                        end = i
                        break

            if end is None:
                # fallback: try to find last closing brace
                last = s.rfind(close_ch)
                if last == -1:
                    raise ValueError('Could not find end of JSON in LLM output')
                end = last

            candidate = s[start:end+1]

            # try parsing candidate; if fails, attempt simple fixes
            def escape_newlines_in_strings(text: str) -> str:
                # Replace raw newline/carriage returns inside JSON string literals with escaped sequences
                pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"', re.DOTALL)
                def _repl(m):
                    inner = m.group(1)
                    inner_escaped = inner.replace('\r', '\\r').replace('\n', '\\n')
                    return f'"{inner_escaped}"'
                return pattern.sub(_repl, text)

            try:
                return json.loads(candidate)
            except Exception:
                # try escaping newlines inside string values
                try:
                    candidate_esc = escape_newlines_in_strings(candidate)
                    return json.loads(candidate_esc)
                except Exception:
                    # remove trailing commas before } or ] and escape newlines there too
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    fixed = escape_newlines_in_strings(fixed)
                    try:
                        return json.loads(fixed)
                    except Exception as e:
                        # as a last resort, attempt to extract a top-level object with regex
                        m = re.search(r"(\{[\s\S]*\})", s)
                        if m:
                            try:
                                cand2 = escape_newlines_in_strings(m.group(1))
                                return json.loads(cand2)
                            except Exception:
                                pass
                        raise e

        try:
            raw_answer = result.get("answer") if isinstance(result, dict) else result
            parsed = None
            try:
                parsed = extract_json_from_text(raw_answer)
            except Exception as pe:
                # If the LLM returned a quoted JSON string (stringified JSON), attempt to unquote and parse
                try:
                    # strip outer quotes
                    if isinstance(raw_answer, str) and raw_answer.startswith('"') and raw_answer.endswith('"'):
                        inner = raw_answer[1:-1]
                        parsed = extract_json_from_text(inner)
                except Exception:
                    pass

            if parsed is None:
                raise ValueError('Unable to parse LLM output as JSON')

            # If parsed is still a string containing JSON, parse again
            if isinstance(parsed, str):
                try:
                    parsed = json.loads(parsed)
                except Exception:
                    # leave as-is
                    pass

            if not isinstance(parsed, dict) or not all(k in parsed for k in ["answer", "sources"]):
                raise ValueError("Parsed JSON did not contain required keys 'answer' and 'sources'")
        except Exception as e:
            logger.error(f"Failed to parse LLM JSON answer. Raw output was: {raw_answer}", exc_info=True)
            raise HTTPException(status_code=500, detail="Invalid JSON format returned from LLM")

        return {
            "answer": parsed["answer"],
            "sources": parsed["sources"],
            "timestamp": datetime.now().isoformat(),
            "document": active_document,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

def log_memory_state(stage: str):
    if hasattr(rag_chain, 'memory') and rag_chain.memory is not None:
        memory = rag_chain.memory
        logger.info(f"{stage} - Memory attributes: {list(vars(memory).keys())}")
        if hasattr(memory, 'buffer'):
            summary = memory.buffer
            logger.info(f"{stage} - Summary buffer exists: {bool(summary)}")
            logger.info(f"{stage} - Summary type: {type(summary)}")
            logger.info(f"{stage} - Summary length: {len(summary) if summary else 0}")
            logger.info(f"{stage} - Summary content: '{summary}'")
        elif hasattr(memory, 'moving_summary_buffer'):
            summary = memory.moving_summary_buffer
            logger.info(f"{stage} - Legacy summary buffer exists: {bool(summary)}")
            logger.info(f"{stage} - Legacy summary type: {type(summary)}")
            logger.info(f"{stage} - Legacy summary length: {len(summary) if summary else 0}")
            logger.info(f"{stage} - Legacy summary content: '{summary}'")
        if hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
            messages = memory.chat_memory.messages
            logger.info(f"{stage} - Message count: {len(messages)}")
            if messages:
                logger.info(f"{stage} - Last message: {messages[-1]}")

# Activate document with improved error handling
@app.post("/activate-document/{filename}")
async def activate_document(filename: str):
    """Set a document as the active one for the RAG chain"""
    global rag_chain, active_document
    
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document '{filename}' not found"
        )
    
    try:
        active_document = str(file_path)
        rag_chain = get_rag_chain(document_path=active_document)
        
        # Reset conversation memory
        if hasattr(rag_chain, 'memory') and rag_chain.memory is not None:
            if hasattr(rag_chain.memory, 'chat_memory'):
                rag_chain.memory.chat_memory.clear()
            if hasattr(rag_chain.memory, 'buffer'):
                rag_chain.memory.buffer = ""
            elif hasattr(rag_chain.memory, 'moving_summary_buffer'):  # Backward compatibility
                rag_chain.memory.moving_summary_buffer = ""
        
        # Update metadata
        if filename in document_metadata:
            document_metadata[filename]["last_modified"] = datetime.now()
        
        logger.info(f"Activated document: {filename} and reset conversation memory")
        return {
            "status": "success",
            "message": f"Activated document: {filename} (memory reset)",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to activate document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to activate document: {str(e)}"
        )

""" Using these can clear the retrieval cache, which may be useful after re-embedding a document."""
# @app.post("/clear-retrieval-cache")
# async def clear_retrieval_cache(doc_id: Optional[int] = None):
#     """Clear retrieval cache entirely or for a specific document id."""
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         c = conn.cursor()
#         if doc_id is None:
#             c.execute("DELETE FROM retrieval_cache")
#             logger.info('Cleared entire retrieval cache')
#         else:
#             c.execute("DELETE FROM retrieval_cache WHERE doc_id=?", (doc_id,))
#             logger.info(f'Cleared retrieval cache for doc_id={doc_id}')
#         conn.commit()
#         conn.close()
#         return {"status": "success", "doc_id": doc_id}
#     except Exception as e:
#         logger.error(f"Failed to clear retrieval cache: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/cache-status")
# async def cache_status(limit: int = 10):
#     """Return simple stats and a small sample of retrieval_cache rows for debugging."""
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         c = conn.cursor()
#         count = c.execute("SELECT count(*) FROM retrieval_cache").fetchone()[0]
#         sample = c.execute("SELECT qhash, doc_id, k, length(result), created_at FROM retrieval_cache ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
#         conn.close()
#         return {"count": count, "sample": sample}
#     except Exception as e:
#         logger.error(f"Failed to read retrieval cache status: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# Reset memory endpoint
@app.post("/reset-chat-memory")
async def reset_chat_memory():
    """Reset the conversation memory"""
    try:
        if hasattr(rag_chain, 'memory') and rag_chain.memory is not None:
            if hasattr(rag_chain.memory, 'chat_memory'):
                rag_chain.memory.chat_memory.clear()
            if hasattr(rag_chain.memory, 'buffer'):
                rag_chain.memory.buffer = ""
            elif hasattr(rag_chain.memory, 'moving_summary_buffer'):  # Backward compatibility
                rag_chain.memory.moving_summary_buffer = ""

         

            logger.info("Conversation memory reset successfully")
            return {
                "status": "success",
                "message": "Conversation memory reset",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="No memory to reset"
            )
    except Exception as e:
        logger.error(f"Failed to reset memory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset memory: {str(e)}"
        )

# Reset document memory endpoint
@app.post("/reset-document-memory")
async def reset_document_memory():
    """Reset the document memory, status, and uploaded files"""
    global rag_chain
    try:
        # Reset document metadata
        if hasattr(rag_chain, 'doc_status') and rag_chain.doc_status is not None:
            rag_chain.doc_status.clear()

        # Clear uploaded files 
        if os.path.exists(UPLOAD_DIR):
            for file_name in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, file_name)
                if file_name.lower().endswith(".pdf") and os.path.isfile(file_path):
                    os.remove(file_path)

        else:
            raise HTTPException(
                status_code=400,
                detail="No memory to reset"
            )
            logger.error("No document memory to reset") 

        return {
            "status": "success",
            "message": "Document memory, status, and uploaded files reset",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to reset document memory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset document memory: {str(e)}"
        )

# Force summary generation endpoint
@app.post("/generate-summary")
async def generate_summary():
    """Force generation of conversation summary"""
    try:
        if hasattr(rag_chain, 'memory') and rag_chain.memory is not None:
            memory = rag_chain.memory
            
            if hasattr(memory, 'chat_memory') and memory.chat_memory is not None and hasattr(memory.chat_memory, 'messages'):
                messages = memory.chat_memory.messages
                
                if len(messages) < 2:
                    return {
                        "status": "warning",
                        "message": "Not enough messages to generate a summary (need at least one human and one AI message)",
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Try to manually generate summary
                if hasattr(memory, 'predict_new_summary'):
                    old_summary = ""
                    
                    # Get existing summary
                    if hasattr(memory, 'buffer'):
                        old_summary = memory.buffer or ""
                    elif hasattr(memory, 'moving_summary_buffer'):
                        old_summary = memory.moving_summary_buffer or ""
                    
                    # Generate new summary
                    new_summary = memory.predict_new_summary(
                        old_summary, 
                        messages
                    )
                    
                    # Store the new summary
                    if hasattr(memory, 'buffer'):
                        memory.buffer = new_summary
                    elif hasattr(memory, 'moving_summary_buffer'):
                        memory.moving_summary_buffer = new_summary
                    
                    logger.info(f"Manually generated summary: {new_summary}")
                    
                    return {
                        "status": "success",
                        "message": "Summary generated successfully",
                        "summary": new_summary,
                        "old_summary": old_summary,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "status": "error",
                        "message": "Memory doesn't have predict_new_summary method",
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                return {
                    "status": "error",
                    "message": "No chat messages found in memory",
                    "timestamp": datetime.now().isoformat()
                }
        else:
            return {
                "status": "error",
                "message": "No memory available",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating summary: {str(e)}"
        )