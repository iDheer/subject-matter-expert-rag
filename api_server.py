# api_server.py
import os
import json
import asyncio
import re
from datetime import datetime
from typing import List, Dict, Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from elasticsearch import Elasticsearch
from llama_index.core import Settings, StorageContext
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex

# Import ChatMemoryManager from your existing file
import sys
sys.path.append('.')

# Copy the ChatMemoryManager class from SME_2_query_elasticsearch_system.py
class ChatMemoryManager:
    """Manages conversational memory using Qwen for summarization and context management"""
    
    def __init__(self, llm, max_context_length: int = 4000, max_history_pairs: int = 10):
        self.llm = llm
        self.max_context_length = max_context_length
        self.max_history_pairs = max_history_pairs
        self.conversation_history: List[Dict] = []
        self.conversation_summary = ""
        self.enabled = True  # Can be toggled for memory/no-memory mode
        
    def add_exchange(self, user_query: str, system_response: str, sources: List[str] = None):
        """Add a user query and system response to the conversation history"""
        if not self.enabled:
            return
            
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "system_response": system_response,
            "sources": sources or []
        }
        self.conversation_history.append(exchange)
        
        # Keep only the last N exchanges
        if len(self.conversation_history) > self.max_history_pairs:
            self.conversation_history = self.conversation_history[-self.max_history_pairs:]
        
        # Update summary if we have too much content
        self._update_summary_if_needed()
    
    def _update_summary_if_needed(self):
        """Update the conversation summary using Qwen if context is getting too long"""
        if not self.enabled:
            return
            
        total_length = self._calculate_total_length()
        
        if total_length > self.max_context_length and len(self.conversation_history) > 3:
            # Summarize the older part of the conversation
            older_exchanges = self.conversation_history[:-2]  # Keep last 2 exchanges
            summary_prompt = self._create_summary_prompt(older_exchanges)
            
            try:
                print("🧠 Updating conversation summary...")
                summary_response = self.llm.complete(summary_prompt)
                self.conversation_summary = summary_response.text.strip()
                
                # Keep only the last 2 exchanges
                self.conversation_history = self.conversation_history[-2:]
                
                print("✅ Conversation summary updated")
            except Exception as e:
                print(f"⚠️ Error updating summary: {e}")
    
    def _calculate_total_length(self) -> int:
        """Calculate total character length of conversation history"""
        total = len(self.conversation_summary)
        for exchange in self.conversation_history:
            total += len(exchange["user_query"]) + len(exchange["system_response"])
        return total
    
    def _create_summary_prompt(self, exchanges: List[Dict]) -> str:
        """Create a prompt for Qwen to summarize conversation history"""
        conversation_text = ""
        for exchange in exchanges:
            conversation_text += f"User: {exchange['user_query']}\n"
            conversation_text += f"Assistant: {exchange['system_response'][:500]}...\n\n"
        
        return f"""Please summarize the following conversation between a user and an AI assistant that helps with document retrieval and analysis. Focus on the main topics, key insights, and context that would be useful for continuing the conversation:

{conversation_text}

Provide a concise summary (2-3 paragraphs maximum) that captures the essence of the discussion:"""
    
    def get_contextual_prompt(self, current_query: str) -> str:
        """Generate a contextual prompt that includes conversation history"""
        if not self.enabled:
            return current_query
            
        context_parts = []
        
        # Add conversation summary if available
        if self.conversation_summary:
            context_parts.append(f"Previous conversation summary:\n{self.conversation_summary}\n")
        
        # Add recent exchanges
        if self.conversation_history:
            context_parts.append("Recent conversation:")
            for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
                context_parts.append(f"User: {exchange['user_query']}")
                # Truncate long responses for context
                response_preview = exchange['system_response'][:300]
                if len(exchange['system_response']) > 300:
                    response_preview += "..."
                context_parts.append(f"Assistant: {response_preview}\n")
        
        # Add current query with context instruction
        if context_parts:
            context_text = "\n".join(context_parts)
            return f"""Given our conversation history:

{context_text}

Current question: {current_query}

Please provide a response that takes into account our previous discussion. If the current question relates to previous topics, reference them appropriately. If it's a new topic, you can start fresh while maintaining the conversational tone."""
        else:
            return current_query
    
    def toggle_memory(self):
        """Toggle memory on/off"""
        self.enabled = not self.enabled
        status = "enabled" if self.enabled else "disabled"
        print(f"🧠 Conversation memory {status}")
        return self.enabled
    
    def clear_memory(self):
        """Clear all conversation history and summary"""
        self.conversation_history = []
        self.conversation_summary = ""
        print("🧹 Conversation memory cleared")
    
    def get_memory_status(self):
        """Get current memory status"""
        if not self.enabled:
            return "Memory disabled"
        
        status = f"Memory enabled - {len(self.conversation_history)} exchanges"
        if self.conversation_summary:
            status += " + summary"
        return status

# --- Configuration ---
ES_ENDPOINT = "http://localhost:9200"
INDEX_NAME = "advanced_docs_elasticsearch_v2"
ES_STORAGE_DIR = "./elasticsearch_storage_v2"

# Global variables to store the initialized components
query_engine = None
memory_manager = None
index = None
storage_context = None

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    use_memory: bool = True
    stream: bool = False

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    memory_status: str

class MemoryToggleResponse(BaseModel):
    memory_enabled: bool
    message: str

class StatusResponse(BaseModel):
    status: str
    memory_status: str
    elasticsearch_connected: bool
    index_exists: bool
    docstore_nodes: int

# --- Startup/Shutdown Logic ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the SME system on startup"""
    global query_engine, memory_manager, index, storage_context
    
    print("🚀 Initializing SME API Server...")
    
    try:
        # Configure models
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        if device == "cuda":
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            os.environ['OLLAMA_NUM_GPU'] = '1'
            os.environ['OLLAMA_GPU_LAYERS'] = '35'
            
        Settings.llm = Ollama(
            model="qwen3:4b",
            request_timeout=300.0,
            base_url="http://localhost:11434",
        )
        
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2",
            device=device
        )
        
        # Initialize memory manager
        memory_manager = ChatMemoryManager(Settings.llm)
        print("✅ Memory manager initialized")
        
        # Connect to Elasticsearch
        es_client = Elasticsearch([ES_ENDPOINT])
        es_info = es_client.info()
        print(f"✅ Connected to Elasticsearch {es_info.body['version']['number']}")
        
        # Check index exists
        if not es_client.indices.exists(index=INDEX_NAME):
            raise Exception(f"Index '{INDEX_NAME}' does not exist! Run SME_1_build_elasticsearch_database.py first.")
        
        # Create vector store
        vector_store = ElasticsearchStore(
            index_name=INDEX_NAME,
            es_url=ES_ENDPOINT,
            vector_field="embedding",
            text_field="content"
        )
        
        # Load storage context
        if not os.path.exists(ES_STORAGE_DIR):
            raise Exception(f"Storage directory not found: {ES_STORAGE_DIR}")
        
        storage_context = StorageContext.from_defaults(
            persist_dir=ES_STORAGE_DIR,
            vector_store=vector_store
        )
        print(f"✅ Loaded storage context with {len(storage_context.docstore.docs)} nodes")
        
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        
        # Configure retriever and query engine
        try:
            base_retriever = index.as_retriever(similarity_top_k=12)
            retriever = AutoMergingRetriever(
                base_retriever,
                storage_context,
                verbose=False  # Disable verbose for API
            )
            
            reranker = SentenceTransformerRerank(
                top_n=4,
                model="BAAI/bge-reranker-v2-m3",
                device=device
            )
            
            query_engine = RetrieverQueryEngine.from_args(
                retriever,
                node_postprocessors=[reranker],
                streaming=True,
            )
            
            print("✅ AutoMergingRetriever configured successfully")
            
        except Exception as e:
            print(f"⚠️ AutoMergingRetriever failed, using fallback: {e}")
            base_retriever = index.as_retriever(similarity_top_k=10)
            reranker = SentenceTransformerRerank(top_n=5, model="BAAI/bge-reranker-v2-m3", device=device)
            query_engine = RetrieverQueryEngine.from_args(base_retriever, node_postprocessors=[reranker], streaming=True)
        
        print("🎉 SME API Server ready!")
        
    except Exception as e:
        print(f"❌ Failed to initialize SME system: {e}")
        raise e
    
    yield
    
    # Cleanup on shutdown
    print("🛑 Shutting down SME API Server...")

# --- FastAPI App ---
app = FastAPI(
    title="SME (Subject Matter Expert) API",
    description="API for querying the SME system with conversational memory",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def process_qwen3_response(response_text: str) -> str:
    """Remove <think> tags from Qwen3 response"""
    # Remove everything between <think> and </think>
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    return cleaned.strip()

def extract_sources(response) -> List[Dict]:
    """Extract source information from query response"""
    sources = []
    for i, node in enumerate(response.source_nodes):
        cleaned_text = ' '.join(node.get_content().split())
        file_name = node.metadata.get('file_name', 'N/A')
        
        # Convert numpy.float32 to Python float for JSON serialization
        score = float(node.score) if hasattr(node, 'score') and node.score is not None else 0.0
        
        sources.append({
            "index": i + 1,
            "score": round(score, 4),
            "file_name": str(file_name),  # Ensure string
            "node_type": "Merged" if len(cleaned_text) > 1000 else "Leaf",
            "content_snippet": cleaned_text[:250] + "..." if len(cleaned_text) > 250 else cleaned_text,
            "node_id": str(node.node_id)  # Ensure string
        })
    
    return sources

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "SME API Server is running", "status": "healthy"}

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get the current status of the SME system"""
    global memory_manager, index, storage_context
    
    try:
        # Check if components are initialized
        es_connected = False
        index_exists = False
        docstore_nodes = 0
        
        # Check Elasticsearch connection
        try:
            es_client = Elasticsearch([ES_ENDPOINT])
            es_info = es_client.info()
            es_connected = True
            # Convert HeadApiResponse to boolean
            index_exists = bool(es_client.indices.exists(index=INDEX_NAME))
        except:
            pass
        
        # Check docstore
        if storage_context and hasattr(storage_context, 'docstore'):
            try:
                docstore_nodes = len(storage_context.docstore.docs)
            except:
                pass
        
        # Determine overall status
        if index is not None and memory_manager is not None and storage_context is not None:
            system_status = "ready"
        elif any([index, memory_manager, storage_context]):
            system_status = "initializing"
        else:
            system_status = "not_initialized"
        
        return StatusResponse(
            status=system_status,
            memory_status=memory_manager.get_memory_status() if memory_manager else "not_initialized",
            elasticsearch_connected=es_connected,
            index_exists=index_exists,
            docstore_nodes=docstore_nodes
        )
        
    except Exception as e:
        print(f"Status check error: {e}")
        return StatusResponse(
            status="error",
            memory_status="error",
            elasticsearch_connected=False,
            index_exists=False,
            docstore_nodes=0
        )

@app.post("/query", response_model=QueryResponse)
async def query_sme(request: QueryRequest):
    """Query the SME system with optional conversation memory"""
    global query_engine, memory_manager
    
    if not query_engine or not memory_manager:
        raise HTTPException(status_code=503, detail="SME system not initialized")
    
    try:
        # Set memory state
        if not request.use_memory and memory_manager.enabled:
            memory_manager.enabled = False
        elif request.use_memory and not memory_manager.enabled:
            memory_manager.enabled = True
        
        # Generate contextual prompt if memory is enabled
        if memory_manager.enabled:
            contextual_query = memory_manager.get_contextual_prompt(request.question)
        else:
            contextual_query = request.question
        
        # Query the system
        response = query_engine.query(contextual_query)
        
        # Process the response
        full_response = ""
        for text in response.response_gen:
            full_response += text
        
        # Clean the response
        cleaned_response = process_qwen3_response(full_response)
        
        # Extract sources
        sources = extract_sources(response)
        
        # Add to memory if enabled
        source_files = [s["file_name"] for s in sources]
        if memory_manager.enabled:
            memory_manager.add_exchange(request.question, cleaned_response, source_files)
        
        return QueryResponse(
            answer=cleaned_response,
            sources=sources,
            memory_status=memory_manager.get_memory_status()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/memory/toggle", response_model=MemoryToggleResponse)
async def toggle_memory():
    """Toggle conversation memory on/off"""
    global memory_manager
    
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    
    enabled = memory_manager.toggle_memory()
    status = "enabled" if enabled else "disabled"
    
    return MemoryToggleResponse(
        memory_enabled=enabled,
        message=f"Conversation memory {status}"
    )

@app.post("/memory/clear")
async def clear_memory():
    """Clear all conversation history and summary"""
    global memory_manager
    
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    
    memory_manager.clear_memory()
    return {"message": "Conversation memory cleared", "memory_status": memory_manager.get_memory_status()}

@app.get("/memory/status")
async def get_memory_status():
    """Get current memory status"""
    global memory_manager
    
    if not memory_manager:
        raise HTTPException(status_code=503, detail="Memory manager not initialized")
    
    return {
        "memory_status": memory_manager.get_memory_status(),
        "enabled": memory_manager.enabled,
        "conversation_count": len(memory_manager.conversation_history),
        "has_summary": bool(memory_manager.conversation_summary)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")