# 2_query_system_elasticsearch_hierarchy.py
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
from llama_index.core import Settings, StorageContext
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import VectorStoreIndex

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
        
        return f"""<think>
I need to create a concise summary of this conversation that captures:
1. Main topics discussed
2. Key information retrieved
3. User's areas of interest
4. Important context for future queries
</think>

Please summarize the following conversation between a user and an AI assistant that helps with document retrieval and analysis. Focus on the main topics, key insights, and context that would be useful for continuing the conversation:

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

# --- 0. Define Constants ---
ES_ENDPOINT = "http://localhost:9200"
INDEX_NAME = "advanced_docs_elasticsearch_v2"
ES_STORAGE_DIR = "./elasticsearch_storage_v2"

# --- 1. CONFIGURE MODELS ---
print("--- Configuring models with conversational memory support ---")

# Auto-detect device (use CPU if CUDA not available)
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {device} ---")

# Configure Ollama with GPU settings
if device == "cuda":
    # For GPU usage, we need to ensure Ollama uses GPU memory
    # Set environment variables to force GPU usage
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
    os.environ['OLLAMA_NUM_GPU'] = '1'       # Number of GPUs to use
    os.environ['OLLAMA_GPU_LAYERS'] = '35'   # Number of layers to offload to GPU
    
    Settings.llm = Ollama(
        model="qwen3:4b",
        request_timeout=300.0,
        base_url="http://localhost:11434",
    )
    print("🟢 Configured Ollama to use GPU memory")
else:
    Settings.llm = Ollama(model="qwen3:4b", request_timeout=300.0)
    print("🟡 Using CPU (CUDA not available)")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2",
    device=device
)

# Initialize chat memory manager
memory_manager = ChatMemoryManager(Settings.llm)
print("🧠 Initialized conversation memory manager")

# --- 2. LOAD WITH PROPER HIERARCHY SUPPORT ---
print(f"--- Loading data from Elasticsearch and local storage ---")

try:
    # Connect to Elasticsearch
    es_client = Elasticsearch([ES_ENDPOINT])
    es_info = es_client.info()
    print(f"✅ Connected to Elasticsearch {es_info.body['version']['number']}")
    
    # Check if index exists
    if not es_client.indices.exists(index=INDEX_NAME):
        print(f"❌ Index '{INDEX_NAME}' does not exist!")
        print("Please run '1_build_database_elasticsearch_fixed.py' first.")
        exit()
    
    # Create Elasticsearch vector store
    vector_store = ElasticsearchStore(
        index_name=INDEX_NAME,
        es_url=ES_ENDPOINT,
        vector_field="embedding",
        text_field="content"
    )
    
    # Load storage context with hierarchy
    if not os.path.exists(ES_STORAGE_DIR):
        print(f"❌ Storage directory not found at '{ES_STORAGE_DIR}'.")
        print("Please run '1_build_database_elasticsearch_fixed.py' first.")
        exit()
    
    storage_context = StorageContext.from_defaults(
        persist_dir=ES_STORAGE_DIR,
        vector_store=vector_store
    )
    print(f"✅ Loaded storage context with {len(storage_context.docstore.docs)} nodes")
    
    # Create index from storage context (maintains hierarchy)
    print("--- Creating index with hierarchy support ---")
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )

except Exception as e:
    print(f"❌ Error loading index: {e}")
    exit()

# --- 3. CONFIGURE ADVANCED RETRIEVER WITH HIERARCHY ---
print("--- Configuring AutoMergingRetriever with hierarchy ---")

try:
    # Base retriever
    base_retriever = index.as_retriever(similarity_top_k=12)
    
    # AutoMerging retriever (this needs the hierarchy)
    retriever = AutoMergingRetriever(
        base_retriever,
        storage_context,  # Pass storage_context directly
        verbose=True
    )
    
    # Reranker
    reranker = SentenceTransformerRerank(
        top_n=4,
        model="BAAI/bge-reranker-v2-m3",
        device=device
    )
    
    # Query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=[reranker],
        streaming=True,
    )
    
    print("✅ AutoMergingRetriever configured successfully")

except Exception as e:
    print(f"❌ Error configuring AutoMergingRetriever: {e}")
    print("Falling back to simple retriever...")
    
    # Fallback to simple retriever
    base_retriever = index.as_retriever(similarity_top_k=10)
    reranker = SentenceTransformerRerank(top_n=5, model="BAAI/bge-reranker-v2-m3", device=device)
    query_engine = RetrieverQueryEngine.from_args(base_retriever, node_postprocessors=[reranker], streaming=True)

def print_qwen3_response_stream(response_stream):
    """Custom function to handle Qwen3's thinking tags while streaming"""
    full_text = ""
    in_think_tag = False
    buffer = ""
    
    print("🤖 Response: ", end="", flush=True)
    
    for text in response_stream.response_gen:
        full_text += text
        buffer += text
        
        while buffer:
            if "<think>" in buffer and not in_think_tag:
                think_start = buffer.find("<think>")
                if think_start > 0:
                    print(buffer[:think_start], end="", flush=True)
                
                in_think_tag = True
                print("\n\n🤔 Thinking: ", end="", flush=True)
                buffer = buffer[think_start + 7:]
                continue
            
            elif "</think>" in buffer and in_think_tag:
                think_end = buffer.find("</think>")
                if think_end > 0:
                    print(buffer[:think_end], end="", flush=True)
                
                in_think_tag = False
                print("\n\n💭 Final Answer: ", end="", flush=True)
                buffer = buffer[think_end + 8:]
                continue
            
            else:
                if not ("<think>" in buffer or "</think>" in buffer):
                    print(buffer, end="", flush=True)
                    buffer = ""
                else:
                    break
    
    if buffer:
        print(buffer, end="", flush=True)
    
    print()
    return full_text

def show_commands():
    """Display available commands"""
    print("\n📋 Available Commands:")
    print("  • Type your question normally for contextual responses")
    print("  • '/memory' - Toggle conversation memory on/off")
    print("  • '/clear' - Clear conversation memory")
    print("  • '/status' - Show memory status")
    print("  • '/help' - Show this help message")
    print("  • 'exit' - Exit the system")

# --- 4. ASK QUESTIONS WITH MEMORY ---
print("\n🚀 Ready to Query with Elasticsearch + AutoMerging + Conversation Memory!")
print("🧠 This SME system now remembers your conversation context!")
show_commands()

try:
    while True:
        question = input("\n💬 Question: ")
        
        # Handle commands
        if question.lower() == 'exit':
            break
        elif question.lower() in ['/help', 'help']:
            show_commands()
            continue
        elif question.lower() == '/memory':
            memory_manager.toggle_memory()
            continue
        elif question.lower() == '/clear':
            memory_manager.clear_memory()
            continue
        elif question.lower() == '/status':
            print(f"📊 {memory_manager.get_memory_status()}")
            continue

        # Generate contextual prompt if memory is enabled
        if memory_manager.enabled:
            contextual_query = memory_manager.get_contextual_prompt(question)
            print(f"\n🔍 Searching with conversation context: '{question}'")
        else:
            contextual_query = question
            print(f"\n🔍 Searching (standalone): '{question}'")
        
        try:
            # Query with contextual prompt if memory enabled, otherwise standalone
            response = query_engine.query(contextual_query)
            full_response = print_qwen3_response_stream(response)

            # Extract source information
            source_info = []
            print("\n--- Source Nodes (Post-AutoMerging & Re-ranking) ---")
            for i, node in enumerate(response.source_nodes):
                print(f"Source {i+1} (Score: {node.score:.4f}):")
                cleaned_text = ' '.join(node.get_content().split())
                file_name = node.metadata.get('file_name', 'N/A')
                print(f"  -> File: {file_name}")
                print(f"  -> Node Type: {'Merged' if len(cleaned_text) > 1000 else 'Leaf'}")
                print(f"  -> Content Snippet: \"{cleaned_text[:250]}...\"\n")
                source_info.append(file_name)
            print("--------------------------------------")
            
            # Add to conversation memory if enabled
            if memory_manager.enabled:
                memory_manager.add_exchange(question, full_response, source_info)
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print("Continuing with next question...")

except KeyboardInterrupt:
    print("\n👋 Exiting gracefully. Goodbye!")