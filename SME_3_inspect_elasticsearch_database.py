# 3_inspect_elasticsearch_fixed.py
import os
from elasticsearch import Elasticsearch
from llama_index.core import StorageContext

# --- 0. Define Constants (FIXED TO MATCH BUILDER) ---
ES_ENDPOINT = "http://localhost:9200"
INDEX_NAME = "advanced_docs_elasticsearch_v2"  # Fixed to match builder
ES_STORAGE_DIR = "./elasticsearch_storage_v2"   # Fixed to match builder

def inspect_elasticsearch_index():
    """Inspect the Elasticsearch index"""
    print("=== Inspecting Elasticsearch Index ===")
    
    try:
        es_client = Elasticsearch([ES_ENDPOINT])
        es_info = es_client.info()
        print(f"Connected to Elasticsearch {es_info.body['version']['number']}")
        
        if not es_client.indices.exists(index=INDEX_NAME):
            print(f"❌ Index '{INDEX_NAME}' does not exist!")
            return
        
        # Get index stats
        stats = es_client.indices.stats(index=INDEX_NAME)
        doc_count = stats['indices'][INDEX_NAME]['total']['docs']['count']
        store_size = stats['indices'][INDEX_NAME]['total']['store']['size_in_bytes']
        
        print(f"\n📊 Index Statistics:")
        print(f"  Index name: {INDEX_NAME}")
        print(f"  Document count: {doc_count}")
        print(f"  Store size: {store_size / (1024*1024):.2f} MB")
        
        # Get index mapping
        mapping = es_client.indices.get_mapping(index=INDEX_NAME)
        properties = mapping[INDEX_NAME]['mappings']['properties']
        
        print(f"\n🔍 Fields in index:")
        for field_name, field_info in properties.items():
            field_type = field_info.get('type', 'object')
            if field_type == 'dense_vector':
                dims = field_info.get('dims', 'unknown')
                print(f"  - {field_name}: {field_type} (dims: {dims})")
            else:
                print(f"  - {field_name}: {field_type}")
        
        # Sample some documents
        print(f"\n📄 Sample documents:")
        search_result = es_client.search(index=INDEX_NAME, size=2)
        
        for i, hit in enumerate(search_result['hits']['hits']):
            source = hit['_source']
            content_preview = source.get('content', '')[:150] + "..."
            print(f"\n  Document {i+1}:")
            print(f"    Content: {content_preview}")
            if 'metadata' in source:
                print(f"    File: {source['metadata'].get('file_name', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error inspecting Elasticsearch: {e}")

def inspect_local_storage():
    """Inspect the local storage (docstore)"""
    print("\n=== Inspecting Local Storage ===")
    
    if not os.path.exists(ES_STORAGE_DIR):
        print(f"❌ Local storage directory not found: {ES_STORAGE_DIR}")
        return
    
    try:
        storage_context = StorageContext.from_defaults(persist_dir=ES_STORAGE_DIR)
        docstore = storage_context.docstore
        
        all_nodes = list(docstore.docs.values())
        print(f"📁 Docstore contains {len(all_nodes)} nodes")
        
        # Count node types
        parent_nodes = [n for n in all_nodes if hasattr(n, 'child_nodes') and n.child_nodes]
        leaf_nodes = [n for n in all_nodes if not (hasattr(n, 'child_nodes') and n.child_nodes)]
        
        print(f"  - Parent nodes: {len(parent_nodes)}")
        print(f"  - Leaf nodes: {len(leaf_nodes)}")
        
        # Show hierarchy info
        if parent_nodes:
            total_children = sum(len(n.child_nodes) for n in parent_nodes)
            print(f"  - Average children per parent: {total_children / len(parent_nodes):.1f}")
        
        # Sample node content
        print(f"\n📝 Sample node content:")
        if all_nodes:
            sample_node = all_nodes[0]
            content_preview = sample_node.get_content()[:200] + "..."
            print(f"  Content: {content_preview}")
            print(f"  Node ID: {sample_node.node_id}")
            print(f"  Has children: {hasattr(sample_node, 'child_nodes') and bool(sample_node.child_nodes)}")
        
    except Exception as e:
        print(f"❌ Error inspecting local storage: {e}")

def inspect_hierarchy_integrity():
    """Check integrity of hierarchical relationships"""
    print("\n=== Checking Hierarchy Integrity ===")
    
    try:
        storage_context = StorageContext.from_defaults(persist_dir=ES_STORAGE_DIR)
        docstore = storage_context.docstore
        all_nodes = list(docstore.docs.values())
        
        # Check parent-child relationships
        orphaned_children = 0
        valid_relationships = 0
        
        for node in all_nodes:
            if hasattr(node, 'parent_node') and node.parent_node:
                parent_id = node.parent_node.node_id
                if parent_id in docstore.docs:
                    valid_relationships += 1
                else:
                    orphaned_children += 1
        
        print(f"✅ Valid parent-child relationships: {valid_relationships}")
        if orphaned_children > 0:
            print(f"⚠️  Orphaned children (missing parents): {orphaned_children}")
        else:
            print(f"✅ No orphaned children found")
        
    except Exception as e:
        print(f"❌ Error checking hierarchy: {e}")

if __name__ == "__main__":
    inspect_elasticsearch_index()
    inspect_local_storage()
    inspect_hierarchy_integrity()