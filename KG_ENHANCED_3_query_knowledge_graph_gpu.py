#!/usr/bin/env python3
"""
GPU-Accelerated Enhanced Chapter-Based Knowledge Graph Query System
Advanced querying system for the enhanced chapter-based knowledge graph
"""
import os
import json
import logging
from typing import List, Dict, Optional, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
from elasticsearch import Elasticsearch
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import networkx as nx
from datetime import datetime
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
ES_ENDPOINT = "http://localhost:9200"
INDEX_NAME = "gpu_chapter_knowledge_v1"
ES_STORAGE_DIR = "./gpu_chapter_elasticsearch_storage"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "knowledge123"

class GPUAcceleratedEnhancedChapterQuerySystem:
    """GPU-accelerated query system for enhanced chapter-based knowledge graph"""
    
    def __init__(self):
        self.setup_gpu()
        self.setup_models()
        self.setup_connections()
        self.load_enhanced_graph_data()
    
    def setup_gpu(self):
        """Setup GPU acceleration"""
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.empty_cache()
            logger.info(f"🚀 GPU acceleration enabled - Using {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu"
            logger.warning("⚠️ CUDA not available, using CPU")
    
    def setup_models(self):
        """Setup GPU-accelerated models"""
        logger.info("🔧 Setting up GPU-accelerated models...")
        
        # Setup LLM
        Settings.llm = Ollama(model="qwen3:4b", request_timeout=3000.0)
        
        # Setup GPU-accelerated embedding model for queries
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2",
            device=self.device
        )
        
        # Setup sentence transformer for similarity searches
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2',
            device=self.device
        )
        
        logger.info(f"✅ Models loaded on {self.device}")
    
    def setup_connections(self):
        """Setup database connections"""
        # Elasticsearch
        try:
            self.es_client = Elasticsearch([ES_ENDPOINT])
            logger.info("✅ Connected to Elasticsearch")
        except Exception as e:
            logger.error(f"❌ Elasticsearch connection failed: {e}")
            raise
        
        # Neo4j
        try:
            self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            logger.info("✅ Connected to Neo4j")
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            raise
        
        # LlamaIndex
        try:
            vector_store = ElasticsearchStore(
                index_name=INDEX_NAME,
                es_url=ES_ENDPOINT,
                vector_field="embedding",
                text_field="content"
            )
            
            storage_context = StorageContext.from_defaults(
                persist_dir=ES_STORAGE_DIR,
                vector_store=vector_store
            )
            
            self.vector_index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            
            logger.info("✅ LlamaIndex connected")
            
        except Exception as e:
            logger.error(f"❌ LlamaIndex connection failed: {e}")
            raise
    
    def load_enhanced_graph_data(self):
        """Load enhanced graph data from files"""
        logger.info("📚 Loading enhanced graph data...")
        
        try:
            # Load enhanced chapters
            with open('./gpu_knowledge_graph_data/enhanced_chapters.json', 'r') as f:
                self.chapters = json.load(f)
            
            # Load enhanced concepts
            with open('./gpu_knowledge_graph_data/enhanced_concepts.json', 'r') as f:
                self.concepts = json.load(f)
            
            # Load enhanced stats
            with open('./gpu_knowledge_graph_data/enhanced_stats.json', 'r') as f:
                self.stats = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.chapters)} chapters, {len(self.concepts)} concepts")
            logger.info(f"📊 {self.stats['num_clusters']} concept clusters, avg {self.stats['avg_concepts_per_chapter']:.1f} concepts/chapter")
            
        except FileNotFoundError as e:
            logger.error(f"❌ Enhanced graph data not found: {e}")
            logger.info("💡 Please run 'KG_ENHANCED_2_build_knowledge_graph_gpu.py' first")
            raise
    
    def find_similar_chapters(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find chapters most similar to query using GPU-accelerated embeddings"""
        logger.info(f"🔍 Finding similar chapters for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], device=self.device)
        
        # Get chapter embeddings
        chapter_embeddings = np.array([ch['embedding'] for ch in self.chapters])
        
        # Calculate similarities using GPU if available
        if self.device == "cuda":
            query_tensor = torch.tensor(query_embedding).cuda()
            chapter_tensor = torch.tensor(chapter_embeddings).cuda()
            similarities = torch.nn.functional.cosine_similarity(
                query_tensor, chapter_tensor, dim=1
            ).cpu().numpy()
        else:
            similarities = cosine_similarity(query_embedding, chapter_embeddings)[0]
        
        # Get top-k similar chapters
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            chapter = self.chapters[idx]
            results.append({
                'chapter': chapter,
                'similarity': float(similarities[idx]),
                'concepts': [c for c in self.concepts if c['chapter_id'] == chapter['id']],
                'clusters': chapter.get('concept_clusters', [])
            })
        
        return results
    
    def find_concepts_in_cluster(self, cluster_id: str) -> Dict:
        """Find all concepts in a specific cluster"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
            MATCH (cl:ConceptCluster {id: $cluster_id})-[:CONTAINS_CONCEPT]->(c:LearningConcept)
            MATCH (ch:Chapter)-[:CONTAINS_CLUSTER]->(cl)
            RETURN ch.title as chapter_title, ch.number as chapter_number,
                   cl.name as cluster_name, cl.type as cluster_type,
                   collect({
                       id: c.id,
                       text: c.text,
                       type: c.type,
                       difficulty: c.difficulty,
                       keywords: c.keywords,
                       confidence: c.confidence
                   }) as concepts
            """, cluster_id=cluster_id)
            
            record = result.single()
            if record:
                return {
                    'chapter_title': record['chapter_title'],
                    'chapter_number': record['chapter_number'],
                    'cluster_name': record['cluster_name'],
                    'cluster_type': record['cluster_type'],
                    'concepts': record['concepts'],
                    'num_concepts': len(record['concepts'])
                }
            return None
    
    def find_learning_path_enhanced(self, target_chapter_id: str, user_level: str = 'intermediate') -> Dict:
        """Find enhanced learning path considering concept clusters and user level"""
        logger.info(f"🗺️ Finding enhanced learning path to chapter: {target_chapter_id}")
        
        with self.neo4j_driver.session() as session:
            # Get comprehensive learning path with concept details
            result = session.run("""
            MATCH path = (start:Chapter)-[:PREREQUISITE_FOR*]->(target:Chapter {id: $target_id})
            WHERE NOT (()-[:PREREQUISITE_FOR]->(start))
            WITH path, length(path) as path_length,
                 [node in nodes(path) | {
                     id: node.id,
                     title: node.title,
                     number: node.number,
                     estimated_hours: node.estimated_hours,
                     num_concepts: node.num_concepts,
                     num_clusters: node.num_clusters,
                     difficulty: node.difficulty
                 }] as path_nodes
            RETURN path_nodes, path_length,
                   reduce(total = 0, node in path_nodes | total + node.estimated_hours) as total_hours
            ORDER BY path_length
            LIMIT 5
            """, target_id=target_chapter_id)
            
            paths = []
            for record in result:
                path_data = {
                    'nodes': record['path_nodes'],
                    'length': record['path_length'],
                    'total_hours': record['total_hours']
                }
                
                # Add concept cluster information for each chapter in path
                for node in path_data['nodes']:
                    chapter_clusters = session.run("""
                    MATCH (c:Chapter {id: $chapter_id})-[:CONTAINS_CLUSTER]->(cl:ConceptCluster)
                    RETURN collect({
                        id: cl.id,
                        name: cl.name,
                        type: cl.type,
                        num_concepts: cl.num_concepts
                    }) as clusters
                    """, chapter_id=node['id']).single()
                    
                    node['clusters'] = chapter_clusters['clusters'] if chapter_clusters else []
                
                paths.append(path_data)
            
            # Get target chapter detailed info
            target_info = session.run("""
            MATCH (c:Chapter {id: $target_id})
            OPTIONAL MATCH (c)-[:CONTAINS_CLUSTER]->(cl:ConceptCluster)
            RETURN c.title as title, c.number as number, c.difficulty as difficulty,
                   c.num_concepts as num_concepts, c.estimated_hours as estimated_hours,
                   collect({
                       id: cl.id,
                       name: cl.name,
                       type: cl.type,
                       num_concepts: cl.num_concepts
                   }) as clusters
            """, target_id=target_chapter_id).single()
            
            # Filter paths based on user level
            filtered_paths = self.filter_paths_by_level(paths, user_level)
            
            return {
                'target_chapter': target_info,
                'learning_paths': filtered_paths,
                'recommended_path': filtered_paths[0] if filtered_paths else None,
                'user_level': user_level
            }
    
    def filter_paths_by_level(self, paths: List[Dict], user_level: str) -> List[Dict]:
        """Filter and prioritize paths based on user level"""
        difficulty_scores = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        user_score = difficulty_scores.get(user_level, 2)
        
        filtered_paths = []
        for path in paths:
            # Calculate path difficulty score
            path_difficulties = [difficulty_scores.get(node.get('difficulty', 'intermediate'), 2) 
                               for node in path['nodes']]
            avg_difficulty = sum(path_difficulties) / len(path_difficulties) if path_difficulties else 2
            
            # Add path suitability score
            suitability = 1.0 - abs(avg_difficulty - user_score) / 2
            path['suitability_score'] = max(0.1, suitability)
            path['avg_difficulty'] = avg_difficulty
            
            # Only include paths that are not too advanced for the user
            if avg_difficulty <= user_score + 0.5:
                filtered_paths.append(path)
        
        # Sort by suitability and length
        filtered_paths.sort(key=lambda x: (x['suitability_score'], -x['length']), reverse=True)
        return filtered_paths
    
    def get_concept_clusters_for_chapter(self, chapter_id: str) -> Dict:
        """Get detailed concept clusters for a specific chapter"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
            MATCH (c:Chapter {id: $chapter_id})-[:CONTAINS_CLUSTER]->(cl:ConceptCluster)
            OPTIONAL MATCH (cl)-[:CONTAINS_CONCEPT]->(concept:LearningConcept)
            WITH c, cl, collect({
                id: concept.id,
                text: concept.text,
                type: concept.type,
                difficulty: concept.difficulty,
                confidence: concept.confidence,
                keywords: concept.keywords
            }) as cluster_concepts
            RETURN c.title as chapter_title, c.number as chapter_number,
                   collect({
                       id: cl.id,
                       name: cl.name,
                       type: cl.type,
                       keywords: cl.keywords,
                       concepts: cluster_concepts
                   }) as clusters
            """, chapter_id=chapter_id)
            
            record = result.single()
            if record:
                return {
                    'chapter_title': record['chapter_title'],
                    'chapter_number': record['chapter_number'],
                    'clusters': record['clusters'],
                    'total_clusters': len(record['clusters']),
                    'total_concepts': sum(len(cluster['concepts']) for cluster in record['clusters'])
                }
            return None
    
    def recommend_personalized_concepts(self, completed_concepts: List[str], target_difficulty: str = 'intermediate', limit: int = 10) -> List[Dict]:
        """Recommend personalized learning concepts based on progress"""
        with self.neo4j_driver.session() as session:
            # Find concepts that have all prerequisites completed
            result = session.run("""
            MATCH (c:LearningConcept)
            WHERE NOT c.id IN $completed AND c.difficulty = $difficulty
            OPTIONAL MATCH (prereq:LearningConcept)-[:PREREQUISITE_FOR]->(c)
            WITH c, collect(prereq.id) as all_prereqs
            WHERE all([p IN all_prereqs WHERE p IN $completed]) OR size(all_prereqs) = 0
            MATCH (ch:Chapter)-[:HAS_CONCEPT]->(c)
            OPTIONAL MATCH (cl:ConceptCluster)-[:CONTAINS_CONCEPT]->(c)
            RETURN c.id as id, c.text as text, c.type as type, 
                   c.difficulty as difficulty, c.confidence as confidence,
                   c.keywords as keywords,
                   ch.title as chapter_title, ch.number as chapter_number,
                   cl.name as cluster_name,
                   size(all_prereqs) as num_prereqs
            ORDER BY c.confidence DESC, size(all_prereqs) ASC
            LIMIT $limit
            """, completed=completed_concepts, difficulty=target_difficulty, limit=limit)
            
            recommendations = []
            for record in result:
                recommendations.append({
                    'concept_id': record['id'],
                    'text': record['text'],
                    'type': record['type'],
                    'difficulty': record['difficulty'],
                    'confidence': record['confidence'],
                    'keywords': record['keywords'],
                    'chapter_title': record['chapter_title'],
                    'chapter_number': record['chapter_number'],
                    'cluster_name': record['cluster_name'],
                    'num_prerequisites': record['num_prereqs']
                })
            
            return recommendations
    
    def analyze_learning_progress(self, user_progress: Dict) -> Dict:
        """Analyze detailed learning progress with concept clusters"""
        completed_concepts = user_progress.get('completed_concepts', [])
        target_chapter = user_progress.get('target_chapter')
        
        with self.neo4j_driver.session() as session:
            # Analyze progress by chapter and cluster
            progress_result = session.run("""
            MATCH (c:Chapter)
            OPTIONAL MATCH (c)-[:HAS_CONCEPT]->(concept:LearningConcept)
            OPTIONAL MATCH (c)-[:CONTAINS_CLUSTER]->(cl:ConceptCluster)-[:CONTAINS_CONCEPT]->(cluster_concept:LearningConcept)
            WITH c, 
                 collect(DISTINCT concept.id) as all_concepts,
                 collect(DISTINCT CASE WHEN concept.id IN $completed THEN concept.id END) as completed_chapter_concepts,
                 collect(DISTINCT {
                     cluster_id: cl.id,
                     cluster_name: cl.name,
                     cluster_concepts: collect(DISTINCT cluster_concept.id),
                     completed_cluster_concepts: collect(DISTINCT CASE WHEN cluster_concept.id IN $completed THEN cluster_concept.id END)
                 }) as cluster_progress
            RETURN c.id as chapter_id, c.title as chapter_title, c.number as chapter_number,
                   size(all_concepts) as total_concepts,
                   size([x IN completed_chapter_concepts WHERE x IS NOT NULL]) as completed_concepts,
                   cluster_progress
            ORDER BY toInteger(c.number)
            """, completed=completed_concepts)
            
            chapter_progress = []
            for record in progress_result:
                chapter_data = {
                    'chapter_id': record['chapter_id'],
                    'chapter_title': record['chapter_title'],
                    'chapter_number': record['chapter_number'],
                    'total_concepts': record['total_concepts'],
                    'completed_concepts': record['completed_concepts'],
                    'completion_percentage': (record['completed_concepts'] / record['total_concepts'] * 100) if record['total_concepts'] > 0 else 0,
                    'cluster_progress': []
                }
                
                # Process cluster progress
                for cluster_info in record['cluster_progress']:
                    if cluster_info['cluster_id']:  # Filter out None values
                        cluster_concepts = cluster_info['cluster_concepts']
                        completed_cluster = [c for c in cluster_info['completed_cluster_concepts'] if c is not None]
                        
                        chapter_data['cluster_progress'].append({
                            'cluster_id': cluster_info['cluster_id'],
                            'cluster_name': cluster_info['cluster_name'],
                            'total_concepts': len(cluster_concepts),
                            'completed_concepts': len(completed_cluster),
                            'completion_percentage': (len(completed_cluster) / len(cluster_concepts) * 100) if cluster_concepts else 0
                        })
                
                chapter_progress.append(chapter_data)
            
            # Find knowledge gaps for target chapter
            gaps = []
            if target_chapter:
                gap_result = session.run("""
                MATCH (target:Chapter {id: $target_id})-[:HAS_CONCEPT]->(target_concept:LearningConcept)
                OPTIONAL MATCH (prereq:LearningConcept)-[:PREREQUISITE_FOR]->(target_concept)
                WHERE NOT prereq.id IN $completed
                RETURN target_concept.text as concept,
                       collect(prereq.text) as missing_prerequisites
                """, target_id=target_chapter, completed=completed_concepts)
                
                for record in gap_result:
                    if record['missing_prerequisites']:
                        gaps.append({
                            'concept': record['concept'],
                            'missing_prerequisites': record['missing_prerequisites']
                        })
            
            # Calculate overall statistics
            total_concepts = sum(ch['total_concepts'] for ch in chapter_progress)
            total_completed = sum(ch['completed_concepts'] for ch in chapter_progress)
            overall_percentage = (total_completed / total_concepts * 100) if total_concepts > 0 else 0
            
            return {
                'chapter_progress': chapter_progress,
                'knowledge_gaps': gaps,
                'overall_statistics': {
                    'total_concepts': total_concepts,
                    'completed_concepts': total_completed,
                    'completion_percentage': overall_percentage,
                    'chapters_started': len([ch for ch in chapter_progress if ch['completed_concepts'] > 0]),
                    'chapters_completed': len([ch for ch in chapter_progress if ch['completion_percentage'] >= 90])
                }
            }
    
    def semantic_concept_search(self, query: str, search_type: str = 'all') -> Dict:
        """Perform semantic search across concepts with filtering"""
        logger.info(f"🔍 Semantic concept search: {query} (type: {search_type})")
        
        # Find similar concepts using embeddings
        query_embedding = self.embedding_model.encode([query], device=self.device)
        
        # Get concept embeddings and filter by type if specified
        filtered_concepts = self.concepts
        if search_type != 'all':
            filtered_concepts = [c for c in self.concepts if c['type'] == search_type]
        
        if not filtered_concepts:
            return {'query': query, 'results': [], 'message': f'No concepts found for type: {search_type}'}
        
        concept_embeddings = np.array([c['embedding'] for c in filtered_concepts])
        
        # Calculate similarities
        if self.device == "cuda":
            query_tensor = torch.tensor(query_embedding).cuda()
            concept_tensor = torch.tensor(concept_embeddings).cuda()
            similarities = torch.nn.functional.cosine_similarity(
                query_tensor, concept_tensor, dim=1
            ).cpu().numpy()
        else:
            similarities = cosine_similarity(query_embedding, concept_embeddings)[0]
        
        # Get top concepts
        top_indices = np.argsort(similarities)[::-1][:10]
        
        results = []
        for idx in top_indices:
            concept = filtered_concepts[idx]
            # Find chapter and cluster info
            chapter = next((ch for ch in self.chapters if ch['id'] == concept['chapter_id']), None)
            
            cluster_info = None
            if chapter and concept.get('cluster_id'):
                for cluster in chapter.get('concept_clusters', []):
                    if cluster['id'] == concept['cluster_id']:
                        cluster_info = cluster
                        break
            
            results.append({
                'concept': concept,
                'similarity': float(similarities[idx]),
                'chapter_info': {
                    'title': chapter['title'] if chapter else 'Unknown',
                    'number': chapter['number'] if chapter else 'Unknown'
                },
                'cluster_info': cluster_info
            })
        
        # Use LlamaIndex for detailed RAG response
        query_engine = self.vector_index.as_query_engine(
            similarity_top_k=5,
            response_mode="tree_summarize"
        )
        
        rag_response = query_engine.query(query)
        
        return {
            'query': query,
            'search_type': search_type,
            'rag_response': str(rag_response),
            'concept_results': results,
            'num_results': len(results),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_enhanced_system_overview(self) -> Dict:
        """Get comprehensive overview of the enhanced knowledge graph system"""
        with self.neo4j_driver.session() as session:
            # Get detailed graph statistics
            stats_result = session.run("""
            MATCH (c:Chapter)
            OPTIONAL MATCH (c)-[:HAS_CONCEPT]->(concept:LearningConcept)
            OPTIONAL MATCH (c)-[:CONTAINS_CLUSTER]->(cluster:ConceptCluster)
            OPTIONAL MATCH (c)-[:PREREQUISITE_FOR]->(next_c:Chapter)
            RETURN count(DISTINCT c) as num_chapters,
                   count(DISTINCT concept) as num_concepts,
                   count(DISTINCT cluster) as num_clusters,
                   count(DISTINCT next_c) as chapters_with_prereqs,
                   avg(c.estimated_hours) as avg_chapter_hours,
                   avg(c.num_concepts) as avg_concepts_per_chapter
            """)
            
            stats = stats_result.single()
            
            # Get concept type distribution
            type_result = session.run("""
            MATCH (c:LearningConcept)
            RETURN c.type as type, count(c) as count
            ORDER BY count DESC
            """)
            
            type_distribution = {record['type']: record['count'] for record in type_result}
            
            # Get difficulty distribution
            difficulty_result = session.run("""
            MATCH (c:LearningConcept)
            RETURN c.difficulty as difficulty, count(c) as count
            ORDER BY count DESC
            """)
            
            difficulty_distribution = {record['difficulty']: record['count'] for record in difficulty_result}
            
            # Get cluster type distribution
            cluster_result = session.run("""
            MATCH (cl:ConceptCluster)
            RETURN cl.type as cluster_type, count(cl) as count
            ORDER BY count DESC
            """)
            
            cluster_distribution = {record['cluster_type']: record['count'] for record in cluster_result}
            
            return {
                'basic_stats': {
                    'num_chapters': stats['num_chapters'],
                    'num_concepts': stats['num_concepts'],
                    'num_clusters': stats['num_clusters'],
                    'chapters_with_prerequisites': stats['chapters_with_prereqs'],
                    'avg_chapter_hours': round(stats['avg_chapter_hours'], 2),
                    'avg_concepts_per_chapter': round(stats['avg_concepts_per_chapter'], 1)
                },
                'distributions': {
                    'concept_types': type_distribution,
                    'difficulty_levels': difficulty_distribution,
                    'cluster_types': cluster_distribution
                },
                'system_info': {
                    'total_estimated_hours': sum(ch['estimated_hours'] for ch in self.chapters),
                    'gpu_accelerated': self.device == "cuda",
                    'system_status': 'operational',
                    'database_type': 'enhanced_chapter_based'
                }
            }
    
    def interactive_enhanced_mode(self):
        """Enhanced interactive command-line interface"""
        print("\n🎯 GPU-Accelerated Enhanced Chapter-Based Knowledge Graph Query System")
        print("🧠 With Concept Clusters and Advanced Learning Analytics")
        print("=" * 80)
        print("Available commands:")
        print("  search <query>           - Semantic search across all concepts")
        print("  search-type <type> <query> - Search within specific concept type")
        print("  chapter <chapter_id>     - Get detailed chapter information with clusters")
        print("  cluster <cluster_id>     - Get all concepts in a cluster")
        print("  path <chapter_id> <level> - Find learning path (level: beginner/intermediate/advanced)")
        print("  recommend <difficulty>   - Get personalized concept recommendations")
        print("  progress <completed_ids> - Analyze learning progress (comma-separated concept IDs)")
        print("  overview                 - Enhanced system overview")
        print("  list-chapters           - List all chapters with cluster info")
        print("  list-types              - List all concept types")
        print("  quit                    - Exit")
        print("=" * 80)
        
        while True:
            try:
                user_input = input("\n🤖 Enter command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                parts = user_input.split(' ', 2)
                command = parts[0].lower()
                
                if command == 'search':
                    if len(parts) < 2:
                        print("❌ Please provide a search query")
                        continue
                    
                    query = ' '.join(parts[1:])
                    result = self.semantic_concept_search(query)
                    
                    print(f"\n📚 Search Results for: '{query}'")
                    print("-" * 60)
                    print(f"🤖 AI Response: {result['rag_response'][:300]}...")
                    print(f"\n🔍 Top Matching Concepts:")
                    
                    for i, item in enumerate(result['concept_results'][:5], 1):
                        concept = item['concept']
                        print(f"  {i}. {concept['text'][:80]}...")
                        print(f"     📊 Type: {concept['type']}, Difficulty: {concept['difficulty']}")
                        print(f"     📚 Chapter {item['chapter_info']['number']}: {item['chapter_info']['title']}")
                        if item['cluster_info']:
                            print(f"     🔗 Cluster: {item['cluster_info']['name']}")
                        print(f"     🎯 Similarity: {item['similarity']:.3f}")
                
                elif command == 'search-type':
                    if len(parts) < 3:
                        print("❌ Please provide concept type and search query")
                        continue
                    
                    concept_type = parts[1]
                    query = parts[2]
                    result = self.semantic_concept_search(query, concept_type)
                    
                    print(f"\n📚 {concept_type.title()} Search Results for: '{query}'")
                    print("-" * 60)
                    print(f"Found {result['num_results']} matching concepts")
                    
                    for i, item in enumerate(result['concept_results'][:5], 1):
                        concept = item['concept']
                        print(f"  {i}. {concept['text'][:80]}...")
                        print(f"     📚 Chapter {item['chapter_info']['number']}: {item['chapter_info']['title']}")
                        print(f"     🎯 Similarity: {item['similarity']:.3f}")
                
                elif command == 'chapter':
                    if len(parts) < 2:
                        print("❌ Please provide a chapter ID")
                        continue
                    
                    chapter_id = parts[1]
                    result = self.get_concept_clusters_for_chapter(chapter_id)
                    
                    if result:
                        print(f"\n📚 Chapter {result['chapter_number']}: {result['chapter_title']}")
                        print("-" * 60)
                        print(f"📊 {result['total_clusters']} clusters, {result['total_concepts']} total concepts")
                        
                        for i, cluster in enumerate(result['clusters'], 1):
                            print(f"\n  🔗 Cluster {i}: {cluster['name']} ({cluster['type']})")
                            print(f"     📝 {len(cluster['concepts'])} concepts")
                            
                            for j, concept in enumerate(cluster['concepts'][:3], 1):
                                print(f"       {j}. {concept['text'][:60]}... ({concept['difficulty']})")
                            
                            if len(cluster['concepts']) > 3:
                                print(f"       ... and {len(cluster['concepts']) - 3} more concepts")
                    else:
                        print(f"❌ Chapter not found: {chapter_id}")
                
                elif command == 'cluster':
                    if len(parts) < 2:
                        print("❌ Please provide a cluster ID")
                        continue
                    
                    cluster_id = parts[1]
                    result = self.find_concepts_in_cluster(cluster_id)
                    
                    if result:
                        print(f"\n🔗 Cluster: {result['cluster_name']} ({result['cluster_type']})")
                        print(f"📚 From Chapter {result['chapter_number']}: {result['chapter_title']}")
                        print("-" * 60)
                        print(f"📝 {result['num_concepts']} concepts:")
                        
                        for i, concept in enumerate(result['concepts'], 1):
                            print(f"  {i}. {concept['text']}")
                            print(f"     📊 Type: {concept['type']}, Difficulty: {concept['difficulty']}")
                            print(f"     🎯 Confidence: {concept['confidence']:.2f}")
                    else:
                        print(f"❌ Cluster not found: {cluster_id}")
                
                elif command == 'path':
                    args = parts[1:] if len(parts) > 1 else []
                    if not args:
                        print("❌ Please provide chapter ID and optionally user level")
                        continue
                    
                    chapter_id = args[0]
                    user_level = args[1] if len(args) > 1 else 'intermediate'
                    
                    result = self.find_learning_path_enhanced(chapter_id, user_level)
                    
                    if result['learning_paths']:
                        print(f"\n🗺️ Learning Path to: {result['target_chapter']['title']}")
                        print(f"👤 User Level: {user_level}")
                        print("-" * 60)
                        
                        path = result['recommended_path']
                        print(f"📊 Recommended Path ({path['total_hours']:.1f} hours, suitability: {path['suitability_score']:.2f}):")
                        
                        for i, node in enumerate(path['nodes'], 1):
                            print(f"  {i}. Chapter {node['number']}: {node['title']}")
                            print(f"     ⏱️ {node['estimated_hours']:.1f}h, 🎯 {node['num_concepts']} concepts, 🔗 {node['num_clusters']} clusters")
                    else:
                        print(f"❌ No learning path found for chapter: {chapter_id}")
                
                elif command == 'recommend':
                    difficulty = parts[1] if len(parts) > 1 else 'intermediate'
                    completed = []  # In a real app, this would come from user profile
                    
                    recommendations = self.recommend_personalized_concepts(completed, difficulty)
                    
                    print(f"\n💡 Personalized Recommendations ({difficulty} level):")
                    print("-" * 60)
                    
                    for i, rec in enumerate(recommendations, 1):
                        print(f"  {i}. {rec['text'][:70]}...")
                        print(f"     📚 Chapter {rec['chapter_number']}: {rec['chapter_title']}")
                        print(f"     🔗 Cluster: {rec['cluster_name'] or 'N/A'}")
                        print(f"     📊 Type: {rec['type']}, Confidence: {rec['confidence']:.2f}")
                
                elif command == 'overview':
                    result = self.get_enhanced_system_overview()
                    
                    print(f"\n📊 Enhanced System Overview:")
                    print("-" * 60)
                    stats = result['basic_stats']
                    print(f"📚 Chapters: {stats['num_chapters']}")
                    print(f"🧠 Learning Concepts: {stats['num_concepts']}")
                    print(f"🔗 Concept Clusters: {stats['num_clusters']}")
                    print(f"➡️  Prerequisites: {stats['chapters_with_prerequisites']} chapters have prerequisites")
                    print(f"⏱️ Total Study Time: {result['system_info']['total_estimated_hours']:.1f} hours")
                    print(f"🚀 GPU Acceleration: {'Enabled' if result['system_info']['gpu_accelerated'] else 'Disabled'}")
                    
                    print(f"\n📈 Concept Type Distribution:")
                    for ctype, count in list(result['distributions']['concept_types'].items())[:5]:
                        print(f"   {ctype}: {count}")
                    
                    print(f"\n📊 Difficulty Distribution:")
                    for diff, count in result['distributions']['difficulty_levels'].items():
                        print(f"   {diff}: {count}")
                
                elif command == 'list-chapters':
                    print(f"\n📋 All Chapters ({len(self.chapters)} total):")
                    print("-" * 60)
                    
                    sorted_chapters = sorted(self.chapters, key=lambda x: int(x['number']) if x['number'].isdigit() else 999)
                    for ch in sorted_chapters:
                        concepts_count = len(ch['learning_concepts'])
                        clusters_count = len(ch.get('concept_clusters', []))
                        print(f"  Chapter {ch['number']}: {ch['title']}")
                        print(f"    📝 {concepts_count} concepts, 🔗 {clusters_count} clusters, ⏱️ {ch['estimated_hours']:.1f}h")
                
                elif command == 'list-types':
                    type_counts = Counter([c['type'] for c in self.concepts])
                    print(f"\n📝 Concept Types ({len(type_counts)} total):")
                    print("-" * 60)
                    
                    for ctype, count in type_counts.most_common():
                        print(f"  {ctype.replace('_', ' ').title()}: {count}")
                
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def close_connections(self):
        """Close database connections"""
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()

def main():
    """Main execution function"""
    system = None
    try:
        system = GPUAcceleratedEnhancedChapterQuerySystem()
        system.interactive_enhanced_mode()
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        raise
    
    finally:
        if system:
            system.close_connections()

if __name__ == "__main__":
    main()
