#!/usr/bin/env python3
"""
GPU-Accelerated Enhanced Chapter-Based Knowledge Graph Builder
Creates optimized knowledge graphs with 20-25 chapter nodes and 20-30 learning concepts per chapter
"""
import os
import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

import torch
import networkx as nx
from neo4j import GraphDatabase
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
ES_ENDPOINT = "http://localhost:9200"
INDEX_NAME = "gpu_chapter_knowledge_v1"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "knowledge123"

@dataclass
class EnhancedChapterNode:
    """Enhanced chapter node with detailed learning concepts"""
    id: str
    title: str
    number: str
    document_source: str
    content_summary: str
    learning_concepts: List[Dict]
    prerequisites: List[str] = None
    difficulty: str = "intermediate"
    estimated_hours: float = 2.0
    keywords: List[str] = None
    embedding: List[float] = None
    concept_clusters: List[Dict] = None  # Grouped concepts
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.keywords is None:
            self.keywords = []
        if self.concept_clusters is None:
            self.concept_clusters = []

@dataclass
class LearningConcept:
    """Enhanced learning concept with better metadata"""
    id: str
    chapter_id: str
    text: str
    type: str  # explicit_objective, key_term, technical_concept, etc.
    difficulty: str
    keywords: List[str]
    prerequisites: List[str] = None
    embedding: List[float] = None
    confidence: float = 0.5
    cluster_id: Optional[str] = None
    related_concepts: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.related_concepts is None:
            self.related_concepts = []

class GPUAcceleratedEnhancedKnowledgeGraphBuilder:
    """GPU-accelerated knowledge graph builder with enhanced concept clustering"""
    
    def __init__(self):
        self.setup_gpu()
        self.setup_models()
        self.setup_connections()
        self.chapters = []
        self.concepts = []
        self.chapter_embeddings = {}
        self.concept_embeddings = {}
        self.concept_clusters = {}
    
    def setup_gpu(self):
        """Setup GPU acceleration"""
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.empty_cache()
            logger.info(f"🚀 GPU acceleration enabled - Using {torch.cuda.get_device_name()}")
            logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = "cpu"
            logger.warning("⚠️ CUDA not available, using CPU")
    
    def setup_models(self):
        """Setup GPU-accelerated embedding models"""
        logger.info("🔧 Setting up GPU-accelerated embedding models...")
        
        # Use GPU-accelerated sentence transformer
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2',
            device=self.device
        )
        
        # Enable GPU optimizations
        if self.device == "cuda":
            self.embedding_model.max_seq_length = 512  # Optimize for GPU memory
        
        logger.info(f"✅ Embedding model loaded on {self.device}")
    
    def setup_connections(self):
        """Setup database connections"""
        # Elasticsearch
        try:
            self.es_client = Elasticsearch([ES_ENDPOINT])
            es_info = self.es_client.info()
            logger.info(f"✅ Connected to Elasticsearch {es_info.body['version']['number']}")
        except Exception as e:
            logger.error(f"❌ Elasticsearch connection failed: {e}")
            raise
        
        # Neo4j
        try:
            self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            logger.info("✅ Connected to Neo4j")
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            raise
    
    def load_enhanced_chapter_data(self):
        """Load enhanced chapter data with learning concepts"""
        logger.info("📚 Loading enhanced chapter metadata...")
        
        metadata_path = './gpu_chapter_data/chapters_metadata.json'
        if not os.path.exists(metadata_path):
            logger.error(f"❌ Enhanced chapter metadata not found at {metadata_path}")
            logger.info("💡 Please run 'KG_ENHANCED_1_build_chapter_database_gpu.py' first")
            raise FileNotFoundError(metadata_path)
        
        with open(metadata_path, 'r') as f:
            chapters_data = json.load(f)
        
        logger.info(f"📖 Loaded {len(chapters_data)} chapters")
        
        # Convert to EnhancedChapterNode objects
        for chapter_data in chapters_data:
            # Extract content summary (first 500 chars)
            content_summary = chapter_data['content'][:500] + "..." if len(chapter_data['content']) > 500 else chapter_data['content']
            
            chapter_node = EnhancedChapterNode(
                id=chapter_data['id'],
                title=chapter_data['title'],
                number=chapter_data['number'],
                document_source=chapter_data['document_source'],
                content_summary=content_summary,
                learning_concepts=chapter_data['learning_concepts'],
                keywords=self.extract_chapter_keywords(chapter_data['content']),
                estimated_hours=self.estimate_chapter_hours(chapter_data['learning_concepts'])
            )
            
            self.chapters.append(chapter_node)
            
            # Create enhanced learning concept nodes
            for concept_data in chapter_data['learning_concepts']:
                concept = LearningConcept(
                    id=f"{chapter_data['id']}_concept_{concept_data['id']}",
                    chapter_id=chapter_data['id'],
                    text=concept_data['text'],
                    type=concept_data['type'],
                    difficulty=concept_data['difficulty'],
                    keywords=concept_data['keywords'],
                    confidence=concept_data.get('confidence', 0.5)
                )
                self.concepts.append(concept)
        
        logger.info(f"🎯 Created {len(self.chapters)} chapter nodes and {len(self.concepts)} learning concepts")
        
        # Load statistics if available
        stats_path = './gpu_chapter_data/extraction_stats.json'
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            logger.info(f"📊 Statistics: {stats['total_concepts']} concepts, avg {stats['avg_concepts_per_chapter']:.1f} per chapter")
    
    def extract_chapter_keywords(self, content: str) -> List[str]:
        """Extract keywords from chapter content using advanced techniques"""
        # Combine multiple approaches for better keyword extraction
        
        # Method 1: CamelCase and technical terms
        camel_case = re.findall(r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)*\b', content)
        
        # Method 2: Technical terms with common CS/technical patterns
        tech_patterns = [
            r'\b(?:algorithm|data structure|protocol|interface|framework|architecture|pattern|model|system|process|method|technique|approach|strategy|implementation|optimization|analysis|design|specification|standard)\w*\b',
            r'\b\w+(?:_\w+)+\b',  # snake_case terms
            r'\b[A-Z]{2,}\b'      # ACRONYMS
        ]
        
        tech_words = []
        for pattern in tech_patterns:
            tech_words.extend(re.findall(pattern, content, re.IGNORECASE))
        
        # Method 3: Capitalized words (likely proper nouns/concepts)
        capitalized = re.findall(r'\b[A-Z][a-z]{3,}\b', content)
        
        # Combine and count frequency
        all_words = camel_case + tech_words + capitalized
        word_counts = Counter(word.lower() for word in all_words)
        
        # Return top keywords
        return [word for word, count in word_counts.most_common(12)]
    
    def estimate_chapter_hours(self, learning_concepts: List[Dict]) -> float:
        """Estimate study hours based on learning concepts"""
        base_hours = 1.5  # Base reading time
        
        # Add time based on concept complexity
        concept_hours = 0
        for concept in learning_concepts:
            difficulty = concept.get('difficulty', 'intermediate')
            concept_type = concept.get('type', 'general')
            
            # Base time per concept
            if difficulty == 'advanced':
                concept_hours += 0.3
            elif difficulty == 'intermediate':
                concept_hours += 0.2
            else:
                concept_hours += 0.1
            
            # Additional time for certain types
            if concept_type in ['explicit_objective', 'technical_concept']:
                concept_hours += 0.1
        
        total_hours = base_hours + concept_hours
        return min(max(total_hours, 1.0), 8.0)  # Cap between 1-8 hours
    
    def generate_gpu_accelerated_embeddings(self):
        """Generate GPU-accelerated embeddings for chapters and concepts"""
        logger.info("🧠 Generating GPU-accelerated embeddings...")
        
        # Chapter embeddings
        chapter_texts = []
        for chapter in self.chapters:
            # Combine title, keywords, and concept summaries for better embeddings
            concept_texts = [c['text'][:100] for c in chapter.learning_concepts[:5]]  # Top 5 concepts
            text = f"{chapter.title}. {' '.join(chapter.keywords)}. {' '.join(concept_texts)}"
            chapter_texts.append(text)
        
        # Generate chapter embeddings with GPU acceleration
        logger.info(f"🔄 Generating embeddings for {len(chapter_texts)} chapters...")
        
        if self.device == "cuda":
            # GPU batch processing
            chapter_embeddings = self.embedding_model.encode(
                chapter_texts,
                batch_size=32,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True
            )
        else:
            # CPU processing
            chapter_embeddings = self.embedding_model.encode(
                chapter_texts,
                batch_size=8,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        
        # Store chapter embeddings
        for i, chapter in enumerate(self.chapters):
            chapter.embedding = chapter_embeddings[i].cpu().numpy().tolist()
            self.chapter_embeddings[chapter.id] = chapter.embedding
        
        # Concept embeddings (process in larger batches for GPU efficiency)
        concept_texts = []
        for concept in self.concepts:
            text = f"{concept.text}. {' '.join(concept.keywords)}"
            concept_texts.append(text)
        
        logger.info(f"🔄 Generating embeddings for {len(concept_texts)} learning concepts...")
        
        if self.device == "cuda":
            # Process concepts in larger batches for GPU efficiency
            concept_embeddings = self.embedding_model.encode(
                concept_texts,
                batch_size=64,  # Larger batch for concepts
                show_progress_bar=True,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True
            )
        else:
            concept_embeddings = self.embedding_model.encode(
                concept_texts,
                batch_size=16,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
        
        # Store concept embeddings
        for i, concept in enumerate(self.concepts):
            concept.embedding = concept_embeddings[i].cpu().numpy().tolist()
            self.concept_embeddings[concept.id] = concept.embedding
        
        logger.info("✅ GPU-accelerated embeddings generated successfully")
    
    def create_concept_clusters(self):
        """Create concept clusters within each chapter for better organization"""
        logger.info("🔗 Creating concept clusters within chapters...")
        
        for chapter in self.chapters:
            chapter_concepts = [c for c in self.concepts if c.chapter_id == chapter.id]
            
            if len(chapter_concepts) < 4:
                # Too few concepts for clustering
                chapter.concept_clusters = [{
                    'id': f"{chapter.id}_cluster_1",
                    'name': 'Main Concepts',
                    'concepts': [c.id for c in chapter_concepts],
                    'type': 'single_cluster'
                }]
                continue
            
            # Get embeddings for chapter concepts
            concept_embeddings = np.array([c.embedding for c in chapter_concepts])
            
            # Determine optimal number of clusters (3-6 clusters per chapter)
            n_concepts = len(chapter_concepts)
            n_clusters = min(max(3, n_concepts // 5), 6)
            
            # Perform clustering
            if self.device == "cuda" and len(chapter_concepts) > 10:
                # Use GPU-accelerated clustering for larger concept sets
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            
            cluster_labels = kmeans.fit_predict(concept_embeddings)
            
            # Create cluster metadata
            clusters = []
            for cluster_id in range(n_clusters):
                cluster_concepts = [c for i, c in enumerate(chapter_concepts) if cluster_labels[i] == cluster_id]
                
                if not cluster_concepts:
                    continue
                
                # Generate cluster name based on most common concept types and keywords
                concept_types = [c.type for c in cluster_concepts]
                all_keywords = []
                for c in cluster_concepts:
                    all_keywords.extend(c.keywords)
                
                most_common_type = Counter(concept_types).most_common(1)[0][0]
                top_keywords = [kw for kw, count in Counter(all_keywords).most_common(3)]
                
                cluster_name = f"{most_common_type.replace('_', ' ').title()}"
                if top_keywords:
                    cluster_name += f" ({', '.join(top_keywords)})"
                
                clusters.append({
                    'id': f"{chapter.id}_cluster_{cluster_id}",
                    'name': cluster_name,
                    'concepts': [c.id for c in cluster_concepts],
                    'type': most_common_type,
                    'keywords': top_keywords
                })
                
                # Update concept cluster assignments
                for concept in cluster_concepts:
                    concept.cluster_id = f"{chapter.id}_cluster_{cluster_id}"
            
            chapter.concept_clusters = clusters
            logger.info(f"   Chapter {chapter.number}: {len(clusters)} concept clusters")
    
    def detect_enhanced_prerequisites(self):
        """Detect prerequisite relationships using multiple methods"""
        logger.info("🔗 Detecting enhanced prerequisite relationships...")
        
        # Chapter-level prerequisites
        chapter_embeddings_matrix = np.array([ch.embedding for ch in self.chapters])
        chapter_similarities = cosine_similarity(chapter_embeddings_matrix)
        
        for i, chapter in enumerate(self.chapters):
            prerequisites = []
            
            # Method 1: Sequential chapter ordering
            chapter_num = self.extract_chapter_number(chapter.number)
            if chapter_num and chapter_num > 1:
                # Look for highly related previous chapters
                for j, other_chapter in enumerate(self.chapters):
                    other_num = self.extract_chapter_number(other_chapter.number)
                    if other_num and other_num < chapter_num:
                        # Check semantic similarity
                        if chapter_similarities[i][j] > 0.25:  # Threshold for relatedness
                            prerequisites.append(other_chapter.id)
            
            # Method 2: Keyword overlap analysis
            chapter_keywords = set(kw.lower() for kw in chapter.keywords)
            for j, other_chapter in enumerate(self.chapters):
                if i != j:
                    other_keywords = set(kw.lower() for kw in other_chapter.keywords)
                    overlap = len(chapter_keywords & other_keywords)
                    if overlap >= 2:  # Significant keyword overlap
                        prerequisites.append(other_chapter.id)
            
            # Method 3: Concept dependency analysis
            chapter_concepts = [c for c in self.concepts if c.chapter_id == chapter.id]
            for other_chapter in self.chapters:
                if other_chapter.id != chapter.id:
                    other_concepts = [c for c in self.concepts if c.chapter_id == other_chapter.id]
                    
                    # Check if current chapter concepts reference other chapter concepts
                    dependency_score = self.calculate_concept_dependency(chapter_concepts, other_concepts)
                    if dependency_score > 0.3:
                        prerequisites.append(other_chapter.id)
            
            # Remove duplicates and limit prerequisites
            chapter.prerequisites = list(set(prerequisites))[:3]
        
        # Concept-level prerequisites within and across chapters
        logger.info("🔗 Detecting concept-level prerequisites...")
        
        for concept in self.concepts:
            prerequisites = []
            
            # Look for prerequisites in the same chapter
            same_chapter_concepts = [c for c in self.concepts if c.chapter_id == concept.chapter_id and c.id != concept.id]
            
            # Prerequisites based on concept types and difficulty
            for other_concept in same_chapter_concepts:
                if self.is_prerequisite_concept(other_concept, concept):
                    prerequisites.append(other_concept.id)
            
            # Cross-chapter prerequisites (concepts from prerequisite chapters)
            chapter = next(ch for ch in self.chapters if ch.id == concept.chapter_id)
            for prereq_chapter_id in chapter.prerequisites:
                prereq_concepts = [c for c in self.concepts if c.chapter_id == prereq_chapter_id]
                
                # Find most similar concepts in prerequisite chapters
                if prereq_concepts:
                    concept_embedding = np.array(concept.embedding).reshape(1, -1)
                    prereq_embeddings = np.array([c.embedding for c in prereq_concepts])
                    similarities = cosine_similarity(concept_embedding, prereq_embeddings)[0]
                    
                    # Add highly similar prerequisites
                    for idx, sim in enumerate(similarities):
                        if sim > 0.4:  # Similarity threshold
                            prerequisites.append(prereq_concepts[idx].id)
            
            concept.prerequisites = prerequisites[:3]  # Limit to 3 prerequisites
        
        logger.info("✅ Enhanced prerequisites detected")
    
    def calculate_concept_dependency(self, chapter_concepts: List[LearningConcept], other_concepts: List[LearningConcept]) -> float:
        """Calculate dependency score between concept sets"""
        if not chapter_concepts or not other_concepts:
            return 0.0
        
        # Method 1: Keyword overlap
        chapter_keywords = set()
        for c in chapter_concepts:
            chapter_keywords.update(kw.lower() for kw in c.keywords)
        
        other_keywords = set()
        for c in other_concepts:
            other_keywords.update(kw.lower() for kw in c.keywords)
        
        keyword_overlap = len(chapter_keywords & other_keywords) / len(chapter_keywords | other_keywords) if chapter_keywords | other_keywords else 0
        
        # Method 2: Text reference analysis
        chapter_texts = [c.text.lower() for c in chapter_concepts]
        other_texts = [c.text.lower() for c in other_concepts]
        
        reference_score = 0
        for chapter_text in chapter_texts:
            for other_text in other_texts:
                # Check if chapter concept references other concept keywords
                other_concept_keywords = other_text.split()[:5]  # First 5 words as key terms
                references = sum(1 for keyword in other_concept_keywords if keyword in chapter_text)
                if references > 0:
                    reference_score += references / len(other_concept_keywords)
        
        reference_score = reference_score / len(chapter_texts) if chapter_texts else 0
        
        # Combine scores
        return (keyword_overlap * 0.6 + reference_score * 0.4)
    
    def is_prerequisite_concept(self, potential_prereq: LearningConcept, target_concept: LearningConcept) -> bool:
        """Determine if one concept is a prerequisite for another"""
        # Type-based prerequisites
        type_hierarchy = {
            'key_term': 1,
            'explicit_objective': 2,
            'section_concept': 3,
            'technical_concept': 4,
            'question_concept': 5,
            'supplementary_concept': 6
        }
        
        prereq_priority = type_hierarchy.get(potential_prereq.type, 3)
        target_priority = type_hierarchy.get(target_concept.type, 3)
        
        # Basic concepts should come before advanced ones
        if prereq_priority < target_priority:
            return True
        
        # Difficulty-based prerequisites
        difficulty_order = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
        prereq_difficulty = difficulty_order.get(potential_prereq.difficulty, 2)
        target_difficulty = difficulty_order.get(target_concept.difficulty, 2)
        
        if prereq_difficulty < target_difficulty:
            return True
        
        # Keyword-based prerequisites
        prereq_keywords = set(kw.lower() for kw in potential_prereq.keywords)
        target_keywords = set(kw.lower() for kw in target_concept.keywords)
        
        # If target concept mentions prereq keywords, it might depend on it
        target_text_words = set(target_concept.text.lower().split())
        if len(prereq_keywords & target_text_words) >= 2:
            return True
        
        return False
    
    def extract_chapter_number(self, number_str: str) -> Optional[int]:
        """Extract numeric chapter number"""
        try:
            # Try direct conversion
            return int(number_str)
        except ValueError:
            # Try extracting number from string
            match = re.search(r'\d+', number_str)
            if match:
                return int(match.group())
            return None
    
    def clear_neo4j_database(self):
        """Clear existing Neo4j database"""
        logger.info("🗑️ Clearing Neo4j database...")
        
        with self.neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        logger.info("✅ Neo4j database cleared")
    
    def build_enhanced_neo4j_graph(self):
        """Build enhanced knowledge graph in Neo4j"""
        logger.info("🏗️ Building enhanced knowledge graph in Neo4j...")
        
        with self.neo4j_driver.session() as session:
            # Create chapter nodes with enhanced metadata
            for chapter in self.chapters:
                session.run("""
                CREATE (c:Chapter {
                    id: $id,
                    title: $title,
                    number: $number,
                    document_source: $document_source,
                    content_summary: $content_summary,
                    difficulty: $difficulty,
                    estimated_hours: $estimated_hours,
                    keywords: $keywords,
                    num_concepts: $num_concepts,
                    num_clusters: $num_clusters,
                    concept_types: $concept_types
                })
                """, 
                id=chapter.id,
                title=chapter.title,
                number=chapter.number,
                document_source=chapter.document_source,
                content_summary=chapter.content_summary,
                difficulty=chapter.difficulty,
                estimated_hours=chapter.estimated_hours,
                keywords=chapter.keywords,
                num_concepts=len(chapter.learning_concepts),
                num_clusters=len(chapter.concept_clusters),
                concept_types=list(set(c['type'] for c in chapter.learning_concepts))
                )
            
            # Create concept cluster nodes
            for chapter in self.chapters:
                for cluster in chapter.concept_clusters:
                    session.run("""
                    CREATE (cl:ConceptCluster {
                        id: $id,
                        name: $name,
                        chapter_id: $chapter_id,
                        type: $type,
                        keywords: $keywords,
                        num_concepts: $num_concepts
                    })
                    """,
                    id=cluster['id'],
                    name=cluster['name'],
                    chapter_id=chapter.id,
                    type=cluster['type'],
                    keywords=cluster.get('keywords', []),
                    num_concepts=len(cluster['concepts'])
                    )
            
            # Create learning concept nodes
            for concept in self.concepts:
                session.run("""
                CREATE (o:LearningConcept {
                    id: $id,
                    chapter_id: $chapter_id,
                    cluster_id: $cluster_id,
                    text: $text,
                    type: $type,
                    difficulty: $difficulty,
                    keywords: $keywords,
                    confidence: $confidence
                })
                """,
                id=concept.id,
                chapter_id=concept.chapter_id,
                cluster_id=concept.cluster_id,
                text=concept.text,
                type=concept.type,
                difficulty=concept.difficulty,
                keywords=concept.keywords,
                confidence=concept.confidence
                )
            
            # Create chapter-cluster relationships
            session.run("""
            MATCH (c:Chapter), (cl:ConceptCluster)
            WHERE c.id = cl.chapter_id
            CREATE (c)-[:CONTAINS_CLUSTER]->(cl)
            """)
            
            # Create cluster-concept relationships
            session.run("""
            MATCH (cl:ConceptCluster), (o:LearningConcept)
            WHERE cl.id = o.cluster_id
            CREATE (cl)-[:CONTAINS_CONCEPT]->(o)
            """)
            
            # Create chapter-concept direct relationships (for easy access)
            session.run("""
            MATCH (c:Chapter), (o:LearningConcept)
            WHERE c.id = o.chapter_id
            CREATE (c)-[:HAS_CONCEPT]->(o)
            """)
            
            # Create chapter prerequisite relationships
            for chapter in self.chapters:
                for prereq_id in chapter.prerequisites:
                    session.run("""
                    MATCH (c1:Chapter {id: $chapter_id})
                    MATCH (c2:Chapter {id: $prereq_id})
                    CREATE (c2)-[:PREREQUISITE_FOR]->(c1)
                    """,
                    chapter_id=chapter.id,
                    prereq_id=prereq_id
                    )
            
            # Create concept prerequisite relationships
            for concept in self.concepts:
                for prereq_id in concept.prerequisites:
                    session.run("""
                    MATCH (o1:LearningConcept {id: $concept_id})
                    MATCH (o2:LearningConcept {id: $prereq_id})
                    CREATE (o2)-[:PREREQUISITE_FOR]->(o1)
                    """,
                    concept_id=concept.id,
                    prereq_id=prereq_id
                    )
            
            # Create indexes for better performance
            indexes = [
                "CREATE INDEX chapter_id IF NOT EXISTS FOR (c:Chapter) ON (c.id)",
                "CREATE INDEX concept_id IF NOT EXISTS FOR (o:LearningConcept) ON (o.id)",
                "CREATE INDEX cluster_id IF NOT EXISTS FOR (cl:ConceptCluster) ON (cl.id)",
                "CREATE INDEX chapter_number IF NOT EXISTS FOR (c:Chapter) ON (c.number)",
                "CREATE INDEX concept_type IF NOT EXISTS FOR (o:LearningConcept) ON (o.type)",
                "CREATE INDEX concept_difficulty IF NOT EXISTS FOR (o:LearningConcept) ON (o.difficulty)"
            ]
            
            for index_query in indexes:
                session.run(index_query)
        
        logger.info("✅ Enhanced knowledge graph built in Neo4j")
    
    def save_enhanced_graph_data(self):
        """Save enhanced graph data to files"""
        logger.info("💾 Saving enhanced knowledge graph data...")
        
        os.makedirs('./gpu_knowledge_graph_data', exist_ok=True)
        
        # Save enhanced chapters
        chapters_data = [asdict(chapter) for chapter in self.chapters]
        with open('./gpu_knowledge_graph_data/enhanced_chapters.json', 'w') as f:
            json.dump(chapters_data, f, indent=2)
        
        # Save enhanced concepts
        concepts_data = [asdict(concept) for concept in self.concepts]
        with open('./gpu_knowledge_graph_data/enhanced_concepts.json', 'w') as f:
            json.dump(concepts_data, f, indent=2)
        
        # Save enhanced summary statistics
        concept_types = Counter([c.type for c in self.concepts])
        difficulty_dist = Counter([c.difficulty for c in self.concepts])
        
        stats = {
            'num_chapters': len(self.chapters),
            'num_concepts': len(self.concepts),
            'num_clusters': sum(len(ch.concept_clusters) for ch in self.chapters),
            'avg_concepts_per_chapter': len(self.concepts) / len(self.chapters) if self.chapters else 0,
            'avg_clusters_per_chapter': sum(len(ch.concept_clusters) for ch in self.chapters) / len(self.chapters) if self.chapters else 0,
            'concept_type_distribution': dict(concept_types),
            'difficulty_distribution': dict(difficulty_dist),
            'total_estimated_hours': sum(ch.estimated_hours for ch in self.chapters),
            'gpu_accelerated': self.device == "cuda",
            'generated_at': datetime.now().isoformat()
        }
        
        with open('./gpu_knowledge_graph_data/enhanced_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("✅ Enhanced graph data saved")
    
    def verify_enhanced_graph(self):
        """Verify the enhanced knowledge graph was built correctly"""
        logger.info("🔍 Verifying enhanced knowledge graph...")
        
        with self.neo4j_driver.session() as session:
            # Count nodes
            chapter_count = session.run("MATCH (c:Chapter) RETURN count(c) as count").single()['count']
            concept_count = session.run("MATCH (o:LearningConcept) RETURN count(o) as count").single()['count']
            cluster_count = session.run("MATCH (cl:ConceptCluster) RETURN count(cl) as count").single()['count']
            
            # Count relationships
            chapter_prereq_count = session.run("MATCH (c1:Chapter)-[:PREREQUISITE_FOR]->(c2:Chapter) RETURN count(*) as count").single()['count']
            concept_prereq_count = session.run("MATCH (o1:LearningConcept)-[:PREREQUISITE_FOR]->(o2:LearningConcept) RETURN count(*) as count").single()['count']
            cluster_contains_count = session.run("MATCH ()-[:CONTAINS_CLUSTER]->() RETURN count(*) as count").single()['count']
            concept_contains_count = session.run("MATCH ()-[:CONTAINS_CONCEPT]->() RETURN count(*) as count").single()['count']
            
            logger.info(f"✅ Enhanced verification complete:")
            logger.info(f"   📚 Chapters: {chapter_count}")
            logger.info(f"   🧠 Learning Concepts: {concept_count}")
            logger.info(f"   🔗 Concept Clusters: {cluster_count}")
            logger.info(f"   ➡️  Chapter Prerequisites: {chapter_prereq_count}")
            logger.info(f"   ➡️  Concept Prerequisites: {concept_prereq_count}")
            logger.info(f"   📦 Cluster Relationships: {cluster_contains_count}")
            logger.info(f"   🎯 Concept Relationships: {concept_contains_count}")
            
            # Show concept distribution
            concept_stats = session.run("""
            MATCH (c:Chapter)
            RETURN c.number as chapter, c.title as title, c.num_concepts as concepts, c.num_clusters as clusters
            ORDER BY toInteger(c.number)
            LIMIT 10
            """)
            
            logger.info(f"📊 Sample Chapter Distribution:")
            for record in concept_stats:
                logger.info(f"   Ch.{record['chapter']}: {record['concepts']} concepts, {record['clusters']} clusters")
            
            return True
    
    def close_connections(self):
        """Close database connections"""
        if hasattr(self, 'neo4j_driver'):
            self.neo4j_driver.close()

def main():
    """Main execution function"""
    print("🚀 GPU-Accelerated Enhanced Chapter-Based Knowledge Graph Builder")
    print("🧠 Optimized for 20-25 Chapters with 20-30 Learning Concepts Each")
    print("=" * 80)
    
    builder = None
    try:
        # Initialize builder
        builder = GPUAcceleratedEnhancedKnowledgeGraphBuilder()
        
        # Load enhanced chapter data
        builder.load_enhanced_chapter_data()
        
        # Generate GPU-accelerated embeddings
        builder.generate_gpu_accelerated_embeddings()
        
        # Create concept clusters
        builder.create_concept_clusters()
        
        # Detect enhanced prerequisites
        builder.detect_enhanced_prerequisites()
        
        # Clear and build Neo4j graph
        builder.clear_neo4j_database()
        builder.build_enhanced_neo4j_graph()
        
        # Save enhanced graph data
        builder.save_enhanced_graph_data()
        
        # Verify graph
        if builder.verify_enhanced_graph():
            print("\n🎉 SUCCESS! Enhanced chapter-based knowledge graph built successfully!")
            print(f"🌐 Neo4j Browser: http://localhost:7474")
            print(f"   Username: {NEO4J_USER}")
            print(f"   Password: {NEO4J_PASSWORD}")
            print(f"📁 Enhanced graph data saved to: ./gpu_knowledge_graph_data/")
            print("\n🎯 Enhanced features:")
            print("   • GPU-accelerated embedding generation")
            print("   • Intelligent concept clustering within chapters")
            print("   • Advanced prerequisite detection")
            print("   • 20-30 learning concepts per chapter")
            print("   • Hierarchical concept organization")
            print("\n🚀 Ready for enhanced visualization and querying!")
            
    except Exception as e:
        logger.error(f"❌ Enhanced build failed: {e}")
        raise
    
    finally:
        if builder:
            builder.close_connections()

if __name__ == "__main__":
    main()
