#!/usr/bin/env python3
"""
Enhanced Elasticsearch-Based Relation Extractor
Leverages Elasticsearch's advanced NLP capabilities for better relation extraction
"""
import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Set
import torch
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import re
from collections import defaultdict, Counter
import spacy
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElasticsearchEnhancedRelationExtractor:
    """Enhanced relation extractor using Elasticsearch's NLP capabilities"""
    
    def __init__(self, es_endpoint: str = "http://localhost:9200"):
        self.setup_gpu()
        self.setup_elasticsearch(es_endpoint)
        self.setup_nlp_models()
        self.setup_relation_patterns()
    
    def setup_gpu(self):
        """Setup GPU acceleration"""
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.empty_cache()
            logger.info(f"🚀 GPU acceleration enabled")
        else:
            self.device = "cpu"
            logger.warning("⚠️ CUDA not available, using CPU")
    
    def setup_elasticsearch(self, es_endpoint: str):
        """Setup Elasticsearch with enhanced NLP pipeline"""
        self.es_client = Elasticsearch([es_endpoint])
        
        # Create enhanced index with NLP pipeline
        self.relations_index = "enhanced_concept_relations_v1"
        self.create_enhanced_index()
    
    def create_enhanced_index(self):
        """Create Elasticsearch index with NLP pipeline for relation extraction"""
        
        # Define NLP pipeline for relation extraction
        pipeline_config = {
            "description": "Enhanced NLP pipeline for concept relation extraction",
            "processors": [
                {
                    "inference": {
                        "model_id": "sentence-transformers__all-mpnet-base-v2",
                        "target_field": "ml_inference",
                        "field_map": {
                            "text": "text_field"
                        }
                    }
                },
                {
                    "script": {
                        "source": """
                        // Extract named entities and relations
                        def text = ctx.text;
                        def relations = new ArrayList();
                        
                        // Simple relation patterns
                        def patterns = [
                            /(\w+)\s+(?:is|are|was|were)\s+(?:a|an)?\s*(\w+)/,
                            /(\w+)\s+(?:requires|needs|depends on)\s+(\w+)/,
                            /(\w+)\s+(?:includes|contains|has)\s+(\w+)/,
                            /(\w+)\s+(?:before|after|following)\s+(\w+)/
                        ];
                        
                        for (pattern in patterns) {
                            def matcher = pattern.matcher(text);
                            while (matcher.find()) {
                                relations.add([
                                    'source': matcher.group(1),
                                    'target': matcher.group(2),
                                    'relation_type': 'dependency'
                                ]);
                            }
                        }
                        
                        ctx.extracted_relations = relations;
                        """
                    }
                }
            ]
        }
        
        # Create or update pipeline
        try:
            self.es_client.ingest.put_pipeline(
                id="enhanced_nlp_pipeline",
                body=pipeline_config
            )
            logger.info("✅ Created enhanced NLP pipeline")
        except Exception as e:
            logger.warning(f"⚠️ Pipeline creation warning: {e}")
        
        # Index mapping with enhanced fields
        index_mapping = {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "concept_id": {"type": "keyword"},
                    "chapter_id": {"type": "keyword"},
                    "concept_type": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768
                    },
                    "entities": {
                        "type": "nested",
                        "properties": {
                            "text": {"type": "text"},
                            "label": {"type": "keyword"},
                            "start": {"type": "integer"},
                            "end": {"type": "integer"}
                        }
                    },
                    "relations": {
                        "type": "nested",
                        "properties": {
                            "source": {"type": "keyword"},
                            "target": {"type": "keyword"},
                            "relation_type": {"type": "keyword"},
                            "confidence": {"type": "float"},
                            "context": {"type": "text"}
                        }
                    },
                    "semantic_similarity": {
                        "type": "nested",
                        "properties": {
                            "concept_id": {"type": "keyword"},
                            "similarity_score": {"type": "float"}
                        }
                    },
                    "prerequisites": {
                        "type": "nested",
                        "properties": {
                            "concept_id": {"type": "keyword"},
                            "confidence": {"type": "float"},
                            "relation_type": {"type": "keyword"}
                        }
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "concept_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "stemmer"]
                        }
                    }
                }
            }
        }
        
        # Create index
        try:
            if self.es_client.indices.exists(index=self.relations_index):
                self.es_client.indices.delete(index=self.relations_index)
            
            self.es_client.indices.create(
                index=self.relations_index,
                body=index_mapping
            )
            logger.info(f"✅ Created enhanced relations index: {self.relations_index}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create index: {e}")
            raise
    
    def setup_nlp_models(self):
        """Setup NLP models for enhanced relation extraction"""
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('maxent_ne_chunker', quiet=True)
            nltk.download('words', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            logger.warning("⚠️ Some NLTK data may not be available")
        
        # Setup spaCy for advanced NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded spaCy model")
        except OSError:
            logger.warning("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Setup sentence transformer
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2',
            device=self.device
        )
        
        self.stopwords = set(stopwords.words('english')) if stopwords else set()
    
    def setup_relation_patterns(self):
        """Setup regex patterns for relation extraction"""
        self.relation_patterns = {
            'prerequisite': [
                r'(\w+(?:\s+\w+)*)\s+(?:requires?|needs?|depends?\s+on|builds?\s+on)\s+(\w+(?:\s+\w+)*)',
                r'(?:before|prior\s+to)\s+(\w+(?:\s+\w+)*),?\s+(?:you\s+)?(?:must|should|need\s+to)\s+(?:understand|learn|know)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+is\s+(?:a\s+)?prerequisite\s+for\s+(\w+(?:\s+\w+)*)'
            ],
            'is_a': [
                r'(\w+(?:\s+\w+)*)\s+is\s+(?:a|an)\s+(?:type\s+of\s+)?(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+are\s+(?:types?\s+of\s+)?(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:belongs?\s+to|falls?\s+under)\s+(\w+(?:\s+\w+)*)'
            ],
            'part_of': [
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:part\s+of|component\s+of|element\s+of)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:includes?|contains?|comprises?)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:consists?\s+of|made\s+up\s+of)\s+(\w+(?:\s+\w+)*)'
            ],
            'similar_to': [
                r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:similar\s+to|like|comparable\s+to)\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)\s+are\s+(?:similar|related|comparable)'
            ],
            'temporal': [
                r'(\w+(?:\s+\w+)*)\s+(?:comes?\s+)?(?:before|after|following)\s+(\w+(?:\s+\w+)*)',
                r'(?:first|then|next|finally),?\s+(\w+(?:\s+\w+)*)',
                r'(\w+(?:\s+\w+)*)\s+(?:leads?\s+to|results?\s+in|causes?)\s+(\w+(?:\s+\w+)*)'
            ]
        }
    
    def extract_entities_with_spacy(self, text: str) -> List[Dict]:
        """Extract named entities using spaCy"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 1.0  # spaCy doesn't provide confidence scores by default
            })
        
        return entities
    
    def extract_entities_with_nltk(self, text: str) -> List[Dict]:
        """Extract named entities using NLTK"""
        try:
            # Tokenize and tag
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            
            # Named entity recognition
            tree = ne_chunk(tagged)
            
            entities = []
            current_entity = []
            current_label = None
            start_pos = 0
            
            for item in tree:
                if isinstance(item, Tree):
                    # This is a named entity
                    entity_text = ' '.join([token for token, pos in item.leaves()])
                    entities.append({
                        'text': entity_text,
                        'label': item.label(),
                        'start': text.find(entity_text, start_pos),
                        'end': text.find(entity_text, start_pos) + len(entity_text),
                        'confidence': 0.8  # Default confidence for NLTK
                    })
                    start_pos = text.find(entity_text, start_pos) + len(entity_text)
            
            return entities
            
        except Exception as e:
            logger.warning(f"⚠️ NLTK entity extraction failed: {e}")
            return []
    
    def extract_relations_with_patterns(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relations using regex patterns"""
        relations = []
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    source = match.group(1).strip()
                    target = match.group(2).strip() if match.lastindex >= 2 else None
                    
                    if target:
                        # Calculate confidence based on pattern quality and context
                        context = text[max(0, match.start()-50):match.end()+50]
                        confidence = self.calculate_relation_confidence(source, target, context, relation_type)
                        
                        relations.append({
                            'source': source,
                            'target': target,
                            'relation_type': relation_type,
                            'confidence': confidence,
                            'context': context.strip(),
                            'pattern_match': match.group(0)
                        })
        
        return relations
    
    def calculate_relation_confidence(self, source: str, target: str, context: str, relation_type: str) -> float:
        """Calculate confidence score for extracted relations"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on various factors
        if len(source.split()) <= 3 and len(target.split()) <= 3:  # Shorter concepts are more likely to be accurate
            confidence += 0.1
        
        if source.lower() not in self.stopwords and target.lower() not in self.stopwords:
            confidence += 0.1
        
        # Check if concepts appear in close proximity (good indicator)
        if abs(context.lower().find(source.lower()) - context.lower().find(target.lower())) < 100:
            confidence += 0.1
        
        # Relation type specific boosts
        if relation_type == 'prerequisite' and any(word in context.lower() 
            for word in ['before', 'first', 'prerequisite', 'required', 'must']):
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def extract_semantic_similarities_with_elasticsearch(self, concept_text: str, concept_embedding: np.ndarray, 
                                                       top_k: int = 10) -> List[Dict]:
        """Use Elasticsearch vector similarity to find related concepts"""
        try:
            # Search for similar concepts using vector similarity
            search_body = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": concept_embedding.tolist()}
                        }
                    }
                },
                "size": top_k,
                "_source": ["concept_id", "text", "concept_type"]
            }
            
            response = self.es_client.search(
                index=self.relations_index,
                body=search_body
            )
            
            similarities = []
            for hit in response['hits']['hits']:
                if hit['_score'] > 1.7:  # Threshold for meaningful similarity
                    similarities.append({
                        'concept_id': hit['_source']['concept_id'],
                        'similarity_score': hit['_score'] - 1.0,  # Normalize back to [0,1]
                        'text': hit['_source']['text'],
                        'concept_type': hit['_source'].get('concept_type', 'unknown')
                    })
            
            return similarities
            
        except Exception as e:
            logger.warning(f"⚠️ Elasticsearch similarity search failed: {e}")
            return []
    
    def extract_comprehensive_relations(self, concept_data: Dict) -> Dict:
        """Extract comprehensive relations for a concept using all methods"""
        text = concept_data['text']
        concept_id = concept_data['id']
        
        logger.info(f"🔍 Extracting relations for concept: {concept_id}")
        
        # 1. Entity extraction
        spacy_entities = self.extract_entities_with_spacy(text)
        nltk_entities = self.extract_entities_with_nltk(text)
        
        # Combine and deduplicate entities
        all_entities = spacy_entities + nltk_entities
        unique_entities = []
        seen_texts = set()
        
        for entity in all_entities:
            if entity['text'].lower() not in seen_texts:
                unique_entities.append(entity)
                seen_texts.add(entity['text'].lower())
        
        # 2. Pattern-based relation extraction
        pattern_relations = self.extract_relations_with_patterns(text, unique_entities)
        
        # 3. Generate embedding for semantic similarity
        embedding = self.embedding_model.encode([text], device=self.device)[0]
        
        # 4. Elasticsearch-based semantic similarity
        semantic_similarities = self.extract_semantic_similarities_with_elasticsearch(
            text, embedding, top_k=10
        )
        
        # 5. Combine all relation types
        comprehensive_relations = {
            'concept_id': concept_id,
            'text': text,
            'chapter_id': concept_data.get('chapter_id'),
            'concept_type': concept_data.get('type'),
            'embedding': embedding.tolist(),
            'entities': unique_entities,
            'pattern_relations': pattern_relations,
            'semantic_similarities': semantic_similarities,
            'extracted_prerequisites': self.identify_prerequisites(pattern_relations, semantic_similarities),
            'relation_summary': self.create_relation_summary(pattern_relations, semantic_similarities)
        }
        
        return comprehensive_relations
    
    def identify_prerequisites(self, pattern_relations: List[Dict], 
                             semantic_similarities: List[Dict]) -> List[Dict]:
        """Identify prerequisite concepts from extracted relations"""
        prerequisites = []
        
        # From pattern relations
        for relation in pattern_relations:
            if relation['relation_type'] in ['prerequisite', 'temporal']:
                prerequisites.append({
                    'concept_id': relation['target'],  # Target is the prerequisite
                    'confidence': relation['confidence'],
                    'relation_type': 'prerequisite_pattern',
                    'evidence': relation['context']
                })
        
        # From semantic similarities (concepts with high similarity might be prerequisites)
        for similarity in semantic_similarities:
            if similarity['similarity_score'] > 0.8:  # High similarity threshold
                prerequisites.append({
                    'concept_id': similarity['concept_id'],
                    'confidence': similarity['similarity_score'] * 0.7,  # Lower confidence for similarity-based
                    'relation_type': 'prerequisite_semantic',
                    'evidence': f"High semantic similarity: {similarity['similarity_score']:.3f}"
                })
        
        # Sort by confidence
        prerequisites.sort(key=lambda x: x['confidence'], reverse=True)
        
        return prerequisites[:5]  # Top 5 prerequisites
    
    def create_relation_summary(self, pattern_relations: List[Dict], 
                              semantic_similarities: List[Dict]) -> Dict:
        """Create a summary of all extracted relations"""
        summary = {
            'total_relations': len(pattern_relations),
            'relation_types': Counter([r['relation_type'] for r in pattern_relations]),
            'high_confidence_relations': len([r for r in pattern_relations if r['confidence'] > 0.7]),
            'semantic_similarities_count': len(semantic_similarities),
            'strong_similarities': len([s for s in semantic_similarities if s['similarity_score'] > 0.8])
        }
        
        return summary
    
    def bulk_index_enhanced_relations(self, concepts_data: List[Dict]):
        """Bulk index concepts with enhanced relation extraction"""
        logger.info(f"🚀 Processing {len(concepts_data)} concepts for enhanced relation extraction...")
        
        enhanced_docs = []
        
        for i, concept in enumerate(concepts_data):
            try:
                # Extract comprehensive relations
                enhanced_concept = self.extract_comprehensive_relations(concept)
                
                # Prepare for Elasticsearch indexing
                doc = {
                    '_index': self.relations_index,
                    '_id': enhanced_concept['concept_id'],
                    '_source': enhanced_concept
                }
                
                enhanced_docs.append(doc)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"📊 Processed {i + 1}/{len(concepts_data)} concepts")
                    
            except Exception as e:
                logger.error(f"❌ Failed to process concept {concept.get('id', 'unknown')}: {e}")
                continue
        
        # Bulk index
        if enhanced_docs:
            try:
                success, failed = bulk(self.es_client, enhanced_docs, chunk_size=100)
                logger.info(f"✅ Successfully indexed {success} enhanced concepts")
                if failed:
                    logger.warning(f"⚠️ Failed to index {len(failed)} concepts")
                    
            except Exception as e:
                logger.error(f"❌ Bulk indexing failed: {e}")
                raise
    
    def query_enhanced_relations(self, query: str, relation_types: List[str] = None, 
                               top_k: int = 10) -> Dict:
        """Query the enhanced relations index"""
        
        # Build query
        if relation_types:
            query_body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["text^2", "entities.text", "pattern_relations.source", "pattern_relations.target"]
                                }
                            },
                            {
                                "nested": {
                                    "path": "pattern_relations",
                                    "query": {
                                        "bool": {
                                            "must": [
                                                {"terms": {"pattern_relations.relation_type": relation_types}},
                                                {"multi_match": {"query": query, "fields": ["pattern_relations.source", "pattern_relations.target"]}}
                                            ]
                                        }
                                    }
                                }
                            }
                        ]
                    }
                },
                "size": top_k,
                "highlight": {
                    "fields": {
                        "text": {},
                        "pattern_relations.context": {}
                    }
                }
            }
        else:
            query_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text^2", "entities.text", "pattern_relations.source", "pattern_relations.target"]
                    }
                },
                "size": top_k,
                "highlight": {
                    "fields": {
                        "text": {},
                        "pattern_relations.context": {}
                    }
                }
            }
        
        try:
            response = self.es_client.search(
                index=self.relations_index,
                body=query_body
            )
            
            results = {
                'query': query,
                'total_hits': response['hits']['total']['value'],
                'concepts': [],
                'relation_aggregations': self.get_relation_aggregations(query)
            }
            
            for hit in response['hits']['hits']:
                concept_result = {
                    'concept_id': hit['_source']['concept_id'],
                    'text': hit['_source']['text'],
                    'score': hit['_score'],
                    'relations': hit['_source'].get('pattern_relations', []),
                    'entities': hit['_source'].get('entities', []),
                    'prerequisites': hit['_source'].get('extracted_prerequisites', []),
                    'highlights': hit.get('highlight', {})
                }
                
                results['concepts'].append(concept_result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Enhanced query failed: {e}")
            return {'query': query, 'error': str(e), 'concepts': []}
    
    def get_relation_aggregations(self, query: str) -> Dict:
        """Get aggregations for relation types and patterns"""
        agg_query = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text", "pattern_relations.source", "pattern_relations.target"]
                }
            },
            "size": 0,
            "aggs": {
                "relation_types": {
                    "nested": {"path": "pattern_relations"},
                    "aggs": {
                        "types": {
                            "terms": {"field": "pattern_relations.relation_type", "size": 10}
                        }
                    }
                },
                "entity_types": {
                    "nested": {"path": "entities"},
                    "aggs": {
                        "types": {
                            "terms": {"field": "entities.label", "size": 10}
                        }
                    }
                }
            }
        }
        
        try:
            response = self.es_client.search(
                index=self.relations_index,
                body=agg_query
            )
            
            return {
                'relation_types': response['aggregations']['relation_types']['types']['buckets'],
                'entity_types': response['aggregations']['entity_types']['types']['buckets']
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Aggregation query failed: {e}")
            return {}

def main():
    """Test the enhanced Elasticsearch relation extractor"""
    extractor = ElasticsearchEnhancedRelationExtractor()
    
    # Test with sample concepts
    sample_concepts = [
        {
            'id': 'concept_1',
            'text': 'Machine learning requires understanding of statistics and linear algebra. Before you can implement neural networks, you must learn about gradient descent.',
            'chapter_id': 'chapter_1',
            'type': 'technical_concept'
        },
        {
            'id': 'concept_2', 
            'text': 'Neural networks are a type of machine learning algorithm. They consist of layers of interconnected nodes that process information.',
            'chapter_id': 'chapter_2',
            'type': 'technical_concept'
        }
    ]
    
    # Extract and index relations
    extractor.bulk_index_enhanced_relations(sample_concepts)
    
    # Test queries
    results = extractor.query_enhanced_relations("machine learning prerequisites")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
