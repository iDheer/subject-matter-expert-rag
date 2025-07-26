#!/usr/bin/env python3
"""
GPU-Accelerated Chapter-Based Database Builder - Enhanced Version
Creates a chapter-focused Elasticsearch database with 20-30 learning concepts per chapter
"""
import os
import re
import json
import logging
from datetime import datetime
from collections import Counter
from elasticsearch import Elasticsearch
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document
)
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
from typing import List, Dict, Optional, Set
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
ES_ENDPOINT = "http://localhost:9200"
INDEX_NAME = "gpu_chapter_knowledge_v1"
ES_STORAGE_DIR = "./gpu_chapter_elasticsearch_storage"
DATA_PATH = "./data_large"

class EnhancedChapterExtractor:
    """Enhanced chapter extractor with better learning concept identification"""
    
    def __init__(self):
        # Patterns to identify chapters and sections
        self.chapter_patterns = [
            r'^(?:chapter|ch\.?)\s*(\d+)[:\s]*(.+?)$',  # Chapter 1: Introduction
            r'^(\d+)\.\s*([^.]+)$',                      # 1. Introduction  
            r'^([IVX]+)\.\s*(.+)$',                      # I. Introduction (Roman numerals)
            r'^part\s*(\d+)[:\s]*(.+?)$',               # Part 1: Basics
            r'^unit\s*(\d+)[:\s]*(.+?)$',               # Unit 1: Fundamentals
            r'^lesson\s*(\d+)[:\s]*(.+?)$',             # Lesson 1: Basics
        ]
        
        self.section_patterns = [
            r'^(\d+\.\d+)\s*(.+)$',                     # 1.1 Section Name
            r'^(\d+\.\d+\.\d+)\s*(.+)$',               # 1.1.1 Subsection
            r'^([A-Z])\)\s*(.+)$',                      # A) Section Name
            r'^([a-z])\)\s*(.+)$',                      # a) Subsection Name
            r'^[•\-]\s*(.+)$',                          # • Bullet point or - Dash point
            r'^\*\s*(.+)$',                             # * Asterisk point
        ]
        
        # Keywords that indicate learning objectives/concepts
        self.learning_indicators = [
            'understand', 'learn', 'know', 'explain', 'describe', 'analyze',
            'evaluate', 'implement', 'design', 'compare', 'contrast', 'apply',
            'demonstrate', 'identify', 'classify', 'solve', 'create', 'develop',
            'objective', 'goal', 'outcome', 'concept', 'principle', 'theory',
            'method', 'technique', 'approach', 'strategy', 'algorithm',
            'by the end', 'after reading', 'you will', 'students will', 'able to',
            'key concept', 'important', 'fundamental', 'essential', 'critical'
        ]
        
        # Technical terms that often represent key concepts
        self.technical_indicators = [
            'definition', 'theorem', 'lemma', 'proof', 'formula', 'equation',
            'model', 'framework', 'architecture', 'structure', 'pattern',
            'protocol', 'specification', 'standard', 'interface', 'api',
            'system', 'process', 'mechanism', 'procedure', 'workflow'
        ]
        
        # Get English stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    def extract_chapters_from_text(self, text: str, document_name: str) -> List[Dict]:
        """Extract chapters and their content from text"""
        lines = text.split('\n')
        chapters = []
        current_chapter = None
        current_content = []
        
        chapter_counter = 0
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this line is a chapter heading
            chapter_match = None
            for pattern in self.chapter_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    chapter_match = match
                    break
            
            if chapter_match:
                # Save previous chapter if exists
                if current_chapter:
                    current_chapter['content'] = '\n'.join(current_content)
                    current_chapter['learning_concepts'] = self.extract_learning_concepts('\n'.join(current_content))
                    chapters.append(current_chapter)
                
                # Start new chapter
                chapter_counter += 1
                chapter_num = chapter_match.group(1) if len(chapter_match.groups()) > 1 else str(chapter_counter)
                chapter_title = chapter_match.group(2) if len(chapter_match.groups()) > 1 else chapter_match.group(1)
                
                current_chapter = {
                    'id': f"{document_name}_chapter_{chapter_counter}",
                    'number': chapter_num,
                    'title': chapter_title.strip(),
                    'document_source': document_name,
                    'start_line': line_num,
                    'type': 'chapter'
                }
                current_content = [line]
            else:
                # Add content to current chapter
                if current_chapter:
                    current_content.append(line)
        
        # Don't forget the last chapter
        if current_chapter:
            current_chapter['content'] = '\n'.join(current_content)
            current_chapter['learning_concepts'] = self.extract_learning_concepts('\n'.join(current_content))
            chapters.append(current_chapter)
        
        # If no chapters found, create chunks based on content length
        if not chapters:
            chapters = self.create_content_based_chapters(text, document_name)
        
        logger.info(f"Extracted {len(chapters)} chapters from {document_name}")
        return chapters
    
    def create_content_based_chapters(self, text: str, document_name: str) -> List[Dict]:
        """Create chapters based on content length when no clear chapter structure exists"""
        # Split text into reasonable chunks (around 5000-8000 characters per chapter)
        chunk_size = 6000
        overlap = 500
        
        chapters = []
        text_length = len(text)
        start_pos = 0
        chapter_counter = 0
        
        while start_pos < text_length:
            chapter_counter += 1
            end_pos = min(start_pos + chunk_size, text_length)
            
            # Try to end at a paragraph boundary
            if end_pos < text_length:
                # Look for paragraph break within last 500 chars
                search_start = max(end_pos - 500, start_pos)
                paragraph_break = text.rfind('\n\n', search_start, end_pos)
                if paragraph_break > search_start:
                    end_pos = paragraph_break
            
            chapter_content = text[start_pos:end_pos].strip()
            
            # Create chapter title from first line or generate one
            first_line = chapter_content.split('\n')[0].strip()[:50]
            if len(first_line) < 10:
                chapter_title = f"Section {chapter_counter}"
            else:
                chapter_title = first_line + "..." if len(first_line) == 50 else first_line
            
            chapters.append({
                'id': f"{document_name}_chapter_{chapter_counter}",
                'number': str(chapter_counter),
                'title': chapter_title,
                'document_source': document_name,
                'start_line': 0,  # Approximate
                'type': 'chapter',
                'content': chapter_content,
                'learning_concepts': self.extract_learning_concepts(chapter_content)
            })
            
            start_pos = end_pos - overlap if end_pos < text_length else text_length
        
        return chapters
    
    def extract_learning_concepts(self, content: str) -> List[Dict]:
        """Extract 20-30 learning concepts from chapter content using advanced NLP"""
        concepts = []
        
        # Method 1: Explicit learning objectives
        explicit_concepts = self.find_explicit_objectives(content)
        concepts.extend(explicit_concepts)
        
        # Method 2: Key terms and definitions
        key_terms = self.extract_key_terms(content)
        concepts.extend(key_terms)
        
        # Method 3: Section headings as concepts
        section_concepts = self.extract_section_concepts(content)
        concepts.extend(section_concepts)
        
        # Method 4: Technical concepts from sentences
        technical_concepts = self.extract_technical_concepts(content)
        concepts.extend(technical_concepts)
        
        # Method 5: Question-based concepts
        question_concepts = self.extract_question_concepts(content)
        concepts.extend(question_concepts)
        
        # Remove duplicates and rank by importance
        concepts = self.deduplicate_and_rank_concepts(concepts)
        
        # Ensure we have 20-30 concepts, pad if necessary
        if len(concepts) < 20:
            concepts.extend(self.generate_supplementary_concepts(content, 20 - len(concepts)))
        
        # Limit to top 30 concepts
        concepts = concepts[:30]
        
        logger.info(f"Extracted {len(concepts)} learning concepts")
        return concepts
    
    def find_explicit_objectives(self, content: str) -> List[Dict]:
        """Find explicitly stated learning objectives"""
        objectives = []
        lines = content.split('\n')
        
        obj_counter = 0
        for line in lines:
            line_clean = line.strip().lower()
            if not line_clean:
                continue
            
            # Check if line contains learning indicators
            if any(indicator in line_clean for indicator in self.learning_indicators):
                obj_counter += 1
                # Clean up the objective text
                clean_objective = re.sub(r'^[•\-\d\.\)\*]*\s*', '', line.strip())
                clean_objective = re.sub(r'^\w+\)\s*', '', clean_objective)  # Remove A), B), etc.
                
                if len(clean_objective) > 10:  # Ensure it's substantial
                    objectives.append({
                        'id': f"explicit_obj_{obj_counter}",
                        'text': clean_objective,
                        'type': 'explicit_objective',
                        'difficulty': self.estimate_difficulty(clean_objective),
                        'keywords': self.extract_concept_keywords(clean_objective),
                        'confidence': 0.9  # High confidence for explicit objectives
                    })
        
        return objectives[:10]  # Limit explicit objectives
    
    def extract_key_terms(self, content: str) -> List[Dict]:
        """Extract key terms and definitions"""
        key_terms = []
        
        # Look for definition patterns
        definition_patterns = [
            r'(\w+(?:\s+\w+)*)\s+is\s+defined\s+as\s+(.+)',
            r'(\w+(?:\s+\w+)*)\s+refers\s+to\s+(.+)',
            r'(\w+(?:\s+\w+)*)\s+means\s+(.+)',
            r'definition:\s*(\w+(?:\s+\w+)*)\s+(.+)',
            r'(\w+(?:\s+\w+)*)\s*:\s*(.+)',  # Term: definition
        ]
        
        term_counter = 0
        for pattern in definition_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                term_counter += 1
                term = match.group(1).strip()
                definition = match.group(2).strip()[:200]  # Limit definition length
                
                if len(term.split()) <= 4 and len(definition) > 20:  # Reasonable term length
                    key_terms.append({
                        'id': f"key_term_{term_counter}",
                        'text': f"understand {term}: {definition}",
                        'type': 'key_term',
                        'difficulty': self.estimate_difficulty(definition),
                        'keywords': [term.lower()] + self.extract_concept_keywords(definition),
                        'confidence': 0.8
                    })
        
        return key_terms[:8]  # Limit key terms
    
    def extract_section_concepts(self, content: str) -> List[Dict]:
        """Extract concepts from section headings"""
        section_concepts = []
        lines = content.split('\n')
        
        section_counter = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this looks like a section heading
            for pattern in self.section_patterns:
                match = re.match(pattern, line)
                if match:
                    section_counter += 1
                    if len(match.groups()) > 1:
                        section_title = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    else:
                        section_title = match.group(1)
                    
                    # Skip very short or very long headings
                    if 5 <= len(section_title) <= 100:
                        concept_text = f"understand {section_title}".lower()
                        
                        section_concepts.append({
                            'id': f"section_concept_{section_counter}",
                            'text': concept_text,
                            'type': 'section_concept',
                            'difficulty': self.estimate_difficulty(section_title),
                            'keywords': self.extract_concept_keywords(section_title),
                            'confidence': 0.7
                        })
                    break
        
        return section_concepts[:6]  # Limit section concepts
    
    def extract_technical_concepts(self, content: str) -> List[Dict]:
        """Extract technical concepts using NLP"""
        technical_concepts = []
        
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(content)
            
            concept_counter = 0
            for sentence in sentences:
                # Skip very short or very long sentences
                if not (20 <= len(sentence) <= 300):
                    continue
                
                # Check for technical indicators
                sentence_lower = sentence.lower()
                if any(indicator in sentence_lower for indicator in self.technical_indicators):
                    concept_counter += 1
                    
                    # Extract the main concept from the sentence
                    concept_text = self.extract_main_concept_from_sentence(sentence)
                    
                    if concept_text and len(concept_text) > 15:
                        technical_concepts.append({
                            'id': f"tech_concept_{concept_counter}",
                            'text': concept_text,
                            'type': 'technical_concept',
                            'difficulty': self.estimate_difficulty(sentence),
                            'keywords': self.extract_concept_keywords(concept_text),
                            'confidence': 0.6
                        })
        
        except Exception as e:
            logger.warning(f"NLP processing failed: {e}")
        
        return technical_concepts[:8]  # Limit technical concepts
    
    def extract_question_concepts(self, content: str) -> List[Dict]:
        """Extract concepts from questions in the text"""
        question_concepts = []
        
        # Find questions
        question_patterns = [
            r'what\s+is\s+([^?]+)\?',
            r'how\s+does\s+([^?]+)\?',
            r'why\s+is\s+([^?]+)\?',
            r'when\s+should\s+([^?]+)\?',
            r'where\s+is\s+([^?]+)\?',
        ]
        
        question_counter = 0
        for pattern in question_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                question_counter += 1
                question_part = match.group(1).strip()
                
                if len(question_part) > 10:
                    concept_text = f"understand {question_part}"
                    
                    question_concepts.append({
                        'id': f"question_concept_{question_counter}",
                        'text': concept_text,
                        'type': 'question_concept',
                        'difficulty': self.estimate_difficulty(question_part),
                        'keywords': self.extract_concept_keywords(question_part),
                        'confidence': 0.5
                    })
        
        return question_concepts[:4]  # Limit question concepts
    
    def extract_main_concept_from_sentence(self, sentence: str) -> str:
        """Extract the main concept from a sentence using NLP"""
        try:
            # Tokenize and POS tag
            tokens = word_tokenize(sentence.lower())
            pos_tags = pos_tag(tokens)
            
            # Extract nouns and noun phrases that might be concepts
            concept_words = []
            for word, pos in pos_tags:
                if pos.startswith('NN') and word not in self.stop_words and len(word) > 3:
                    concept_words.append(word)
            
            if concept_words:
                # Create concept text
                main_concepts = concept_words[:3]  # Take first 3 nouns
                return f"understand {' and '.join(main_concepts)}"
            
        except Exception:
            pass
        
        # Fallback: return first 50 characters
        return sentence[:50].strip()
    
    def deduplicate_and_rank_concepts(self, concepts: List[Dict]) -> List[Dict]:
        """Remove duplicates and rank concepts by importance"""
        # Remove duplicates based on similar text
        unique_concepts = []
        seen_texts = set()
        
        for concept in concepts:
            # Normalize text for comparison
            normalized_text = re.sub(r'[^\w\s]', '', concept['text'].lower())
            normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
            
            # Check for similar existing concepts
            is_duplicate = False
            for seen_text in seen_texts:
                # Simple similarity check
                if self.text_similarity(normalized_text, seen_text) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_concepts.append(concept)
                seen_texts.add(normalized_text)
        
        # Sort by confidence and type priority
        type_priority = {
            'explicit_objective': 5,
            'key_term': 4,
            'technical_concept': 3,
            'section_concept': 2,
            'question_concept': 1
        }
        
        unique_concepts.sort(
            key=lambda x: (type_priority.get(x['type'], 0), x['confidence']),
            reverse=True
        )
        
        return unique_concepts
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def generate_supplementary_concepts(self, content: str, num_needed: int) -> List[Dict]:
        """Generate supplementary concepts to reach target count"""
        supplementary = []
        
        try:
            # Extract frequent noun phrases
            sentences = sent_tokenize(content)
            noun_phrases = []
            
            for sentence in sentences[:20]:  # Limit processing
                tokens = word_tokenize(sentence.lower())
                pos_tags = pos_tag(tokens)
                
                # Find noun phrases
                current_phrase = []
                for word, pos in pos_tags:
                    if pos.startswith('NN') or pos.startswith('JJ'):
                        current_phrase.append(word)
                    else:
                        if len(current_phrase) >= 2:
                            phrase = ' '.join(current_phrase)
                            if len(phrase) > 6:
                                noun_phrases.append(phrase)
                        current_phrase = []
            
            # Get most common phrases
            phrase_counts = Counter(noun_phrases)
            common_phrases = phrase_counts.most_common(num_needed)
            
            for i, (phrase, count) in enumerate(common_phrases):
                supplementary.append({
                    'id': f"supplementary_{i+1}",
                    'text': f"understand {phrase}",
                    'type': 'supplementary_concept',
                    'difficulty': 'intermediate',
                    'keywords': phrase.split(),
                    'confidence': 0.3
                })
        
        except Exception as e:
            logger.warning(f"Supplementary concept generation failed: {e}")
        
        return supplementary
    
    def estimate_difficulty(self, text: str) -> str:
        """Estimate difficulty level based on text complexity"""
        text_lower = text.lower()
        
        # Advanced keywords
        advanced_keywords = [
            'complex', 'sophisticated', 'intricate', 'algorithm', 'implementation',
            'optimization', 'architecture', 'framework', 'paradigm', 'methodology',
            'theoretical', 'mathematical', 'statistical', 'computational'
        ]
        
        # Intermediate keywords
        intermediate_keywords = [
            'analyze', 'evaluate', 'compare', 'design', 'develop', 'apply',
            'demonstrate', 'implement', 'construct', 'synthesize', 'integrate'
        ]
        
        # Count advanced and intermediate indicators
        advanced_count = sum(1 for keyword in advanced_keywords if keyword in text_lower)
        intermediate_count = sum(1 for keyword in intermediate_keywords if keyword in text_lower)
        
        # Consider text length and complexity
        word_count = len(text.split())
        sentence_complexity = text.count(',') + text.count(';') + text.count('(')
        
        if advanced_count >= 2 or (word_count > 30 and sentence_complexity > 5):
            return 'advanced'
        elif intermediate_count >= 1 or (word_count > 15 and sentence_complexity > 2):
            return 'intermediate'
        else:
            return 'beginner'
    
    def extract_concept_keywords(self, text: str) -> List[str]:
        """Extract keywords from concept text"""
        try:
            # Tokenize and get POS tags
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # Extract meaningful words (nouns, adjectives, verbs)
            keywords = []
            for word, pos in pos_tags:
                if (pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('VB')) and \
                   word not in self.stop_words and len(word) > 3:
                    keywords.append(word)
            
            # Also extract technical terms and capitalized words from original text
            tech_terms = re.findall(r'[A-Z][a-zA-Z]{2,}', text)
            keywords.extend([term.lower() for term in tech_terms])
            
            # Remove duplicates and limit
            return list(set(keywords))[:8]
            
        except Exception:
            # Fallback: simple word extraction
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            return [word for word in words if word not in self.stop_words][:8]

class GPUAcceleratedChapterBuilder:
    """GPU-accelerated chapter-based database builder with enhanced concept extraction"""
    
    def __init__(self):
        self.setup_gpu()
        self.setup_models()
        self.setup_elasticsearch()
        self.chapter_extractor = EnhancedChapterExtractor()
    
    def setup_gpu(self):
        """Setup GPU acceleration"""
        if torch.cuda.is_available():
            self.device = "cuda"
            # Set CUDA memory management
            torch.cuda.empty_cache()
            logger.info(f"🚀 GPU acceleration enabled - Using {torch.cuda.get_device_name()}")
            logger.info(f"   CUDA version: {torch.version.cuda}")
            logger.info(f"   Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = "cpu"
            logger.warning("⚠️ CUDA not available, using CPU")
    
    def setup_models(self):
        """Setup LLM and embedding models with GPU acceleration"""
        logger.info("🔧 Setting up models...")
        
        # Setup LLM
        Settings.llm = Ollama(model="qwen3:4b", request_timeout=300.0)
        
        # Setup GPU-accelerated embeddings
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-mpnet-base-v2",
            device=self.device,
            # GPU optimization settings
            
            
        )
        
        logger.info(f"✅ Models configured for {self.device}")
    
    def setup_elasticsearch(self):
        """Setup Elasticsearch connection"""
        logger.info(f"🔍 Connecting to Elasticsearch at {ES_ENDPOINT}")
        
        try:
            self.es_client = Elasticsearch([ES_ENDPOINT])
            es_info = self.es_client.info()
            logger.info(f"✅ Connected to Elasticsearch {es_info.body['version']['number']}")
            
            # Delete existing index if it exists
            if self.es_client.indices.exists(index=INDEX_NAME):
                self.es_client.indices.delete(index=INDEX_NAME)
                logger.info(f"🗑️ Deleted existing index: {INDEX_NAME}")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to Elasticsearch: {e}")
            raise
    
    def load_documents(self) -> List[Document]:
        """Load and process documents into chapters with enhanced concept extraction"""
        logger.info(f"📚 Loading documents from '{DATA_PATH}'")
        
        if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
            os.makedirs(DATA_PATH, exist_ok=True)
            logger.error(f"❌ No documents found in '{DATA_PATH}'. Please add documents and run again.")
            raise FileNotFoundError(f"No documents in {DATA_PATH}")
        
        # Load original documents
        documents = SimpleDirectoryReader(DATA_PATH).load_data()
        logger.info(f"📖 Loaded {len(documents)} document(s)")
        
        # Extract chapters from each document
        chapter_documents = []
        all_chapters_data = []
        
        for doc in documents:
            # Get document name
            doc_name = os.path.basename(doc.metadata.get('file_name', 'unknown'))
            logger.info(f"🔍 Extracting chapters from: {doc_name}")
            
            # Extract chapters with enhanced concept extraction
            chapters = self.chapter_extractor.extract_chapters_from_text(doc.text, doc_name)
            
            for chapter in chapters:
                # Create LlamaIndex document for each chapter
                chapter_doc = Document(
                    text=chapter['content'],
                    metadata={
                        'chapter_id': chapter['id'],
                        'chapter_number': chapter['number'],
                        'chapter_title': chapter['title'],
                        'document_source': chapter['document_source'],
                        'type': 'chapter',
                        'learning_concepts': json.dumps(chapter['learning_concepts']),
                        'num_concepts': len(chapter['learning_concepts']),
                        'concept_types': json.dumps(list(set(c['type'] for c in chapter['learning_concepts'])))
                    }
                )
                chapter_documents.append(chapter_doc)
                all_chapters_data.append(chapter)
        
        logger.info(f"🎯 Created {len(chapter_documents)} chapter documents")
        
        # Calculate and log statistics
        total_concepts = sum(len(ch['learning_concepts']) for ch in all_chapters_data)
        avg_concepts = total_concepts / len(all_chapters_data) if all_chapters_data else 0
        
        concept_types = {}
        for chapter in all_chapters_data:
            for concept in chapter['learning_concepts']:
                concept_type = concept['type']
                concept_types[concept_type] = concept_types.get(concept_type, 0) + 1
        
        logger.info(f"📊 Concept Statistics:")
        logger.info(f"   Total concepts: {total_concepts}")
        logger.info(f"   Average per chapter: {avg_concepts:.1f}")
        logger.info(f"   Concept types: {dict(concept_types)}")
        
        # Save chapter data for knowledge graph
        os.makedirs('./gpu_chapter_data', exist_ok=True)
        with open('./gpu_chapter_data/chapters_metadata.json', 'w') as f:
            json.dump(all_chapters_data, f, indent=2)
        
        # Save statistics
        stats = {
            'total_chapters': len(all_chapters_data),
            'total_concepts': total_concepts,
            'avg_concepts_per_chapter': avg_concepts,
            'concept_types_distribution': concept_types,
            'generated_at': datetime.now().isoformat()
        }
        
        with open('./gpu_chapter_data/extraction_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        return chapter_documents
    
    def build_index(self, documents: List[Document]) -> VectorStoreIndex:
        """Build GPU-accelerated vector index"""
        logger.info("🏗️ Building GPU-accelerated vector index...")
        
        # Setup vector store
        vector_store = ElasticsearchStore(
            index_name=INDEX_NAME,
            es_url=ES_ENDPOINT,
            vector_field="embedding",
            text_field="content"
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Use simple node parser for chapters (no hierarchical splitting needed)
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=6144,  # Larger chunks for chapters
            chunk_overlap=300
        )
        
        # Parse documents into nodes
        nodes = node_parser.get_nodes_from_documents(documents)
        logger.info(f"📝 Created {len(nodes)} chapter nodes")
        
        # Build index with GPU acceleration
        logger.info("🚀 Building index with GPU acceleration...")
        if self.device == "cuda":
            with torch.cuda.device(0):
                vector_index = VectorStoreIndex(
                    nodes,
                    storage_context=storage_context,
                    show_progress=True,
                )
        else:
            vector_index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                show_progress=True,
            )
        
        # Persist storage
        logger.info(f"💾 Persisting storage to '{ES_STORAGE_DIR}'")
        if os.path.exists(ES_STORAGE_DIR):
            import shutil
            shutil.rmtree(ES_STORAGE_DIR)
        
        storage_context.persist(persist_dir=ES_STORAGE_DIR)
        
        return vector_index
    
    def verify_build(self):
        """Verify the build was successful"""
        logger.info("🔍 Verifying build...")
        
        try:
            # Check Elasticsearch
            doc_count = self.es_client.count(index=INDEX_NAME)['count']
            logger.info(f"✅ Elasticsearch index '{INDEX_NAME}': {doc_count} documents")
            
            # Check storage
            if os.path.exists(ES_STORAGE_DIR):
                logger.info(f"✅ Storage directory created: {ES_STORAGE_DIR}")
            
            # Check chapter metadata
            if os.path.exists('./gpu_chapter_data/chapters_metadata.json'):
                with open('./gpu_chapter_data/chapters_metadata.json', 'r') as f:
                    chapters = json.load(f)
                logger.info(f"✅ Chapter metadata: {len(chapters)} chapters")
                
                # Show detailed summary
                total_concepts = sum(len(ch['learning_concepts']) for ch in chapters)
                avg_concepts = total_concepts / len(chapters) if chapters else 0
                
                logger.info(f"📊 Enhanced Summary:")
                logger.info(f"   Chapters: {len(chapters)}")
                logger.info(f"   Total Learning Concepts: {total_concepts}")
                logger.info(f"   Average Concepts per Chapter: {avg_concepts:.1f}")
                
                # Show concept type distribution
                concept_types = {}
                for chapter in chapters:
                    for concept in chapter['learning_concepts']:
                        concept_type = concept['type']
                        concept_types[concept_type] = concept_types.get(concept_type, 0) + 1
                
                logger.info(f"   Concept Types: {dict(concept_types)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False

def main():
    """Main execution function"""
    print("🚀 GPU-Accelerated Enhanced Chapter-Based Database Builder")
    print("🧠 With Advanced Learning Concept Extraction (20-30 concepts per chapter)")
    print("=" * 80)
    
    try:
        # Initialize builder
        builder = GPUAcceleratedChapterBuilder()
        
        # Load documents and extract chapters
        documents = builder.load_documents()
        
        # Build index
        index = builder.build_index(documents)
        
        # Verify build
        if builder.verify_build():
            print("\n🎉 SUCCESS! Enhanced chapter-based database built successfully!")
            print(f"📁 Index: {INDEX_NAME}")
            print(f"📁 Storage: {ES_STORAGE_DIR}")
            print(f"📁 Chapter metadata: ./gpu_chapter_data/chapters_metadata.json")
            print(f"📁 Statistics: ./gpu_chapter_data/extraction_stats.json")
            print("\n🎯 Ready for knowledge graph construction!")
            
        else:
            print("\n❌ Build verification failed!")
            
    except Exception as e:
        logger.error(f"❌ Build failed: {e}")
        raise

if __name__ == "__main__":
    main()
