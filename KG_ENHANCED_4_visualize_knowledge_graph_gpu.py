#!/usr/bin/env python3
"""
GPU-Accelerated Enhanced Chapter-Based Knowledge Graph Visualization System
Advanced visualization system for the enhanced chapter-based knowledge graph with concept clusters
"""
import os
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from neo4j import GraphDatabase
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from datetime import datetime
import colorsys
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "knowledge123"

class GPUAcceleratedEnhancedKnowledgeGraphVisualizer:
    """GPU-accelerated visualization system for enhanced chapter-based knowledge graph"""
    
    def __init__(self):
        self.setup_gpu()
        self.setup_connections()
        self.load_enhanced_data()
        self.setup_style()
    
    def setup_gpu(self):
        """Setup GPU for computations"""
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.empty_cache()
            logger.info(f"🚀 GPU acceleration enabled - Using {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu"
            logger.warning("⚠️ CUDA not available, using CPU")
    
    def setup_connections(self):
        """Setup Neo4j connection"""
        try:
            self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            logger.info("✅ Connected to Neo4j")
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            raise
    
    def load_enhanced_data(self):
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
            
        except FileNotFoundError as e:
            logger.error(f"❌ Enhanced graph data not found: {e}")
            logger.info("💡 Please run 'KG_ENHANCED_2_build_knowledge_graph_gpu.py' first")
            raise
    
    def setup_style(self):
        """Setup visualization styles"""
        # Define color schemes
        self.chapter_colors = px.colors.qualitative.Set3
        self.difficulty_colors = {
            'beginner': '#2ecc71',      # Green
            'intermediate': '#f39c12',   # Orange
            'advanced': '#e74c3c'        # Red
        }
        
        self.concept_type_colors = {
            'explicit_objective': '#3498db',    # Blue
            'key_term': '#9b59b6',             # Purple
            'technical_concept': '#e67e22',     # Orange
            'section_concept': '#1abc9c',       # Teal
            'question_concept': '#f1c40f'       # Yellow
        }
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_enhanced_chapter_overview(self) -> go.Figure:
        """Generate enhanced interactive chapter overview with cluster information"""
        logger.info("📊 Generating enhanced chapter overview...")
        
        # Prepare data
        chapter_data = []
        for ch in self.chapters:
            concepts = ch['learning_concepts']
            clusters = ch.get('concept_clusters', [])
            
            # Calculate difficulty distribution
            difficulty_dist = Counter([c['difficulty'] for c in concepts])
            
            chapter_data.append({
                'chapter_number': int(ch['number']) if ch['number'].isdigit() else 999,
                'title': ch['title'],
                'num_concepts': len(concepts),
                'num_clusters': len(clusters),
                'estimated_hours': ch['estimated_hours'],
                'difficulty': ch['difficulty'],
                'beginner_concepts': difficulty_dist.get('beginner', 0),
                'intermediate_concepts': difficulty_dist.get('intermediate', 0),
                'advanced_concepts': difficulty_dist.get('advanced', 0),
                'cluster_names': [cl['name'] for cl in clusters[:3]]  # Top 3 clusters
            })
        
        df = pd.DataFrame(chapter_data).sort_values('chapter_number')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Concepts per Chapter', 'Study Hours per Chapter',
                'Concept Clusters per Chapter', 'Difficulty Distribution',
                'Chapter Complexity Overview', 'Learning Trajectory'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Concepts per Chapter
        fig.add_trace(
            go.Bar(
                x=df['chapter_number'],
                y=df['num_concepts'],
                name='Concepts',
                text=df['title'],
                textposition='outside',
                marker_color='lightblue',
                hovertemplate='<b>Chapter %{x}</b><br>%{text}<br>Concepts: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Study Hours per Chapter
        fig.add_trace(
            go.Scatter(
                x=df['chapter_number'],
                y=df['estimated_hours'],
                mode='lines+markers',
                name='Study Hours',
                line=dict(color='orange', width=3),
                marker=dict(size=8),
                hovertemplate='<b>Chapter %{x}</b><br>Study Hours: %{y:.1f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Concept Clusters per Chapter
        fig.add_trace(
            go.Bar(
                x=df['chapter_number'],
                y=df['num_clusters'],
                name='Clusters',
                marker_color='lightgreen',
                hovertemplate='<b>Chapter %{x}</b><br>Clusters: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Difficulty Distribution - Stacked Bar
        fig.add_trace(
            go.Bar(
                x=df['chapter_number'],
                y=df['beginner_concepts'],
                name='Beginner',
                marker_color=self.difficulty_colors['beginner'],
                hovertemplate='<b>Chapter %{x}</b><br>Beginner: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=df['chapter_number'],
                y=df['intermediate_concepts'],
                name='Intermediate',
                marker_color=self.difficulty_colors['intermediate'],
                hovertemplate='<b>Chapter %{x}</b><br>Intermediate: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=df['chapter_number'],
                y=df['advanced_concepts'],
                name='Advanced',
                marker_color=self.difficulty_colors['advanced'],
                hovertemplate='<b>Chapter %{x}</b><br>Advanced: %{y}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 5. Chapter Complexity (bubble chart)
        fig.add_trace(
            go.Scatter(
                x=df['num_concepts'],
                y=df['estimated_hours'],
                mode='markers',
                marker=dict(
                    size=df['num_clusters'] * 5,
                    color=df['chapter_number'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Chapter Number')
                ),
                text=df['title'],
                name='Complexity',
                hovertemplate='<b>%{text}</b><br>Concepts: %{x}<br>Hours: %{y}<br>Clusters: %{marker.size}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 6. Learning Trajectory
        cumulative_concepts = df['num_concepts'].cumsum()
        cumulative_hours = df['estimated_hours'].cumsum()
        
        fig.add_trace(
            go.Scatter(
                x=df['chapter_number'],
                y=cumulative_concepts,
                mode='lines+markers',
                name='Cumulative Concepts',
                line=dict(color='blue', width=2),
                yaxis='y'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="📚 Enhanced Chapter-Based Knowledge Graph Overview",
            title_x=0.5,
            showlegend=True,
            barmode='stack'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Chapter Number", row=1, col=1)
        fig.update_xaxes(title_text="Chapter Number", row=1, col=2)
        fig.update_xaxes(title_text="Chapter Number", row=2, col=1)
        fig.update_xaxes(title_text="Chapter Number", row=2, col=2)
        fig.update_xaxes(title_text="Number of Concepts", row=3, col=1)
        fig.update_xaxes(title_text="Chapter Number", row=3, col=2)
        
        fig.update_yaxes(title_text="Number of Concepts", row=1, col=1)
        fig.update_yaxes(title_text="Study Hours", row=1, col=2)
        fig.update_yaxes(title_text="Number of Clusters", row=2, col=1)
        fig.update_yaxes(title_text="Number of Concepts", row=2, col=2)
        fig.update_yaxes(title_text="Study Hours", row=3, col=1)
        fig.update_yaxes(title_text="Cumulative Concepts", row=3, col=2)
        
        return fig
    
    def visualize_concept_clusters_3d(self, chapter_id: Optional[str] = None) -> go.Figure:
        """Create 3D visualization of concept clusters using GPU-accelerated dimensionality reduction"""
        logger.info(f"🔮 Creating 3D concept cluster visualization for chapter: {chapter_id or 'All'}")
        
        # Filter concepts
        if chapter_id:
            concepts = [c for c in self.concepts if c['chapter_id'] == chapter_id]
            title_suffix = f" - Chapter {chapter_id}"
        else:
            concepts = self.concepts[:500]  # Limit for performance
            title_suffix = " - Sample (500 concepts)"
        
        if len(concepts) < 3:
            logger.warning("Not enough concepts for 3D visualization")
            return go.Figure()
        
        # Extract embeddings and metadata
        embeddings = np.array([c['embedding'] for c in concepts])
        
        # GPU-accelerated dimensionality reduction
        if len(embeddings) > 50:
            if self.device == "cuda":
                # Use GPU for PCA if available
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).cuda()
                # Perform PCA on GPU
                U, S, V = torch.pca_lowrank(embeddings_tensor, q=50)
                embeddings_reduced = U @ torch.diag(S)
                embeddings_reduced = embeddings_reduced.cpu().numpy()
            else:
                # Use CPU PCA
                pca = PCA(n_components=50)
                embeddings_reduced = pca.fit_transform(embeddings)
            
            # Apply t-SNE for final 3D reduction
            tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(concepts)//4))
            embeddings_3d = tsne.fit_transform(embeddings_reduced)
        else:
            # Direct PCA for small datasets
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(embeddings)
        
        # Prepare data for visualization
        concept_data = []
        for i, concept in enumerate(concepts):
            chapter = next((ch for ch in self.chapters if ch['id'] == concept['chapter_id']), None)
            concept_data.append({
                'x': embeddings_3d[i, 0],
                'y': embeddings_3d[i, 1],
                'z': embeddings_3d[i, 2],
                'text': concept['text'][:50] + '...',
                'type': concept['type'],
                'difficulty': concept['difficulty'],
                'confidence': concept['confidence'],
                'chapter_title': chapter['title'] if chapter else 'Unknown',
                'chapter_number': chapter['number'] if chapter else 'Unknown',
                'cluster_id': concept.get('cluster_id', 'No Cluster')
            })
        
        df = pd.DataFrame(concept_data)
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add traces for each concept type
        for concept_type in df['type'].unique():
            mask = df['type'] == concept_type
            type_data = df[mask]
            
            fig.add_trace(go.Scatter3d(
                x=type_data['x'],
                y=type_data['y'],
                z=type_data['z'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.concept_type_colors.get(concept_type, '#7f7f7f'),
                    opacity=0.8
                ),
                text=type_data['text'],
                name=concept_type.replace('_', ' ').title(),
                hovertemplate=
                '<b>%{text}</b><br>' +
                'Type: ' + concept_type + '<br>' +
                'Difficulty: %{customdata[0]}<br>' +
                'Confidence: %{customdata[1]:.2f}<br>' +
                'Chapter: %{customdata[2]}<br>' +
                '<extra></extra>',
                customdata=list(zip(type_data['difficulty'], type_data['confidence'], type_data['chapter_title']))
            ))
        
        # Update layout
        fig.update_layout(
            title=f'🔮 3D Concept Cluster Visualization{title_suffix}',
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_enhanced_prerequisite_network(self) -> go.Figure:
        """Create enhanced prerequisite network visualization with concept clusters"""
        logger.info("🌐 Creating enhanced prerequisite network...")
        
        # Fetch graph data from Neo4j
        with self.neo4j_driver.session() as session:
            result = session.run("""
            MATCH (c:Chapter)
            OPTIONAL MATCH (c)-[:PREREQUISITE_FOR]->(next:Chapter)
            OPTIONAL MATCH (c)-[:CONTAINS_CLUSTER]->(cl:ConceptCluster)
            RETURN c.id as id, c.title as title, c.number as number,
                   c.difficulty as difficulty, c.num_concepts as num_concepts,
                   collect(DISTINCT next.id) as prerequisites,
                   collect(DISTINCT {
                       id: cl.id,
                       name: cl.name,
                       type: cl.type,
                       num_concepts: cl.num_concepts
                   }) as clusters
            """)
            
            chapters_data = []
            for record in result:
                chapters_data.append({
                    'id': record['id'],
                    'title': record['title'],
                    'number': record['number'],
                    'difficulty': record['difficulty'],
                    'num_concepts': record['num_concepts'],
                    'prerequisites': record['prerequisites'],
                    'clusters': [cl for cl in record['clusters'] if cl['id']]
                })
        
        # Build NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for ch in chapters_data:
            size = ch['num_concepts'] * 2 + 10  # Scale node size by concepts
            G.add_node(
                ch['id'],
                title=ch['title'],
                number=ch['number'],
                difficulty=ch['difficulty'],
                num_concepts=ch['num_concepts'],
                num_clusters=len(ch['clusters']),
                size=size
            )
        
        # Add edges
        for ch in chapters_data:
            for prereq_id in ch['prerequisites']:
                if prereq_id in G.nodes():
                    G.add_edge(ch['id'], prereq_id)
        
        # Calculate layout using GPU if available
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, k=2, iterations=50)
        else:
            pos = {}
        
        # Prepare traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightgray'),
            hoverinfo='none',
            mode='lines',
            name='Prerequisites'
        ))
        
        # Add nodes by difficulty
        for difficulty in ['beginner', 'intermediate', 'advanced']:
            difficulty_nodes = [n for n, d in G.nodes(data=True) if d.get('difficulty') == difficulty]
            if not difficulty_nodes:
                continue
            
            node_x = [pos[node][0] for node in difficulty_nodes]
            node_y = [pos[node][1] for node in difficulty_nodes]
            node_sizes = [G.nodes[node]['size'] for node in difficulty_nodes]
            node_text = [f"Ch {G.nodes[node]['number']}: {G.nodes[node]['title']}" for node in difficulty_nodes]
            node_info = [f"Concepts: {G.nodes[node]['num_concepts']}<br>Clusters: {G.nodes[node]['num_clusters']}" 
                        for node in difficulty_nodes]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=self.difficulty_colors[difficulty],
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                text=[f"Ch {G.nodes[node]['number']}" for node in difficulty_nodes],
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                name=difficulty.title(),
                hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>',
                customdata=list(zip(node_text, node_info))
            ))
        
        # Update layout
        fig.update_layout(
            title='🌐 Enhanced Chapter Prerequisite Network',
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Node size represents number of concepts<br>Colors represent difficulty levels",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_concept_difficulty_heatmap(self) -> go.Figure:
        """Create heatmap showing concept difficulty distribution across chapters"""
        logger.info("🔥 Creating concept difficulty heatmap...")
        
        # Prepare data
        heatmap_data = []
        chapter_labels = []
        
        sorted_chapters = sorted(self.chapters, key=lambda x: int(x['number']) if x['number'].isdigit() else 999)
        
        for ch in sorted_chapters:
            concepts = ch['learning_concepts']
            difficulty_counts = {'beginner': 0, 'intermediate': 0, 'advanced': 0}
            
            for concept in concepts:
                difficulty = concept['difficulty']
                if difficulty in difficulty_counts:
                    difficulty_counts[difficulty] += 1
            
            total_concepts = sum(difficulty_counts.values())
            if total_concepts > 0:
                # Calculate percentages
                percentages = [
                    difficulty_counts['beginner'] / total_concepts * 100,
                    difficulty_counts['intermediate'] / total_concepts * 100,
                    difficulty_counts['advanced'] / total_concepts * 100
                ]
            else:
                percentages = [0, 0, 0]
            
            heatmap_data.append(percentages)
            chapter_labels.append(f"Ch {ch['number']}")
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=np.array(heatmap_data).T,
            x=chapter_labels,
            y=['Beginner', 'Intermediate', 'Advanced'],
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title="Percentage"),
            hovertemplate='<b>%{y} Concepts</b><br>Chapter: %{x}<br>Percentage: %{z:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='🔥 Concept Difficulty Distribution Across Chapters',
            xaxis_title='Chapter',
            yaxis_title='Difficulty Level',
            height=400
        )
        
        return fig
    
    def create_learning_analytics_dashboard(self, user_progress: Optional[Dict] = None) -> go.Figure:
        """Create comprehensive learning analytics dashboard"""
        logger.info("📊 Creating learning analytics dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Concept Type Distribution',
                'Chapter Complexity Analysis', 
                'Learning Path Efficiency',
                'Concept Cluster Analysis',
                'Difficulty Progression',
                'Study Time Estimation'
            ),
            specs=[
                [{"type": "pie"}, {"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # 1. Concept Type Distribution (Pie Chart)
        type_counts = Counter([c['type'] for c in self.concepts])
        fig.add_trace(
            go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                name="Concept Types",
                hole=0.3
            ),
            row=1, col=1
        )
        
        # 2. Chapter Complexity Analysis (Scatter)
        chapter_complexity = []
        for ch in self.chapters:
            concepts = ch['learning_concepts']
            avg_confidence = np.mean([c['confidence'] for c in concepts]) if concepts else 0
            chapter_complexity.append({
                'chapter': f"Ch {ch['number']}",
                'num_concepts': len(concepts),
                'estimated_hours': ch['estimated_hours'],
                'avg_confidence': avg_confidence,
                'num_clusters': len(ch.get('concept_clusters', []))
            })
        
        complexity_df = pd.DataFrame(chapter_complexity)
        fig.add_trace(
            go.Scatter(
                x=complexity_df['num_concepts'],
                y=complexity_df['estimated_hours'],
                mode='markers',
                marker=dict(
                    size=complexity_df['num_clusters'] * 3,
                    color=complexity_df['avg_confidence'],
                    colorscale='Viridis',
                    showscale=False
                ),
                text=complexity_df['chapter'],
                name='Complexity'
            ),
            row=1, col=2
        )
        
        # 3. Learning Path Efficiency (Bar)
        path_efficiency = []
        sorted_chapters = sorted(self.chapters, key=lambda x: int(x['number']) if x['number'].isdigit() else 999)
        
        for i, ch in enumerate(sorted_chapters[:10]):  # First 10 chapters
            concepts_per_hour = len(ch['learning_concepts']) / ch['estimated_hours'] if ch['estimated_hours'] > 0 else 0
            path_efficiency.append({
                'chapter': f"Ch {ch['number']}",
                'efficiency': concepts_per_hour
            })
        
        efficiency_df = pd.DataFrame(path_efficiency)
        fig.add_trace(
            go.Bar(
                x=efficiency_df['chapter'],
                y=efficiency_df['efficiency'],
                name='Concepts/Hour',
                marker_color='lightblue'
            ),
            row=1, col=3
        )
        
        # 4. Concept Cluster Analysis (Bar)
        cluster_types = Counter()
        cluster_sizes = []
        
        for ch in self.chapters:
            for cluster in ch.get('concept_clusters', []):
                cluster_types[cluster['type']] += 1
                cluster_sizes.append(cluster['num_concepts'])
        
        fig.add_trace(
            go.Bar(
                x=list(cluster_types.keys()),
                y=list(cluster_types.values()),
                name='Cluster Types',
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # 5. Difficulty Progression (Scatter)
        difficulty_progression = []
        for ch in sorted_chapters:
            concepts = ch['learning_concepts']
            difficulty_scores = {'beginner': 1, 'intermediate': 2, 'advanced': 3}
            
            if concepts:
                avg_difficulty = np.mean([difficulty_scores.get(c['difficulty'], 2) for c in concepts])
            else:
                avg_difficulty = 2
            
            difficulty_progression.append({
                'chapter_num': int(ch['number']) if ch['number'].isdigit() else 999,
                'avg_difficulty': avg_difficulty,
                'num_concepts': len(concepts)
            })
        
        progression_df = pd.DataFrame(difficulty_progression).sort_values('chapter_num')
        fig.add_trace(
            go.Scatter(
                x=progression_df['chapter_num'],
                y=progression_df['avg_difficulty'],
                mode='lines+markers',
                name='Avg Difficulty',
                line=dict(color='red', width=2)
            ),
            row=2, col=2
        )
        
        # 6. Study Time Estimation (Bar)
        time_distribution = []
        for ch in sorted_chapters[:10]:  # First 10 chapters
            time_distribution.append({
                'chapter': f"Ch {ch['number']}",
                'estimated_hours': ch['estimated_hours'],
                'concepts': len(ch['learning_concepts'])
            })
        
        time_df = pd.DataFrame(time_distribution)
        fig.add_trace(
            go.Bar(
                x=time_df['chapter'],
                y=time_df['estimated_hours'],
                name='Study Hours',
                marker_color='orange'
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="📊 Enhanced Learning Analytics Dashboard",
            title_x=0.5,
            showlegend=False
        )
        
        return fig
    
    def create_concept_similarity_network(self, chapter_id: str, max_concepts: int = 30) -> go.Figure:
        """Create network showing concept similarities within a chapter"""
        logger.info(f"🕸️ Creating concept similarity network for chapter: {chapter_id}")
        
        # Get concepts for the chapter
        chapter_concepts = [c for c in self.concepts if c['chapter_id'] == chapter_id][:max_concepts]
        
        if len(chapter_concepts) < 2:
            return go.Figure().add_annotation(text="Not enough concepts for network visualization")
        
        # Calculate similarity matrix using GPU if available
        embeddings = np.array([c['embedding'] for c in chapter_concepts])
        
        if self.device == "cuda":
            embeddings_tensor = torch.tensor(embeddings).cuda()
            similarity_matrix = torch.mm(embeddings_tensor, embeddings_tensor.t()).cpu().numpy()
        else:
            similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, concept in enumerate(chapter_concepts):
            G.add_node(i, 
                      text=concept['text'][:30] + '...',
                      type=concept['type'],
                      difficulty=concept['difficulty'],
                      confidence=concept['confidence'])
        
        # Add edges (only for strong similarities)
        threshold = 0.7
        for i in range(len(chapter_concepts)):
            for j in range(i+1, len(chapter_concepts)):
                if similarity_matrix[i][j] > threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i][j])
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare traces
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.extend([edge[2]['weight'], edge[2]['weight'], None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightgray'),
            hoverinfo='none',
            mode='lines',
            name='Similarities'
        ))
        
        # Add nodes by type
        for concept_type in set(c['type'] for c in chapter_concepts):
            type_nodes = [i for i, c in enumerate(chapter_concepts) if c['type'] == concept_type]
            
            node_x = [pos[i][0] for i in type_nodes]
            node_y = [pos[i][1] for i in type_nodes]
            node_text = [chapter_concepts[i]['text'][:30] + '...' for i in type_nodes]
            
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(
                    size=15,
                    color=self.concept_type_colors.get(concept_type, '#7f7f7f'),
                    line=dict(width=2, color='white')
                ),
                text=node_text,
                name=concept_type.replace('_', ' ').title(),
                hovertemplate='<b>%{text}</b><br>Type: ' + concept_type + '<extra></extra>'
            ))
        
        # Find chapter title
        chapter = next((ch for ch in self.chapters if ch['id'] == chapter_id), None)
        chapter_title = chapter['title'] if chapter else f'Chapter {chapter_id}'
        
        fig.update_layout(
            title=f'🕸️ Concept Similarity Network - {chapter_title}',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def generate_enhanced_report(self, output_dir: str = "./enhanced_visualizations"):
        """Generate comprehensive enhanced visualization report"""
        logger.info("📝 Generating enhanced visualization report...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all visualizations
        visualizations = {}
        
        try:
            # 1. Chapter Overview
            logger.info("Creating chapter overview...")
            visualizations['chapter_overview'] = self.generate_enhanced_chapter_overview()
            
            # 2. 3D Concept Clusters
            logger.info("Creating 3D concept visualization...")
            visualizations['concept_3d'] = self.visualize_concept_clusters_3d()
            
            # 3. Prerequisite Network
            logger.info("Creating prerequisite network...")
            visualizations['prerequisite_network'] = self.create_enhanced_prerequisite_network()
            
            # 4. Difficulty Heatmap
            logger.info("Creating difficulty heatmap...")
            visualizations['difficulty_heatmap'] = self.create_concept_difficulty_heatmap()
            
            # 5. Analytics Dashboard
            logger.info("Creating analytics dashboard...")
            visualizations['analytics_dashboard'] = self.create_learning_analytics_dashboard()
            
            # 6. Sample Concept Network (first chapter)
            if self.chapters:
                first_chapter_id = self.chapters[0]['id']
                logger.info(f"Creating concept network for chapter: {first_chapter_id}")
                visualizations['concept_network'] = self.create_concept_similarity_network(first_chapter_id)
            
            # Save all visualizations
            for name, fig in visualizations.items():
                if fig.data:  # Check if figure has data
                    html_path = os.path.join(output_dir, f"{name}.html")
                    fig.write_html(html_path)
                    logger.info(f"💾 Saved {name} to {html_path}")
            
            # Create summary report
            self.create_summary_report(output_dir)
            
            logger.info(f"✅ Enhanced visualization report generated in: {output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")
            raise
    
    def create_summary_report(self, output_dir: str):
        """Create HTML summary report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Knowledge Graph Visualization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .stats {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .visualization {{ margin: 20px 0; }}
                .link {{ color: #3498db; text-decoration: none; }}
                .link:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>
            <h1>🚀 Enhanced Knowledge Graph Visualization Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="stats">
                <h2>📊 System Statistics</h2>
                <ul>
                    <li><strong>Total Chapters:</strong> {len(self.chapters)}</li>
                    <li><strong>Total Learning Concepts:</strong> {len(self.concepts)}</li>
                    <li><strong>Average Concepts per Chapter:</strong> {np.mean([len(ch['learning_concepts']) for ch in self.chapters]):.1f}</li>
                    <li><strong>Total Study Hours:</strong> {sum(ch['estimated_hours'] for ch in self.chapters):.1f}</li>
                    <li><strong>GPU Acceleration:</strong> {'Enabled' if self.device == 'cuda' else 'Disabled'}</li>
                </ul>
            </div>
            
            <h2>📈 Available Visualizations</h2>
            
            <div class="visualization">
                <h3>1. <a href="chapter_overview.html" class="link">📚 Enhanced Chapter Overview</a></h3>
                <p>Comprehensive overview of all chapters with concepts, clusters, and study time analysis.</p>
            </div>
            
            <div class="visualization">
                <h3>2. <a href="concept_3d.html" class="link">🔮 3D Concept Cluster Visualization</a></h3>
                <p>Interactive 3D visualization of concept clusters using GPU-accelerated dimensionality reduction.</p>
            </div>
            
            <div class="visualization">
                <h3>3. <a href="prerequisite_network.html" class="link">🌐 Enhanced Prerequisite Network</a></h3>
                <p>Network diagram showing chapter prerequisites and concept cluster relationships.</p>
            </div>
            
            <div class="visualization">
                <h3>4. <a href="difficulty_heatmap.html" class="link">🔥 Concept Difficulty Heatmap</a></h3>
                <p>Heatmap showing difficulty distribution of concepts across chapters.</p>
            </div>
            
            <div class="visualization">
                <h3>5. <a href="analytics_dashboard.html" class="link">📊 Learning Analytics Dashboard</a></h3>
                <p>Comprehensive dashboard with learning analytics and progress tracking capabilities.</p>
            </div>
            
            <div class="visualization">
                <h3>6. <a href="concept_network.html" class="link">🕸️ Concept Similarity Network</a></h3>
                <p>Network showing concept similarities within individual chapters.</p>
            </div>
            
            <div class="stats">
                <h2>🛠️ Technical Details</h2>
                <ul>
                    <li><strong>Embedding Model:</strong> sentence-transformers/all-mpnet-base-v2</li>
                    <li><strong>Dimensionality Reduction:</strong> GPU-accelerated PCA + t-SNE</li>
                    <li><strong>Graph Database:</strong> Neo4j with concept clusters</li>
                    <li><strong>Visualization Library:</strong> Plotly with interactive features</li>
                    <li><strong>Concept Extraction:</strong> 5 different extraction methods per chapter</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(output_dir, 'index.html'), 'w') as f:
            f.write(html_content)
    
    def interactive_visualization_mode(self):
        """Interactive mode for generating specific visualizations"""
        print("\n🎨 GPU-Accelerated Enhanced Knowledge Graph Visualization System")
        print("=" * 70)
        print("Available visualizations:")
        print("  1. overview         - Enhanced chapter overview")
        print("  2. 3d [chapter_id]  - 3D concept clusters (optionally for specific chapter)")
        print("  3. network          - Prerequisite network")
        print("  4. heatmap          - Difficulty heatmap")
        print("  5. dashboard        - Analytics dashboard")
        print("  6. concepts <ch_id> - Concept similarity network for chapter")
        print("  7. report           - Generate complete report")
        print("  8. list-chapters    - List all available chapters")
        print("  quit                - Exit")
        print("=" * 70)
        
        while True:
            try:
                user_input = input("\n🎨 Enter visualization command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                parts = user_input.split()
                command = parts[0].lower()
                
                if command == 'overview' or command == '1':
                    fig = self.generate_enhanced_chapter_overview()
                    fig.show()
                
                elif command == '3d' or command == '2':
                    chapter_id = parts[1] if len(parts) > 1 else None
                    fig = self.visualize_concept_clusters_3d(chapter_id)
                    fig.show()
                
                elif command == 'network' or command == '3':
                    fig = self.create_enhanced_prerequisite_network()
                    fig.show()
                
                elif command == 'heatmap' or command == '4':
                    fig = self.create_concept_difficulty_heatmap()
                    fig.show()
                
                elif command == 'dashboard' or command == '5':
                    fig = self.create_learning_analytics_dashboard()
                    fig.show()
                
                elif command == 'concepts' or command == '6':
                    if len(parts) < 2:
                        print("❌ Please provide a chapter ID")
                        continue
                    
                    chapter_id = parts[1]
                    fig = self.create_concept_similarity_network(chapter_id)
                    fig.show()
                
                elif command == 'report' or command == '7':
                    self.generate_enhanced_report()
                    print("✅ Complete report generated in ./enhanced_visualizations/")
                
                elif command == 'list-chapters' or command == '8':
                    print(f"\n📋 Available Chapters ({len(self.chapters)} total):")
                    print("-" * 50)
                    
                    sorted_chapters = sorted(self.chapters, key=lambda x: int(x['number']) if x['number'].isdigit() else 999)
                    for ch in sorted_chapters:
                        concepts_count = len(ch['learning_concepts'])
                        clusters_count = len(ch.get('concept_clusters', []))
                        print(f"  {ch['id']}: Chapter {ch['number']} - {ch['title']}")
                        print(f"    📝 {concepts_count} concepts, 🔗 {clusters_count} clusters")
                
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
    visualizer = None
    try:
        visualizer = GPUAcceleratedEnhancedKnowledgeGraphVisualizer()
        visualizer.interactive_visualization_mode()
        
    except Exception as e:
        logger.error(f"❌ Visualization system initialization failed: {e}")
        raise
    
    finally:
        if visualizer:
            visualizer.close_connections()

if __name__ == "__main__":
    main()
