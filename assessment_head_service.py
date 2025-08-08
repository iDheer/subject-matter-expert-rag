# assessment_head_service.py - LangChain Version
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import requests

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from flask import Flask, request, jsonify

class AssessmentType(Enum):
    DIAGNOSTIC = "diagnostic"          # Identify knowledge gaps
    FORMATIVE = "formative"           # Ongoing learning check
    SUMMATIVE = "summative"           # Final evaluation
    COMPETENCY = "competency"         # Skill-based assessment
    ADAPTIVE = "adaptive"             # Difficulty adjusts based on performance

class AssessmentFormat(Enum):
    TRADITIONAL = "traditional"       # Standard Q&A format
    SCENARIO_BASED = "scenario"      # Real-world scenarios
    PROJECT_BASED = "project"        # Hands-on projects
    PORTFOLIO = "portfolio"          # Collection of work
    PEER_REVIEW = "peer_review"      # Peer evaluation component

@dataclass
class AssessmentCriterion:
    criterion_id: str
    name: str
    description: str
    weight: float  # 0.0 to 1.0
    rubric_levels: Dict[str, str]  # e.g., {"excellent": "Description", "good": "..."}
    measurable_outcomes: List[str]

@dataclass
class AssessmentTask:
    task_id: str
    task_type: str  # "question", "scenario", "project", "demonstration"
    title: str
    description: str
    instructions: str
    criteria: List[str]  # References to AssessmentCriterion IDs
    estimated_time_minutes: int
    difficulty_level: str
    required_resources: List[str]
    success_indicators: List[str]
    sample_response: Optional[str] = None

@dataclass
class AssessmentRequest:
    strategy_context: Dict[str, Any]
    topic: str
    assessment_type: str = "formative"
    assessment_format: str = "traditional"
    duration_minutes: int = 30
    competencies_to_assess: List[str] = None
    learning_objectives: List[str] = None
    performance_standards: Dict[str, Any] = None

@dataclass
class GeneratedAssessment:
    assessment_id: str
    title: str
    description: str
    assessment_type: str
    format_type: str
    topic: str
    total_duration_minutes: int
    criteria: List[AssessmentCriterion]
    tasks: List[AssessmentTask]
    scoring_rubric: Dict[str, Any]
    adaptive_parameters: Optional[Dict[str, Any]]
    generation_metadata: Dict[str, Any]
    created_at: str

# Custom output parsers for LangChain
class CriteriaOutputParser(BaseOutputParser):
    """Parse LLM output into assessment criteria"""
    
    def parse(self, text: str) -> List[Dict]:
        try:
            start_idx = text.find("[")
            end_idx = text.rfind("]") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = text[start_idx:end_idx]
                return json.loads(json_text)
            else:
                print("⚠️ Could not find JSON in LLM response")
                return []
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            return []

class TasksOutputParser(BaseOutputParser):
    """Parse LLM output into assessment tasks"""
    
    def parse(self, text: str) -> List[Dict]:
        try:
            start_idx = text.find("[")
            end_idx = text.rfind("]") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = text[start_idx:end_idx]
                return json.loads(json_text)
            else:
                print("⚠️ Could not find JSON in LLM response")
                return []
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            return []

class AssessmentHeadService:
    """Standalone Assessment Generation Service using LangChain"""
    
    def __init__(self, sme_endpoint: str = "http://localhost:8000", llm_model: str = "qwen3:4b"):
        self.sme_endpoint = sme_endpoint
        
        # Initialize LangChain components
        self.llm = OllamaLLM(model=llm_model, temperature=0.7)
        self.criteria_parser = CriteriaOutputParser()
        self.tasks_parser = TasksOutputParser()
        
        # Assessment profiles based on learner categories
        self.assessment_profiles = {
            "Rising Improver": {
                "preferred_format": "scenario",
                "assessment_focus": "progress_tracking",
                "feedback_style": "constructive_detailed",
                "difficulty_level": "intermediate",
                "support_level": "high",
                "success_celebration": True,
                "remediation_suggestions": True
            },
            "Stable Expert": {
                "preferred_format": "project",
                "assessment_focus": "application_mastery",
                "feedback_style": "concise_actionable",
                "difficulty_level": "advanced",
                "support_level": "minimal",
                "success_celebration": False,
                "innovation_challenges": True
            },
            "Struggling Learner": {
                "preferred_format": "traditional",
                "assessment_focus": "foundational_skills",
                "feedback_style": "supportive_detailed",
                "difficulty_level": "beginner",
                "support_level": "very_high",
                "confidence_building": True,
                "small_wins": True
            },
            "Advanced Explorer": {
                "preferred_format": "portfolio",
                "assessment_focus": "creative_application",
                "feedback_style": "analytical_deep",
                "difficulty_level": "expert",
                "support_level": "minimal",
                "open_ended_tasks": True,
                "research_components": True
            }
        }
        
        # Standard competency frameworks
        self.competency_frameworks = {
            "bloom_taxonomy": [
                "remembering", "understanding", "applying", 
                "analyzing", "evaluating", "creating"
            ],
            "technical_skills": [
                "basic_knowledge", "application", "problem_solving",
                "optimization", "innovation", "teaching_others"
            ],
            "soft_skills": [
                "communication", "collaboration", "critical_thinking",
                "adaptability", "leadership", "emotional_intelligence"
            ]
        }
        
        # Create LangChain prompt templates
        self.criteria_prompt = PromptTemplate(
            input_variables=[
                "topic", "learner_category", "assessment_type", "feedback_style", 
                "assessment_focus", "difficulty_level"
            ],
            template="""<think>
I need to create assessment criteria for a {learner_category} learner on the topic "{topic}".
This should be a {assessment_type} assessment with {feedback_style} feedback.

The criteria should match their skill level and provide appropriate challenge.
</think>

ASSESSMENT CRITERIA GENERATION

TOPIC: {topic}
LEARNER PROFILE: {learner_category}
ASSESSMENT TYPE: {assessment_type}
FEEDBACK STYLE: {feedback_style}

Generate 3-5 assessment criteria that:
1. Match the learner's current level and challenge needs
2. Provide clear, measurable outcomes
3. Support {assessment_focus} focus
4. Include appropriate rubric levels

OUTPUT FORMAT (JSON):
```json
[
  {{
    "name": "Conceptual Understanding",
    "description": "Demonstrates clear understanding of key concepts",
    "weight": 0.3,
    "rubric_levels": {{
      "excellent": "Complete and nuanced understanding",
      "proficient": "Good understanding with minor gaps",
      "developing": "Basic understanding, some confusion",
      "beginning": "Limited understanding, major gaps"
    }},
    "measurable_outcomes": ["Defines key terms", "Explains relationships", "Provides examples"]
  }}
]
```"""
        )
        
        self.tasks_prompt = PromptTemplate(
            input_variables=[
                "topic", "learner_category", "format_type", "assessment_type", 
                "duration_minutes", "difficulty_level", "knowledge_context", 
                "support_level", "assessment_focus"
            ],
            template="""<think>
I need to create assessment tasks for a {learner_category} learner.
Format: {format_type}
Assessment type: {assessment_type}
Duration: {duration_minutes} minutes
Support level needed: {support_level}

I should create tasks that match their learning style and challenge level.
</think>

ASSESSMENT TASK GENERATION

TOPIC: {topic}
LEARNER PROFILE: {learner_category}
FORMAT: {format_type}
ASSESSMENT TYPE: {assessment_type}
DURATION: {duration_minutes} minutes
DIFFICULTY: {difficulty_level}

KNOWLEDGE CONTEXT:
{knowledge_context}

Create 2-4 assessment tasks that:
1. Match the {format_type} format
2. Align with {difficulty_level} difficulty
3. Support {assessment_focus} goals
4. Fit within the time duration

OUTPUT FORMAT (JSON):
```json
[
  {{
    "task_type": "scenario",
    "title": "Real-World Application",
    "description": "Apply concepts to solve a practical problem",
    "instructions": "Given the scenario below, analyze and provide solutions...",
    "estimated_time_minutes": 15,
    "required_resources": ["Calculator", "Reference material"],
    "success_indicators": ["Identifies key issues", "Proposes viable solutions"]
  }}
]
```"""
        )
        
        # Create LangChain chains (modern syntax)
        self.criteria_chain = self.criteria_prompt | self.llm | self.criteria_parser
        self.tasks_chain = self.tasks_prompt | self.llm | self.tasks_parser
    
    def generate_assessment(self, request: AssessmentRequest) -> GeneratedAssessment:
        """Generate a personalized assessment using LangChain"""
        print(f"📋 Generating assessment for topic: {request.topic}")
        
        # Extract learner profile
        learner_profile = self._extract_learner_profile(request.strategy_context)
        
        # Get assessment profile for this learner type
        profile_category = learner_profile.get("category", "Rising Improver")
        assessment_profile = self.assessment_profiles.get(profile_category, self.assessment_profiles["Rising Improver"])
        
        # Fetch knowledge context from SME system
        knowledge_context = self._fetch_knowledge_context(request.topic, request.competencies_to_assess)
        
        # Generate assessment criteria using LangChain
        criteria = self._generate_assessment_criteria_with_langchain(request, learner_profile, assessment_profile)
        
        # Generate assessment tasks using LangChain
        tasks = self._generate_assessment_tasks_with_langchain(request, learner_profile, assessment_profile, knowledge_context)
        
        # Create scoring rubric
        scoring_rubric = self._generate_scoring_rubric(criteria, assessment_profile)
        
        # Set up adaptive parameters if needed
        adaptive_params = None
        if request.assessment_type == "adaptive":
            adaptive_params = self._create_adaptive_parameters(learner_profile)
        
        # Create assessment metadata
        assessment_id = f"assess_{int(time.time())}"
        
        assessment = GeneratedAssessment(
            assessment_id=assessment_id,
            title=f"{request.assessment_type.title()} Assessment: {request.topic}",
            description=self._generate_assessment_description(request, learner_profile),
            assessment_type=request.assessment_type,
            format_type=assessment_profile.get("preferred_format", request.assessment_format),
            topic=request.topic,
            total_duration_minutes=request.duration_minutes,
            criteria=criteria,
            tasks=tasks,
            scoring_rubric=scoring_rubric,
            adaptive_parameters=adaptive_params,
            generation_metadata={
                "learner_profile": learner_profile,
                "assessment_profile": assessment_profile,
                "generation_time": datetime.now().isoformat(),
                "strategy_used": learner_profile.get("category_prompt", ""),
                "delivery_method": request.strategy_context.get("strategy_recommendation", {}).get("strategy_label", ""),
                "langchain_used": True
            },
            created_at=datetime.now().isoformat()
        )
        
        print(f"✅ Generated assessment with {len(tasks)} tasks and {len(criteria)} criteria using LangChain")
        return assessment
    
    def _generate_assessment_criteria_with_langchain(self, request: AssessmentRequest, 
                                                   learner_profile: Dict, assessment_profile: Dict) -> List[AssessmentCriterion]:
        """Generate assessment criteria using LangChain"""
        
        try:
            print("🤖 Generating criteria with LangChain...")
            
            # Prepare inputs for LangChain
            criteria_inputs = {
                "topic": request.topic,
                "learner_category": learner_profile.get("category"),
                "assessment_type": request.assessment_type,
                "feedback_style": assessment_profile.get("feedback_style"),
                "assessment_focus": assessment_profile.get("assessment_focus"),
                "difficulty_level": assessment_profile.get("difficulty_level")
            }
            
            # Run the LangChain chain
            criteria_data = self.criteria_chain.invoke(criteria_inputs)
            
            criteria = []
            for i, c_data in enumerate(criteria_data):
                criterion = AssessmentCriterion(
                    criterion_id=f"criteria_{int(time.time())}_{i}",
                    name=c_data.get("name", f"Criterion {i+1}"),
                    description=c_data.get("description", ""),
                    weight=c_data.get("weight", 0.2),
                    rubric_levels=c_data.get("rubric_levels", {}),
                    measurable_outcomes=c_data.get("measurable_outcomes", [])
                )
                criteria.append(criterion)
            
            return criteria
            
        except Exception as e:
            print(f"⚠️ Error with LangChain criteria generation: {e}")
            return self._generate_default_criteria(request.topic)
    
    def _generate_assessment_tasks_with_langchain(self, request: AssessmentRequest, 
                                                learner_profile: Dict, assessment_profile: Dict, 
                                                knowledge_context: str) -> List[AssessmentTask]:
        """Generate assessment tasks using LangChain"""
        
        try:
            print("🤖 Generating tasks with LangChain...")
            
            format_type = assessment_profile.get("preferred_format", "traditional")
            
            # Prepare inputs for LangChain
            tasks_inputs = {
                "topic": request.topic,
                "learner_category": learner_profile.get("category"),
                "format_type": format_type,
                "assessment_type": request.assessment_type,
                "duration_minutes": request.duration_minutes,
                "difficulty_level": assessment_profile.get("difficulty_level"),
                "knowledge_context": knowledge_context,
                "support_level": assessment_profile.get("support_level"),
                "assessment_focus": assessment_profile.get("assessment_focus")
            }
            
            # Run the LangChain chain
            tasks_data = self.tasks_chain.invoke(tasks_inputs)
            
            tasks = []
            for i, t_data in enumerate(tasks_data):
                task = AssessmentTask(
                    task_id=f"task_{int(time.time())}_{i}",
                    task_type=t_data.get("task_type", "question"),
                    title=t_data.get("title", f"Task {i+1}"),
                    description=t_data.get("description", ""),
                    instructions=t_data.get("instructions", ""),
                    criteria=[],  # Will be linked to criteria IDs later
                    estimated_time_minutes=t_data.get("estimated_time_minutes", 10),
                    difficulty_level=assessment_profile.get("difficulty_level", "intermediate"),
                    required_resources=t_data.get("required_resources", []),
                    success_indicators=t_data.get("success_indicators", [])
                )
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            print(f"⚠️ Error with LangChain task generation: {e}")
            return self._generate_default_tasks(request.topic)
    
    def _extract_learner_profile(self, strategy_context: Dict) -> Dict:
        """Extract learner profile from strategy context"""
        learner_profile = strategy_context.get("learner_profile", {})
        strategy_rec = strategy_context.get("strategy_recommendation", {})
        teaching_approach = strategy_context.get("combined_output", {}).get("teaching_approach", {})
        
        return {
            "category": learner_profile.get("category", "Rising Improver"),
            "category_prompt": learner_profile.get("category_prompt", ""),
            "confidence_score": learner_profile.get("confidence_score", 0.5),
            "redo_topics_flag": learner_profile.get("redo_topics_flag", False),
            "delivery_method": strategy_rec.get("strategy_label", ""),
            "challenge_level": teaching_approach.get("challenge_level", "moderate"),
            "support_level": teaching_approach.get("support_level", "moderate"),
            "pacing": teaching_approach.get("pacing", "moderate"),
            "learner_data": learner_profile.get("input_data", {})
        }
    
    def _fetch_knowledge_context(self, topic: str, competencies: List[str] = None) -> str:
        """Fetch relevant knowledge from SME system"""
        try:
            query_parts = [topic]
            if competencies:
                query_parts.extend(competencies)
            query = " ".join(query_parts)
            
            response = requests.post(f"{self.sme_endpoint}/query", 
                                   json={"query": query, "max_results": 8}, 
                                   timeout=30)
            
            if response.status_code == 200:
                return response.json().get("content", topic)
            else:
                print(f"⚠️ SME system unavailable, using topic only")
                return topic
                
        except Exception as e:
            print(f"⚠️ Error fetching knowledge context: {e}")
            return topic
    
    def _generate_scoring_rubric(self, criteria: List[AssessmentCriterion], 
                               assessment_profile: Dict) -> Dict[str, Any]:
        """Generate scoring rubric based on criteria"""
        
        total_weight = sum(c.weight for c in criteria) if criteria else 1.0
        
        rubric = {
            "scoring_method": "weighted_average",
            "total_points": 100,
            "criteria_weights": {c.name: c.weight / total_weight for c in criteria} if criteria else {},
            "performance_levels": {
                "excellent": {"points": 90, "description": "Exceeds expectations"},
                "proficient": {"points": 80, "description": "Meets expectations"},
                "developing": {"points": 70, "description": "Approaching expectations"},
                "beginning": {"points": 60, "description": "Below expectations"}
            },
            "feedback_style": assessment_profile.get("feedback_style", "balanced")
        }
        
        return rubric
    
    def _create_adaptive_parameters(self, learner_profile: Dict) -> Dict[str, Any]:
        """Create adaptive assessment parameters"""
        
        return {
            "difficulty_adjustment": True,
            "initial_difficulty": learner_profile.get("difficulty_level", "intermediate"),
            "adjustment_threshold": 0.7,  # Adjust if performance > 70%
            "max_difficulty_change": 1,   # Maximum level change
            "question_bank_size": 50,     # Pool of questions to draw from
            "branching_logic": {
                "high_performance": "increase_difficulty",
                "low_performance": "provide_support",
                "medium_performance": "maintain_level"
            }
        }
    
    def _generate_assessment_description(self, request: AssessmentRequest, 
                                       learner_profile: Dict) -> str:
        """Generate personalized assessment description"""
        
        category = learner_profile.get("category", "Learner")
        delivery = learner_profile.get("delivery_method", "Standard")
        
        if learner_profile.get("redo_topics_flag"):
            context = "This assessment helps evaluate your understanding of concepts you've been reviewing."
        else:
            context = "This assessment evaluates your grasp of new concepts and skills."
        
        return f"""Personalized {request.assessment_type} assessment for {category} learners.
{context}

Topic: {request.topic}
Format: {request.assessment_format}
Duration: {request.duration_minutes} minutes
Approach: {delivery}

This assessment is designed to match your learning style and provide appropriate challenge level.
Generated with LangChain for enhanced personalization."""
    
    def _generate_default_criteria(self, topic: str) -> List[AssessmentCriterion]:
        """Generate fallback criteria if LangChain fails"""
        
        return [
            AssessmentCriterion(
                criterion_id=f"default_criteria_{int(time.time())}_1",
                name="Understanding",
                description=f"Demonstrates understanding of {topic}",
                weight=0.4,
                rubric_levels={
                    "excellent": "Complete understanding with examples",
                    "proficient": "Good understanding with minor gaps",
                    "developing": "Basic understanding, some confusion",
                    "beginning": "Limited understanding"
                },
                measurable_outcomes=["Defines concepts", "Explains relationships"]
            ),
            AssessmentCriterion(
                criterion_id=f"default_criteria_{int(time.time())}_2",
                name="Application",
                description=f"Applies {topic} concepts to solve problems",
                weight=0.6,
                rubric_levels={
                    "excellent": "Creative and effective application",
                    "proficient": "Correct application with good reasoning",
                    "developing": "Basic application with guidance",
                    "beginning": "Struggling with application"
                },
                measurable_outcomes=["Solves problems", "Makes connections"]
            )
        ]
    
    def _generate_default_tasks(self, topic: str) -> List[AssessmentTask]:
        """Generate fallback tasks if LangChain fails"""
        
        return [
            AssessmentTask(
                task_id=f"default_task_{int(time.time())}_1",
                task_type="question",
                title=f"{topic} Knowledge Check",
                description=f"Assess understanding of key {topic} concepts",
                instructions=f"Answer the following questions about {topic}",
                criteria=[],
                estimated_time_minutes=15,
                difficulty_level="intermediate",
                required_resources=["Writing materials"],
                success_indicators=["Clear answers", "Correct concepts"]
            )
        ]
    
    def export_assessment(self, assessment: GeneratedAssessment, format: str = "json") -> str:
        """Export assessment in various formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            filename = f"assessment_{assessment.assessment_id}_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(asdict(assessment), f, indent=2, ensure_ascii=False)
        
        return filename

# API Server (Flask-based microservice)
def create_assessment_service_app():
    """Create Flask app for LangChain Assessment Head Service"""
    app = Flask(__name__)
    assessment_service = AssessmentHeadService()
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "healthy", 
            "service": "assessment_head", 
            "framework": "langchain",
            "llm": "qwen3:4b"
        })
    
    @app.route('/generate_assessment', methods=['POST'])
    def generate_assessment():
        try:
            data = request.json
            
            # Create request object
            assessment_request = AssessmentRequest(
                strategy_context=data.get('strategy_context', {}),
                topic=data.get('topic', ''),
                assessment_type=data.get('assessment_type', 'formative'),
                assessment_format=data.get('assessment_format', 'traditional'),
                duration_minutes=data.get('duration_minutes', 30),
                competencies_to_assess=data.get('competencies_to_assess'),
                learning_objectives=data.get('learning_objectives'),
                performance_standards=data.get('performance_standards')
            )
            
            # Generate assessment using LangChain
            assessment = assessment_service.generate_assessment(assessment_request)
            
            return jsonify({
                "status": "success",
                "framework": "langchain",
                "assessment": asdict(assessment)
            })
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
    
    @app.route('/export_assessment', methods=['POST'])
    def export_assessment():
        try:
            data = request.json
            assessment_data = data.get('assessment')
            format_type = data.get('format', 'json')
            
            # Convert dict back to GeneratedAssessment object
            assessment = GeneratedAssessment(**assessment_data)
            filename = assessment_service.export_assessment(assessment, format_type)
            
            return jsonify({
                "status": "success", 
                "filename": filename
            })
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
    
    return app

if __name__ == "__main__":
    # Run as standalone service
    app = create_assessment_service_app()
    print("📋 LangChain Assessment Head Service starting on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=True)