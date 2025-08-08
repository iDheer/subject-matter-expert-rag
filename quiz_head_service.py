# quiz_head_service.py - LangChain Version
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import requests

from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from flask import Flask, request, jsonify

class QuizType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    FILL_BLANKS = "fill_blanks"
    MATCHING = "matching"

class DifficultyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class QuizQuestion:
    question_id: str
    question_type: str
    question_text: str
    options: Optional[List[str]] = None
    correct_answer: str = ""
    explanation: str = ""
    difficulty: str = "intermediate"
    topic_tags: List[str] = None
    estimated_time_minutes: float = 1.0

@dataclass
class QuizGenerationRequest:
    strategy_context: Dict[str, Any]
    topic: str
    num_questions: int = 5
    question_types: List[str] = None
    difficulty_override: Optional[str] = None
    focus_areas: List[str] = None
    time_limit_minutes: Optional[int] = None

@dataclass
class GeneratedQuiz:
    quiz_id: str
    title: str
    description: str
    questions: List[QuizQuestion]
    total_time_minutes: int
    difficulty_level: str
    topic: str
    generation_metadata: Dict[str, Any]
    created_at: str

# Custom output parser for quiz questions
class QuizOutputParser(BaseOutputParser):
    """Parse LLM output into structured quiz questions"""
    
    def parse(self, text: str) -> List[Dict]:
        try:
            # Find JSON block in the text
            start_idx = text.find("[")
            end_idx = text.rfind("]") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_text = text[start_idx:end_idx]
                return json.loads(json_text)
            else:
                print("⚠️ Could not find JSON in LLM response, using fallback")
                return []
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            return []

class QuizHeadService:
    """Standalone Quiz Generation Service using LangChain"""
    
    def __init__(self, sme_endpoint: str = "http://localhost:8000", llm_model: str = "qwen3:4b"):
        self.sme_endpoint = sme_endpoint
        
        # Initialize LangChain components
        self.llm = OllamaLLM(model=llm_model, temperature=0.7)
        self.output_parser = QuizOutputParser()
        
        # Quiz generation templates based on learner profiles
        self.profile_templates = {
            "Rising Improver": {
                "question_distribution": {"multiple_choice": 0.4, "true_false": 0.3, "short_answer": 0.3},
                "difficulty_level": "intermediate",
                "explanation_detail": "detailed",
                "encouragement_tone": "supportive"
            },
            "Stable Expert": {
                "question_distribution": {"multiple_choice": 0.2, "short_answer": 0.4, "fill_blanks": 0.4},
                "difficulty_level": "advanced",
                "explanation_detail": "concise",
                "encouragement_tone": "challenging"
            },
            "Struggling Learner": {
                "question_distribution": {"true_false": 0.5, "multiple_choice": 0.4, "short_answer": 0.1},
                "difficulty_level": "beginner",
                "explanation_detail": "very_detailed",
                "encouragement_tone": "highly_supportive"
            },
            "Advanced Explorer": {
                "question_distribution": {"short_answer": 0.6, "matching": 0.2, "fill_blanks": 0.2},
                "difficulty_level": "expert",
                "explanation_detail": "minimal",
                "encouragement_tone": "innovative"
            }
        }
        
        # Create LangChain prompt template
        self.quiz_prompt = PromptTemplate(
            input_variables=[
                "topic", "num_questions", "learner_category", "category_prompt", 
                "delivery_strategy", "question_types", "difficulty_level", 
                "explanation_detail", "encouragement_tone", "knowledge_context",
                "challenge_level", "support_level", "redo_topics_flag"
            ],
            template="""<think>
I need to generate {num_questions} quiz questions for the topic "{topic}" based on this learner profile and strategy.

Learner Profile: {learner_category}
Challenge Level: {challenge_level}
Support Level: {support_level}
Revisiting Topic: {redo_topics_flag}

I should create questions that match this profile's needs and follow the delivery strategy.
</think>

QUIZ GENERATION TASK

TOPIC: {topic}

LEARNER CONTEXT: {category_prompt}

DELIVERY STRATEGY: {delivery_strategy}

KNOWLEDGE CONTENT TO BASE QUESTIONS ON:
{knowledge_context}

REQUIREMENTS:
- Generate exactly {num_questions} questions
- Question types to include: {question_types}
- Difficulty should match learner profile: {learner_category}
- Tone should be: {encouragement_tone}
- Explanations should be: {explanation_detail}

OUTPUT FORMAT (JSON):
```json
[
  {{
    "type": "multiple_choice",
    "question": "What is the main concept of...?",
    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
    "answer": "B) Option 2",
    "explanation": "Detailed explanation of why this is correct...",
    "estimated_time": 2.0
  }},
  {{
    "type": "true_false",
    "question": "True or False: ...",
    "options": ["True", "False"],
    "answer": "True",
    "explanation": "Explanation...",
    "estimated_time": 1.5
  }}
]
```

Generate questions that will help this learner progress according to their profile and learning strategy."""
        )
        
        # Create LangChain chain (modern syntax)
        self.quiz_chain = self.quiz_prompt | self.llm | self.output_parser
    
    def generate_quiz(self, request: QuizGenerationRequest) -> GeneratedQuiz:
        """Generate a personalized quiz based on strategy context"""
        print(f"🎯 Generating quiz for topic: {request.topic}")
        
        # Extract learner profile information
        learner_profile = self._extract_learner_profile(request.strategy_context)
        
        # Get relevant content from SME system
        knowledge_context = self._fetch_knowledge_context(request.topic, request.focus_areas)
        
        # Generate quiz questions using LangChain
        questions = self._generate_questions_with_langchain(request, learner_profile, knowledge_context)
        
        # Create quiz metadata
        quiz_id = f"quiz_{int(time.time())}"
        total_time = sum(q.estimated_time_minutes for q in questions)
        
        quiz = GeneratedQuiz(
            quiz_id=quiz_id,
            title=f"Quiz: {request.topic}",
            description=self._generate_quiz_description(request, learner_profile),
            questions=questions,
            total_time_minutes=int(total_time),
            difficulty_level=learner_profile.get("category", "intermediate"),
            topic=request.topic,
            generation_metadata={
                "learner_profile": learner_profile,
                "generation_time": datetime.now().isoformat(),
                "strategy_used": learner_profile.get("category_prompt", ""),
                "delivery_method": request.strategy_context.get("strategy_recommendation", {}).get("strategy_label", ""),
                "langchain_used": True
            },
            created_at=datetime.now().isoformat()
        )
        
        print(f"✅ Generated quiz with {len(questions)} questions using LangChain")
        return quiz
    
    def _extract_learner_profile(self, strategy_context: Dict) -> Dict:
        """Extract and process learner profile information"""
        learner_profile = strategy_context.get("learner_profile", {})
        strategy_rec = strategy_context.get("strategy_recommendation", {})
        teaching_approach = strategy_context.get("combined_output", {}).get("teaching_approach", {})
        
        return {
            "category": learner_profile.get("category", "intermediate"),
            "category_prompt": learner_profile.get("category_prompt", ""),
            "confidence_score": learner_profile.get("confidence_score", 0.5),
            "redo_topics_flag": learner_profile.get("redo_topics_flag", False),
            "delivery_method": strategy_rec.get("strategy_label", ""),
            "challenge_level": teaching_approach.get("challenge_level", "moderate"),
            "support_level": teaching_approach.get("support_level", "moderate"),
            "pacing": teaching_approach.get("pacing", "moderate")
        }
    
    def _fetch_knowledge_context(self, topic: str, focus_areas: List[str] = None) -> str:
        """Fetch relevant knowledge from SME system"""
        try:
            # Build query based on topic and focus areas
            query_parts = [topic]
            if focus_areas:
                query_parts.extend(focus_areas)
            query = " ".join(query_parts)
            
            # Call SME system API
            response = requests.post(f"{self.sme_endpoint}/query", 
                                   json={"query": query, "max_results": 5}, 
                                   timeout=30)
            
            if response.status_code == 200:
                return response.json().get("content", "")
            else:
                print(f"⚠️ SME system unavailable, using topic name only")
                return topic
                
        except Exception as e:
            print(f"⚠️ Error fetching knowledge context: {e}")
            return topic
    
    def _generate_questions_with_langchain(self, request: QuizGenerationRequest, 
                                         learner_profile: Dict, knowledge_context: str) -> List[QuizQuestion]:
        """Generate quiz questions using LangChain"""
        
        # Get profile-specific settings
        profile_category = learner_profile.get("category", "intermediate")
        template = self.profile_templates.get(profile_category, self.profile_templates["Rising Improver"])
        
        # Determine question types and distribution
        question_types = request.question_types or list(template["question_distribution"].keys())
        num_questions = request.num_questions
        
        # Prepare inputs for LangChain chain
        chain_inputs = {
            "topic": request.topic,
            "num_questions": num_questions,
            "learner_category": profile_category,
            "category_prompt": learner_profile.get("category_prompt", ""),
            "delivery_strategy": request.strategy_context.get("strategy_recommendation", {}).get("strategy_prompt", ""),
            "question_types": ', '.join(question_types),
            "difficulty_level": template["difficulty_level"],
            "explanation_detail": template["explanation_detail"],
            "encouragement_tone": template["encouragement_tone"],
            "knowledge_context": knowledge_context,
            "challenge_level": learner_profile.get("challenge_level", "moderate"),
            "support_level": learner_profile.get("support_level", "moderate"),
            "redo_topics_flag": learner_profile.get("redo_topics_flag", False)
        }
        
        try:
            print("🤖 Generating questions with LangChain...")
            # Run the LangChain chain
            questions_data = self.quiz_chain.invoke(chain_inputs)
            
            # Convert to QuizQuestion objects
            questions = []
            for i, q_data in enumerate(questions_data[:num_questions]):
                question = QuizQuestion(
                    question_id=f"q_{int(time.time())}_{i}",
                    question_type=q_data.get("type", "multiple_choice"),
                    question_text=q_data.get("question", ""),
                    options=q_data.get("options", []),
                    correct_answer=q_data.get("answer", ""),
                    explanation=q_data.get("explanation", ""),
                    difficulty=template["difficulty_level"],
                    topic_tags=[request.topic] + (request.focus_areas or []),
                    estimated_time_minutes=q_data.get("estimated_time", 2.0)
                )
                questions.append(question)
            
            return questions
            
        except Exception as e:
            print(f"❌ Error with LangChain generation: {e}")
            return self._generate_fallback_questions(request.topic, num_questions)
    
    def _generate_fallback_questions(self, topic: str, num_questions: int) -> List[QuizQuestion]:
        """Generate basic fallback questions if LangChain fails"""
        questions = []
        for i in range(num_questions):
            question = QuizQuestion(
                question_id=f"fallback_{int(time.time())}_{i}",
                question_type="multiple_choice",
                question_text=f"Question about {topic} (Question {i+1})",
                options=["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                correct_answer="A) Option 1",
                explanation=f"This is a fallback question for {topic}",
                difficulty="intermediate",
                topic_tags=[topic]
            )
            questions.append(question)
        return questions
    
    def _generate_quiz_description(self, request: QuizGenerationRequest, 
                                 learner_profile: Dict) -> str:
        """Generate personalized quiz description"""
        category = learner_profile.get("category", "Learner")
        delivery = learner_profile.get("delivery_method", "Standard")
        
        if learner_profile.get("redo_topics_flag"):
            context = "This quiz helps reinforce and clarify concepts you've studied before."
        else:
            context = "This quiz introduces new concepts to advance your learning."
        
        return f"""Personalized quiz for {category} learners using {delivery} approach.
{context}
Topic: {request.topic}
Questions: {request.num_questions}
Generated with LangChain for enhanced personalization."""

    def export_quiz(self, quiz: GeneratedQuiz, format: str = "json") -> str:
        """Export quiz in various formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "json":
            filename = f"quiz_{quiz.quiz_id}_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(asdict(quiz), f, indent=2, ensure_ascii=False)
        
        return filename

# API Server (Flask-based microservice)
def create_quiz_service_app():
    """Create Flask app for LangChain Quiz Head Service"""
    app = Flask(__name__)
    quiz_service = QuizHeadService()
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "healthy", 
            "service": "quiz_head", 
            "framework": "langchain",
            "llm": "qwen3:4b"
        })
    
    @app.route('/generate_quiz', methods=['POST'])
    def generate_quiz():
        try:
            data = request.json
            
            # Create request object
            quiz_request = QuizGenerationRequest(
                strategy_context=data.get('strategy_context', {}),
                topic=data.get('topic', ''),
                num_questions=data.get('num_questions', 5),
                question_types=data.get('question_types'),
                difficulty_override=data.get('difficulty_override'),
                focus_areas=data.get('focus_areas'),
                time_limit_minutes=data.get('time_limit_minutes')
            )
            
            # Generate quiz using LangChain
            quiz = quiz_service.generate_quiz(quiz_request)
            
            return jsonify({
                "status": "success",
                "framework": "langchain",
                "quiz": asdict(quiz)
            })
            
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
    
    @app.route('/export_quiz', methods=['POST'])
    def export_quiz():
        try:
            data = request.json
            quiz_data = data.get('quiz')
            format_type = data.get('format', 'json')
            
            # Convert dict back to GeneratedQuiz object
            quiz = GeneratedQuiz(**quiz_data)
            filename = quiz_service.export_quiz(quiz, format_type)
            
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
    app = create_quiz_service_app()
    print("🎯 LangChain Quiz Head Service starting on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=True)