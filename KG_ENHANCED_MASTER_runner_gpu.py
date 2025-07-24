#!/usr/bin/env python3
"""
GPU-Accelerated Enhanced Chapter-Based Knowledge Graph Master Runner
Comprehensive orchestration system for the enhanced chapter-based RAG system
"""
import os
import sys
import time
import json
import logging
import subprocess
from typing import Dict, List, Optional
from datetime import datetime
import psutil
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GPUAcceleratedEnhancedSystemRunner:
    """Master orchestration system for the enhanced chapter-based RAG system"""
    
    def __init__(self):
        self.setup_gpu_info()
        self.setup_system_info()
        self.check_dependencies()
        self.pipeline_steps = [
            {
                'name': 'Enhanced Chapter Database',
                'script': 'KG_ENHANCED_1_build_chapter_database_gpu.py',
                'description': 'Build enhanced chapter-based database with 20-30 learning concepts per chapter',
                'estimated_time': 300,  # 5 minutes
                'required': True
            },
            {
                'name': 'Enhanced Knowledge Graph',
                'script': 'KG_ENHANCED_2_build_knowledge_graph_gpu.py', 
                'description': 'Build enhanced knowledge graph with concept clusters and GPU acceleration',
                'estimated_time': 600,  # 10 minutes
                'required': True
            },
            {
                'name': 'Enhanced Query System',
                'script': 'KG_ENHANCED_3_query_knowledge_graph_gpu.py',
                'description': 'Interactive enhanced query system with concept clusters',
                'estimated_time': 60,   # 1 minute
                'required': False
            },
            {
                'name': 'Enhanced Visualization',
                'script': 'KG_ENHANCED_4_visualize_knowledge_graph_gpu.py',
                'description': 'Advanced visualization system with 3D clusters and analytics',
                'estimated_time': 120,  # 2 minutes
                'required': False
            }
        ]
    
    def setup_gpu_info(self):
        """Setup GPU information"""
        if torch.cuda.is_available():
            self.gpu_available = True
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 GPU Available: {self.gpu_name} ({self.gpu_memory:.1f} GB)")
        else:
            self.gpu_available = False
            logger.warning("⚠️ No GPU available - system will use CPU (slower)")
    
    def setup_system_info(self):
        """Setup system information"""
        self.system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024**3,
            'disk_free_gb': psutil.disk_usage('.').free / 1024**3,
            'python_version': sys.version.split()[0],
            'platform': sys.platform
        }
        
        logger.info(f"💻 System: {self.system_info['cpu_count']} CPUs, {self.system_info['memory_gb']:.1f} GB RAM")
    
    def check_dependencies(self):
        """Check for required dependencies and services"""
        logger.info("🔍 Checking dependencies and services...")
        
        self.dependencies_status = {}
        
        # Check Python packages
        required_packages = [
            'torch', 'transformers', 'sentence-transformers', 'elasticsearch',
            'neo4j', 'llama-index', 'networkx', 'plotly', 'scikit-learn',
            'pandas', 'numpy', 'nltk', 'matplotlib', 'seaborn'
        ]
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                self.dependencies_status[package] = True
            except ImportError:
                self.dependencies_status[package] = False
                logger.warning(f"⚠️ Missing package: {package}")
        
        # Check services
        self.services_status = {}
        
        # Check Elasticsearch
        try:
            import requests
            response = requests.get('http://localhost:9200', timeout=5)
            self.services_status['elasticsearch'] = response.status_code == 200
        except:
            self.services_status['elasticsearch'] = False
        
        # Check Neo4j
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "knowledge123"))
            with driver.session() as session:
                session.run("RETURN 1")
            driver.close()
            self.services_status['neo4j'] = True
        except:
            self.services_status['neo4j'] = False
        
        # Check Ollama
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            self.services_status['ollama'] = response.status_code == 200
        except:
            self.services_status['ollama'] = False
        
        # Report status
        for service, status in self.services_status.items():
            if status:
                logger.info(f"✅ {service.title()} is running")
            else:
                logger.error(f"❌ {service.title()} is not available")
    
    def display_system_status(self):
        """Display comprehensive system status"""
        print("\n" + "="*80)
        print("🚀 GPU-ACCELERATED ENHANCED CHAPTER-BASED KNOWLEDGE GRAPH SYSTEM")
        print("="*80)
        
        print(f"\n💻 SYSTEM INFORMATION:")
        print(f"   Platform: {self.system_info['platform']}")
        print(f"   Python: {self.system_info['python_version']}")
        print(f"   CPUs: {self.system_info['cpu_count']}")
        print(f"   Memory: {self.system_info['memory_gb']:.1f} GB")
        print(f"   Disk Free: {self.system_info['disk_free_gb']:.1f} GB")
        
        print(f"\n🚀 GPU INFORMATION:")
        if self.gpu_available:
            print(f"   GPU: {self.gpu_name}")
            print(f"   Memory: {self.gpu_memory:.1f} GB")
            print(f"   CUDA: {torch.version.cuda}")
            print(f"   Status: ✅ GPU acceleration enabled")
        else:
            print(f"   Status: ⚠️ No GPU available (CPU mode)")
        
        print(f"\n🛠️ SERVICES STATUS:")
        for service, status in self.services_status.items():
            status_icon = "✅" if status else "❌"
            print(f"   {service.title()}: {status_icon}")
        
        print(f"\n📦 DEPENDENCIES STATUS:")
        missing_deps = [pkg for pkg, status in self.dependencies_status.items() if not status]
        if missing_deps:
            print(f"   Missing packages: {', '.join(missing_deps)}")
        else:
            print(f"   All dependencies: ✅")
        
        print(f"\n📋 PIPELINE STEPS:")
        total_time = sum(step['estimated_time'] for step in self.pipeline_steps)
        print(f"   Total estimated time: {total_time//60} minutes {total_time%60} seconds")
        
        for i, step in enumerate(self.pipeline_steps, 1):
            required_text = "Required" if step['required'] else "Optional"
            print(f"   {i}. {step['name']} ({step['estimated_time']//60}m {step['estimated_time']%60}s) - {required_text}")
            print(f"      {step['description']}")
        
        print("="*80)
    
    def install_missing_dependencies(self):
        """Install missing Python dependencies"""
        missing_deps = [pkg for pkg, status in self.dependencies_status.items() if not status]
        
        if not missing_deps:
            logger.info("✅ All dependencies are already installed")
            return True
        
        logger.info(f"📦 Installing missing dependencies: {', '.join(missing_deps)}")
        
        for package in missing_deps:
            try:
                logger.info(f"Installing {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                self.dependencies_status[package] = True
                logger.info(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {e}")
                return False
        
        return True
    
    def start_services(self):
        """Start required services if not running"""
        logger.info("🔧 Checking and starting required services...")
        
        # Start Elasticsearch if not running
        if not self.services_status.get('elasticsearch', False):
            logger.info("Starting Elasticsearch...")
            # Note: This assumes docker-compose is available
            try:
                subprocess.run(['docker-compose', '-f', 'docker-compose-elasticsearch.yml', 'up', '-d'], 
                             check=True, capture_output=True)
                time.sleep(30)  # Wait for startup
                self.services_status['elasticsearch'] = True
                logger.info("✅ Elasticsearch started")
            except subprocess.CalledProcessError:
                logger.error("❌ Failed to start Elasticsearch")
        
        # Check Neo4j (assume it's managed separately)
        if not self.services_status.get('neo4j', False):
            logger.warning("⚠️ Neo4j is not running. Please start it manually.")
        
        # Check Ollama
        if not self.services_status.get('ollama', False):
            logger.warning("⚠️ Ollama is not running. Please start it manually.")
    
    def run_pipeline_step(self, step: Dict) -> bool:
        """Run a single pipeline step"""
        logger.info(f"🚀 Running: {step['name']}")
        logger.info(f"📝 Description: {step['description']}")
        logger.info(f"⏱️ Estimated time: {step['estimated_time']//60}m {step['estimated_time']%60}s")
        
        script_path = step['script']
        if not os.path.exists(script_path):
            logger.error(f"❌ Script not found: {script_path}")
            return False
        
        start_time = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=step['estimated_time'] * 2  # Double the estimated time for timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {step['name']} completed successfully in {execution_time:.1f}s")
                
                # Log any output if verbose
                if result.stdout:
                    logger.debug(f"Output: {result.stdout[-500:]}")  # Last 500 chars
                
                return True
            else:
                logger.error(f"❌ {step['name']} failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {step['name']} timed out after {step['estimated_time'] * 2}s")
            return False
        except Exception as e:
            logger.error(f"❌ {step['name']} failed with exception: {e}")
            return False
    
    def run_full_pipeline(self, skip_optional: bool = False):
        """Run the complete pipeline"""
        logger.info("🎯 Starting enhanced chapter-based knowledge graph pipeline...")
        
        start_time = time.time()
        completed_steps = []
        failed_steps = []
        
        for step in self.pipeline_steps:
            if skip_optional and not step['required']:
                logger.info(f"⏭️ Skipping optional step: {step['name']}")
                continue
            
            success = self.run_pipeline_step(step)
            
            if success:
                completed_steps.append(step['name'])
            else:
                failed_steps.append(step['name'])
                
                if step['required']:
                    logger.error(f"💥 Required step failed: {step['name']}")
                    logger.error("🛑 Pipeline aborted due to required step failure")
                    break
                else:
                    logger.warning(f"⚠️ Optional step failed: {step['name']} - continuing...")
        
        total_time = time.time() - start_time
        
        # Generate summary
        logger.info(f"\n📊 PIPELINE SUMMARY:")
        logger.info(f"   Total execution time: {total_time//60:.0f}m {total_time%60:.0f}s")
        logger.info(f"   Completed steps: {len(completed_steps)}")
        logger.info(f"   Failed steps: {len(failed_steps)}")
        
        if completed_steps:
            logger.info(f"   ✅ Completed: {', '.join(completed_steps)}")
        
        if failed_steps:
            logger.info(f"   ❌ Failed: {', '.join(failed_steps)}")
        
        return len(failed_steps) == 0 or all(not step['required'] for step in self.pipeline_steps if step['name'] in failed_steps)
    
    def run_specific_step(self, step_name: str):
        """Run a specific pipeline step by name"""
        step = next((s for s in self.pipeline_steps if s['name'].lower() == step_name.lower()), None)
        
        if not step:
            logger.error(f"❌ Step not found: {step_name}")
            logger.info(f"Available steps: {', '.join([s['name'] for s in self.pipeline_steps])}")
            return False
        
        return self.run_pipeline_step(step)
    
    def create_system_report(self):
        """Create comprehensive system report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.system_info,
            'gpu_info': {
                'available': self.gpu_available,
                'name': self.gpu_name if self.gpu_available else None,
                'memory_gb': self.gpu_memory if self.gpu_available else None
            },
            'services_status': self.services_status,
            'dependencies_status': self.dependencies_status,
            'pipeline_steps': self.pipeline_steps
        }
        
        report_path = f"enhanced_system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 System report saved to: {report_path}")
        return report_path
    
    def interactive_mode(self):
        """Interactive command-line interface for the enhanced system"""
        self.display_system_status()
        
        print(f"\n🤖 ENHANCED SYSTEM INTERACTIVE MODE")
        print("Available commands:")
        print("  full                - Run complete pipeline")
        print("  required            - Run only required steps")
        print("  step <name>         - Run specific step")
        print("  status              - Show system status")
        print("  install-deps        - Install missing dependencies")
        print("  start-services      - Start required services")
        print("  report              - Generate system report")
        print("  list-steps          - List all pipeline steps")
        print("  query               - Launch query system (if built)")
        print("  visualize           - Launch visualization system (if built)")
        print("  quit                - Exit")
        print("-" * 80)
        
        while True:
            try:
                user_input = input("\n🎯 Enter command: ").strip().lower()
                
                if user_input in ['quit', 'exit', 'q']:
                    break
                
                parts = user_input.split()
                command = parts[0]
                
                if command == 'full':
                    print("\n🚀 Running complete pipeline...")
                    success = self.run_full_pipeline(skip_optional=False)
                    if success:
                        print("✅ Pipeline completed successfully!")
                    else:
                        print("❌ Pipeline completed with errors")
                
                elif command == 'required':
                    print("\n🚀 Running required steps only...")
                    success = self.run_full_pipeline(skip_optional=True)
                    if success:
                        print("✅ Required steps completed successfully!")
                    else:
                        print("❌ Required steps completed with errors")
                
                elif command == 'step':
                    if len(parts) < 2:
                        print("❌ Please specify step name")
                        continue
                    
                    step_name = ' '.join(parts[1:])
                    success = self.run_specific_step(step_name)
                    if success:
                        print(f"✅ Step '{step_name}' completed successfully!")
                    else:
                        print(f"❌ Step '{step_name}' failed")
                
                elif command == 'status':
                    self.display_system_status()
                
                elif command == 'install-deps':
                    success = self.install_missing_dependencies()
                    if success:
                        print("✅ Dependencies installed successfully!")
                    else:
                        print("❌ Failed to install some dependencies")
                
                elif command == 'start-services':
                    self.start_services()
                
                elif command == 'report':
                    report_path = self.create_system_report()
                    print(f"✅ System report generated: {report_path}")
                
                elif command == 'list-steps':
                    print(f"\n📋 Available Pipeline Steps:")
                    for i, step in enumerate(self.pipeline_steps, 1):
                        required_text = "Required" if step['required'] else "Optional"
                        print(f"  {i}. {step['name']} ({required_text})")
                        print(f"     {step['description']}")
                
                elif command == 'query':
                    if os.path.exists('KG_ENHANCED_3_query_knowledge_graph_gpu.py'):
                        print("🚀 Launching query system...")
                        subprocess.run([sys.executable, 'KG_ENHANCED_3_query_knowledge_graph_gpu.py'])
                    else:
                        print("❌ Query system not found. Run the pipeline first.")
                
                elif command == 'visualize':
                    if os.path.exists('KG_ENHANCED_4_visualize_knowledge_graph_gpu.py'):
                        print("🚀 Launching visualization system...")
                        subprocess.run([sys.executable, 'KG_ENHANCED_4_visualize_knowledge_graph_gpu.py'])
                    else:
                        print("❌ Visualization system not found. Run the pipeline first.")
                
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def quick_setup(self):
        """Quick setup for first-time users"""
        print("\n🎯 ENHANCED KNOWLEDGE GRAPH QUICK SETUP")
        print("=" * 50)
        
        # Check system requirements
        if self.system_info['memory_gb'] < 8:
            logger.warning("⚠️ Less than 8GB RAM detected. System may be slow.")
        
        if not self.gpu_available:
            logger.warning("⚠️ No GPU detected. Processing will be slower.")
        
        # Install dependencies
        print("\n1️⃣ Installing missing dependencies...")
        if not self.install_missing_dependencies():
            print("❌ Failed to install dependencies. Please install manually.")
            return False
        
        # Start services
        print("\n2️⃣ Starting services...")
        self.start_services()
        
        # Run pipeline
        print("\n3️⃣ Running pipeline...")
        success = self.run_full_pipeline(skip_optional=False)
        
        if success:
            print("\n✅ SETUP COMPLETE!")
            print("You can now use:")
            print("  - Query system: python NEW_6_query_enhanced_knowledge_graph_gpu.py")
            print("  - Visualization: python NEW_7_visualize_enhanced_knowledge_graph_gpu.py")
        else:
            print("\n❌ Setup completed with errors. Check logs for details.")
        
        return success

def main():
    """Main execution function"""
    runner = GPUAcceleratedEnhancedSystemRunner()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'setup':
            runner.quick_setup()
        elif command == 'full':
            runner.run_full_pipeline(skip_optional=False)
        elif command == 'required':
            runner.run_full_pipeline(skip_optional=True)
        elif command == 'status':
            runner.display_system_status()
        elif command == 'report':
            runner.create_system_report()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: setup, full, required, status, report")
    else:
        runner.interactive_mode()

if __name__ == "__main__":
    main()
