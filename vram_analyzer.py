#!/usr/bin/env python3
"""
VRAM Usage Analysis for SME and Knowledge Graph Systems
"""
import torch
import json
from typing import Dict, List

class VRAMAnalyzer:
    """Analyze VRAM usage for different AI components"""
    
    def __init__(self):
        self.model_sizes = {
            # LLM Models (approximate VRAM usage)
            'qwen3:4b': {
                'parameters': '4B',
                'fp16_vram': 8.0,  # GB - 4B params * 2 bytes (fp16)
                'int8_vram': 4.5,  # GB - quantized
                'int4_vram': 2.8,  # GB - heavily quantized
                'description': 'Qwen 4B language model'
            },
            
            # Embedding Models
            'sentence-transformers/all-mpnet-base-v2': {
                'parameters': '110M',
                'fp16_vram': 0.4,  # GB
                'fp32_vram': 0.8,  # GB
                'description': 'MPNET embedding model'
            },
            
            # Reranking Models
            'BAAI/bge-reranker-v2-m3': {
                'parameters': '560M',
                'fp16_vram': 1.2,  # GB
                'fp32_vram': 2.4,  # GB
                'description': 'BGE reranker model'
            },
            
            # Additional components overhead
            'pytorch_overhead': {
                'vram': 0.5,  # GB - PyTorch CUDA overhead
                'description': 'PyTorch CUDA memory overhead'
            },
            
            'sentence_transformer_overhead': {
                'vram': 0.3,  # GB - SentenceTransformer overhead
                'description': 'SentenceTransformer GPU overhead'
            }
        }
        
        self.system_configs = {
            'SME_System': {
                'components': [
                    'qwen3:4b',
                    'sentence-transformers/all-mpnet-base-v2',
                    'BAAI/bge-reranker-v2-m3',
                    'pytorch_overhead'
                ],
                'description': 'Subject Matter Expert RAG System'
            },
            
            'Knowledge_Graph': {
                'components': [
                    'qwen3:4b',
                    'sentence-transformers/all-mpnet-base-v2',
                    'sentence_transformer_overhead',  # Additional SentenceTransformer instance
                    'pytorch_overhead'
                ],
                'description': 'Enhanced Knowledge Graph System'
            }
        }
    
    def calculate_system_vram(self, system_name: str, precision: str = 'fp16') -> Dict:
        """Calculate VRAM usage for a system"""
        if system_name not in self.system_configs:
            raise ValueError(f"Unknown system: {system_name}")
        
        config = self.system_configs[system_name]
        total_vram = 0.0
        component_details = []
        
        for component in config['components']:
            if component not in self.model_sizes:
                continue
                
            model_info = self.model_sizes[component]
            
            # Get VRAM usage based on precision
            if precision + '_vram' in model_info:
                vram = model_info[precision + '_vram']
            elif 'vram' in model_info:
                vram = model_info['vram']
            else:
                vram = model_info.get('fp16_vram', 0.0)
            
            total_vram += vram
            component_details.append({
                'component': component,
                'vram_gb': vram,
                'description': model_info['description']
            })
        
        return {
            'system': system_name,
            'total_vram_gb': round(total_vram, 2),
            'precision': precision,
            'components': component_details,
            'description': config['description']
        }
    
    def get_gpu_info(self):
        """Get GPU information"""
        if not torch.cuda.is_available():
            return {
                'cuda_available': False,
                'message': 'CUDA not available'
            }
        
        return {
            'cuda_available': True,
            'gpu_count': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0),
            'total_memory_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            'current_allocated_gb': round(torch.cuda.memory_allocated(0) / (1024**3), 2),
            'current_cached_gb': round(torch.cuda.memory_reserved(0) / (1024**3), 2)
        }
    
    def analyze_all_systems(self) -> Dict:
        """Analyze VRAM usage for all systems"""
        results = {
            'gpu_info': self.get_gpu_info(),
            'systems': {}
        }
        
        precisions = ['fp16', 'int8', 'int4'] if 'qwen3:4b' in str(self.model_sizes) else ['fp16']
        
        for system_name in self.system_configs.keys():
            results['systems'][system_name] = {}
            for precision in precisions:
                try:
                    analysis = self.calculate_system_vram(system_name, precision)
                    results['systems'][system_name][precision] = analysis
                except:
                    continue
        
        return results
    
    def print_analysis(self):
        """Print detailed VRAM analysis"""
        results = self.analyze_all_systems()
        
        print("🔍 VRAM Usage Analysis for SME and Knowledge Graph Systems")
        print("=" * 80)
        
        # GPU Info
        gpu_info = results['gpu_info']
        if gpu_info['cuda_available']:
            print(f"\n🖥️  GPU Information:")
            print(f"   GPU: {gpu_info['gpu_name']}")
            print(f"   Total VRAM: {gpu_info['total_memory_gb']} GB")
            print(f"   Currently Allocated: {gpu_info['current_allocated_gb']} GB")
            print(f"   Currently Cached: {gpu_info['current_cached_gb']} GB")
        else:
            print(f"\n⚠️  {gpu_info['message']}")
        
        print(f"\n📊 System VRAM Requirements:")
        print("-" * 80)
        
        # System Analysis
        for system_name, system_data in results['systems'].items():
            print(f"\n🔹 {system_name}")
            
            for precision, analysis in system_data.items():
                print(f"   {precision.upper()}: {analysis['total_vram_gb']} GB")
                
                for comp in analysis['components']:
                    print(f"     • {comp['component']}: {comp['vram_gb']} GB - {comp['description']}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        print("-" * 40)
        
        if gpu_info['cuda_available']:
            gpu_memory = gpu_info['total_memory_gb']
            
            for system_name, system_data in results['systems'].items():
                print(f"\n🔹 {system_name}:")
                
                for precision, analysis in system_data.items():
                    required = analysis['total_vram_gb']
                    if required <= gpu_memory * 0.8:  # 80% usage threshold
                        print(f"   ✅ {precision.upper()}: {required} GB - FITS comfortably")
                    elif required <= gpu_memory:
                        print(f"   ⚠️  {precision.upper()}: {required} GB - Tight fit, may cause issues")
                    else:
                        print(f"   ❌ {precision.upper()}: {required} GB - Exceeds GPU memory")
        
        # Memory optimization tips
        print(f"\n🛠️  Memory Optimization Tips:")
        print("-" * 40)
        print("1. Use INT8 or INT4 quantization for the LLM to reduce VRAM usage")
        print("2. Run only one system at a time to avoid memory conflicts")
        print("3. Clear GPU cache between switching systems: torch.cuda.empty_cache()")
        print("4. Consider using gradient checkpointing for large models")
        print("5. Monitor GPU memory usage with: nvidia-smi")
        
        return results

def main():
    analyzer = VRAMAnalyzer()
    results = analyzer.print_analysis()
    
    # Save results to file
    with open('vram_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Detailed analysis saved to: vram_analysis.json")

if __name__ == "__main__":
    main()
