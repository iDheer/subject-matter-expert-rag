#!/usr/bin/env python3
"""
RTX 4050 6GB Specific VRAM Analysis
"""

def analyze_rtx_4050_compatibility():
    """Analyze compatibility with RTX 4050 6GB"""
    
    gpu_memory = 6.0  # GB
    safe_threshold = gpu_memory * 0.85  # 85% usage threshold for stability
    
    systems = {
        'SME System': {
            'fp16': 10.1,
            'int8': 6.6,
            'int4': 4.9
        },
        'Knowledge Graph': {
            'fp16': 9.2,
            'int8': 5.7,
            'int4': 4.0
        }
    }
    
    print("🎯 RTX 4050 6GB VRAM Compatibility Analysis")
    print("=" * 60)
    print(f"GPU Memory: {gpu_memory} GB")
    print(f"Safe Usage Threshold: {safe_threshold} GB (85%)")
    print()
    
    for system_name, configs in systems.items():
        print(f"🔹 {system_name}:")
        
        for precision, required in configs.items():
            status = ""
            recommendation = ""
            
            if required <= safe_threshold:
                status = "✅ EXCELLENT"
                recommendation = f"Fits comfortably with {safe_threshold - required:.1f} GB headroom"
            elif required <= gpu_memory:
                status = "⚠️  TIGHT FIT"
                recommendation = f"May work but could cause stability issues. Only {gpu_memory - required:.1f} GB free"
            else:
                status = "❌ WON'T FIT"
                recommendation = f"Exceeds GPU memory by {required - gpu_memory:.1f} GB"
            
            print(f"   {precision.upper()}: {required} GB - {status}")
            print(f"      → {recommendation}")
        print()
    
    print("📋 Summary for RTX 4050 6GB:")
    print("-" * 40)
    print("✅ RECOMMENDED CONFIGURATIONS:")
    print("   • SME System: INT4 quantization (4.9 GB)")
    print("   • Knowledge Graph: INT8 quantization (5.7 GB) or INT4 (4.0 GB)")
    print()
    print("❌ NOT RECOMMENDED:")
    print("   • Any FP16 configuration (too much VRAM)")
    print("   • SME System with INT8 (6.6 GB - too close to limit)")
    print()
    print("💡 OPTIMIZATION STRATEGIES:")
    print("   1. Use Ollama with quantization: 'qwen3:4b-q4_0' or 'qwen3:4b-q8_0'")
    print("   2. Monitor GPU memory: nvidia-smi -l 1")
    print("   3. Run systems separately, not simultaneously")
    print("   4. Clear GPU cache between runs: torch.cuda.empty_cache()")
    print("   5. Consider CPU fallback for embedding models if needed")

if __name__ == "__main__":
    analyze_rtx_4050_compatibility()
