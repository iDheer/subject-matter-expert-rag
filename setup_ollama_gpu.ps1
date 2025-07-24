# PowerShell script to configure Ollama for GPU usage
Write-Host "🔧 Configuring Ollama for GPU usage..." -ForegroundColor Yellow

# Set environment variables for GPU usage
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OLLAMA_NUM_GPU = "1"
$env:OLLAMA_GPU_LAYERS = "35"
$env:OLLAMA_MAX_LOADED_MODELS = "1"

Write-Host "Environment variables set:" -ForegroundColor Green
Write-Host "  CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES" -ForegroundColor Cyan
Write-Host "  OLLAMA_NUM_GPU=$env:OLLAMA_NUM_GPU" -ForegroundColor Cyan
Write-Host "  OLLAMA_GPU_LAYERS=$env:OLLAMA_GPU_LAYERS" -ForegroundColor Cyan
Write-Host "  OLLAMA_MAX_LOADED_MODELS=$env:OLLAMA_MAX_LOADED_MODELS" -ForegroundColor Cyan

Write-Host ""
Write-Host "🛑 Stopping existing Ollama service..." -ForegroundColor Yellow
try {
    Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
} catch {
    Write-Host "No existing Ollama process found" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🚀 Starting Ollama with GPU configuration..." -ForegroundColor Yellow
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Normal

Write-Host ""
Write-Host "⏳ Waiting for Ollama to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "🧪 Testing Ollama connection..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:11434/api/version" -TimeoutSec 10
    Write-Host "✅ Ollama is running (version: $($response.version))" -ForegroundColor Green
} catch {
    Write-Host "❌ Could not connect to Ollama" -ForegroundColor Red
    Write-Host "Please check if Ollama started correctly" -ForegroundColor Red
}

Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "  - Monitor GPU usage with: nvidia-smi" -ForegroundColor Cyan
Write-Host "  - Check GPU memory usage while running queries" -ForegroundColor Cyan
Write-Host "  - The qwen3:4b model should now load in GPU memory" -ForegroundColor Cyan

Write-Host ""
Write-Host "🎉 Ollama GPU configuration complete!" -ForegroundColor Green
Write-Host "You can now run your query script: python SME_2_query_elasticsearch_system.py" -ForegroundColor Green
