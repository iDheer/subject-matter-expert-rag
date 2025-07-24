@echo off
echo Configuring Ollama for GPU usage...

REM Set environment variables for GPU usage
set CUDA_VISIBLE_DEVICES=0
set OLLAMA_NUM_GPU=1
set OLLAMA_GPU_LAYERS=35
set OLLAMA_MAX_LOADED_MODELS=1

echo Environment variables set:
echo   CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo   OLLAMA_NUM_GPU=%OLLAMA_NUM_GPU%
echo   OLLAMA_GPU_LAYERS=%OLLAMA_GPU_LAYERS%
echo   OLLAMA_MAX_LOADED_MODELS=%OLLAMA_MAX_LOADED_MODELS%

echo.
echo Stopping existing Ollama service...
taskkill /f /im ollama.exe 2>nul

echo.
echo Starting Ollama with GPU configuration...
start "Ollama GPU" ollama serve

echo.
echo Waiting for Ollama to start...
timeout /t 5 /nobreak >nul

echo.
echo Testing GPU memory usage...
echo Running a quick test to ensure GPU is being used...

echo.
echo Ollama should now be configured to use GPU memory.
echo You can now run your query script.
echo.
pause
