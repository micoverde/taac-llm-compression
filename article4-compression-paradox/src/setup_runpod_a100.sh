#!/usr/bin/env bash
# =============================================================================
# Article 4: RunPod A100 Setup — GPU Energy Proxy Validation
# =============================================================================
# Target: RunPod A100 80GB SXM4
# Purpose: Validate energy proxy formula with direct NVML measurements
#
# Usage:
#   1. Create RunPod pod with A100 80GB + pytorch template
#   2. SSH in: ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
#   3. Run: bash setup_runpod_a100.sh
#   4. Start vLLM: tmux new -s vllm './start_vllm_llama.sh'
#   5. Run experiment: tmux new -s exp 'python runpod_gpu_validation.py --model llama'
#   6. Switch model: stop vLLM, './start_vllm_mistral.sh', re-run with --model mistral
# =============================================================================

set -euo pipefail

echo "================================================================"
echo "Article 4: GPU Energy Proxy Validation — RunPod Setup"
echo "================================================================"

# ---------------------------------------------------------------------------
# 1. System check
# ---------------------------------------------------------------------------
echo "[1/6] Verifying GPU..."
nvidia-smi --query-gpu=name,memory.total,power.max_limit --format=csv,noheader
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME"

# ---------------------------------------------------------------------------
# 2. Install Python dependencies
# ---------------------------------------------------------------------------
echo "[2/6] Installing Python dependencies..."
pip install --quiet \
    vllm>=0.6.0 \
    pynvml>=11.5.0 \
    aiohttp>=3.11.0 \
    datasets>=3.0.0 \
    llmlingua>=0.2.0 \
    transformers>=4.45.0 \
    huggingface_hub>=0.27.0 \
    numpy scipy

# ---------------------------------------------------------------------------
# 3. Validate pynvml
# ---------------------------------------------------------------------------
echo "[3/6] Validating power monitoring..."
python3 -c "
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
name = pynvml.nvmlDeviceGetName(h)
power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
print(f'GPU: {name}')
print(f'Power: {power:.1f}W')
print(f'Temp: {temp}C')
print('pynvml: OK')
pynvml.nvmlShutdown()
"

# ---------------------------------------------------------------------------
# 4. Create vLLM launch scripts
# ---------------------------------------------------------------------------
echo "[4/6] Creating vLLM launch scripts..."

cat > start_vllm_llama.sh << 'EOF'
#!/usr/bin/env bash
echo "Starting vLLM with Llama-3.1-8B-Instruct..."
python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --host 0.0.0.0
EOF
chmod +x start_vllm_llama.sh

cat > start_vllm_mistral.sh << 'EOF'
#!/usr/bin/env bash
echo "Starting vLLM with Mistral-7B-Instruct-v0.3..."
python3 -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --host 0.0.0.0
EOF
chmod +x start_vllm_mistral.sh

# ---------------------------------------------------------------------------
# 5. Create results directory
# ---------------------------------------------------------------------------
echo "[5/6] Creating results directory..."
mkdir -p results

# ---------------------------------------------------------------------------
# 6. Validate LLMLingua-2
# ---------------------------------------------------------------------------
echo "[6/6] Validating LLMLingua-2 compressor..."
python3 -c "
from llmlingua import PromptCompressor
c = PromptCompressor(
    model_name='microsoft/llmlingua-2-xlm-roberta-large-meetingbank',
    use_llmlingua2=True,
    device_map='cpu',
)
result = c.compress_prompt('This is a test prompt for compression validation.', rate=0.5)
print(f'Compressed: {result[\"compressed_prompt\"][:50]}...')
print('LLMLingua-2: OK')
" 2>/dev/null || echo "WARNING: LLMLingua-2 not available, will use fallback compression"

echo ""
echo "================================================================"
echo "Setup complete!"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  1. Start vLLM:  tmux new -s vllm './start_vllm_llama.sh'"
echo "  2. Wait for model load (~2-3 min)"
echo "  3. Verify:      curl http://localhost:8000/v1/models"
echo "  4. Pilot run:   python3 runpod_gpu_validation.py --model llama --subset 20"
echo "  5. Full run:    tmux new -s exp 'python3 runpod_gpu_validation.py --model llama'"
echo "  6. Switch model: kill vLLM, './start_vllm_mistral.sh', re-run with --model mistral"
echo ""
echo "Estimated: ~150 samples x 4 ratios x 2 models = 1,200 trials"
echo "Runtime: ~6-8 hours total (~$9-11 at A100 spot)"
echo "================================================================"
