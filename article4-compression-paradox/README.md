# Article 4: The Compression Paradox

GPU energy proxy validation experiment for "The Compression Paradox: Provider-Dependent Energy Effects of Prompt Compression in LLM Inference."

## Experiment Summary

Direct GPU power measurement validates the paper's energy proxy formula using NVIDIA NVML at 50 Hz sampling with trapezoidal integration.

**Hardware**: NVIDIA L40S (48GB)
**Model**: TinyLlama-1.1B-Chat-v1.0
**Trials**: 48 (12 prompts x 4 compression ratios)

### Key Results

| Metric | Value |
|--------|-------|
| Pearson r | 0.96 (p < 1e-25) |
| Spearman rho | 0.84 (p < 1e-13) |
| Calibrated MAPE | 30% |
| Energy saving at 70% compression | 20% |

The proxy formula achieves excellent rank-order fidelity (r=0.96) but requires hardware-specific calibration of epsilon (bare-metal inference requires ~42x the API-calibrated constant).

## Files

- `data/gpu_energy_validation.csv` — Raw trial data (48 rows)
- `data/gpu_energy_validation.json` — Same data in JSON format
- `data/validation_summary.json` — Summary statistics
- `src/direct_gpu_experiment.py` — Experiment runner (HuggingFace Transformers + NVML)
- `src/setup_runpod_a100.sh` — RunPod setup script

## Reproduction

```bash
# On a GPU machine with NVIDIA drivers:
pip install torch transformers pynvml scipy
python src/direct_gpu_experiment.py --pilot 10   # Quick test
python src/direct_gpu_experiment.py               # Full run (48 trials)
```
