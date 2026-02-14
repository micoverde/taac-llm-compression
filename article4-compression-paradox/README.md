# Article 4: The Compression Paradox

Complete experiment code, data, and analysis for "The Compression Paradox: Provider-Dependent Energy Effects of Prompt Compression in LLM Inference."

## Repository Structure

```
article4-compression-paradox/
├── data/
│   ├── runpod-validation/          # GPU energy proxy validation (48 trials)
│   │   ├── gpu_energy_validation.csv
│   │   ├── gpu_energy_validation.json
│   │   └── validation_summary.json
│   ├── api-experiments/            # Main API experiment data
│   │   ├── trial_matrix.jsonl      # 28,421 trial results (18MB)
│   │   ├── execution_manifest.json
│   │   └── phase2/                 # Phase 2 follow-up experiments
│   └── analysis/                   # Processed analysis outputs
│       ├── energy_cost_analysis_detailed.csv
│       ├── projections_1m_calls.csv
│       ├── benchmarks/
│       └── figures/
├── src/                            # Core implementation
│   ├── direct_gpu_experiment.py    # RunPod GPU validation script
│   ├── power_monitor.py           # NVML power monitoring (1078 lines)
│   ├── calibration_protocol.py    # Proxy calibration protocol
│   ├── edge_cases.py              # Edge case handling
│   ├── schemas.py                 # Data schemas
│   └── setup_runpod_a100.sh       # RunPod setup script
├── scripts/                        # Experiment runners
│   ├── article4_experiment_runner.py  # Main 3-provider experiment
│   ├── generate_trial_matrix.py       # Trial matrix generation
│   ├── co2_calculations.py            # Carbon emission calculations
│   ├── phase2_statistical_analysis.py # Phase 2 statistics
│   └── workload_generator.py          # Benchmark workload generator
└── analysis/                       # Proxy calibration analysis
    ├── sde8_proxy_calibration.py
    └── sde8_proxy_calibration_v2.py
```

## Experiment Overview

### API Experiments (Main Paper)
- **Providers**: OpenAI GPT-4o-mini, Anthropic Claude 3.5 Sonnet, DeepSeek-Chat
- **Benchmarks**: HumanEval (164), MBPP (500), GSM8K (200), MATH (100), MMLU-STEM (200)
- **Compression**: LLMLingua-2 at ratios 1.0, 0.7, 0.5, 0.3
- **Trials**: 28,421 completed (99.98% of planned 28,428)

### GPU Energy Validation
- **Hardware**: NVIDIA L40S (48GB) on RunPod
- **Model**: TinyLlama-1.1B via HuggingFace Transformers
- **Method**: NVML 50Hz power sampling with trapezoidal integration
- **Trials**: 48 randomized (12 prompts x 4 ratios)

| Metric | Value |
|--------|-------|
| Pearson r | 0.96 (p < 1e-25) |
| Spearman rho | 0.84 (p < 1e-13) |
| Calibrated MAPE | 30% |
| Energy saving at 70% compression | 20% |

## Reproduction

### GPU Validation
```bash
pip install torch transformers pynvml scipy
python src/direct_gpu_experiment.py --pilot 10   # Quick test
python src/direct_gpu_experiment.py               # Full run (48 trials)
```

### API Experiments
```bash
pip install openai anthropic datasets llmlingua
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export DEEPSEEK_API_KEY=...
python scripts/article4_experiment_runner.py
```

## License

Research code and data released for academic use.
