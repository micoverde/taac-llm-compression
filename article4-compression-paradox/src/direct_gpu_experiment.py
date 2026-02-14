#!/usr/bin/env python3
"""
Article 4: Direct GPU Energy Proxy Validation
==============================================
Measures real GPU energy (NVML) during LLM inference at different
prompt compression ratios, then compares against the paper's proxy formula.

Uses HuggingFace transformers directly (no vLLM needed).
Designed for RunPod L40S with limited budget — runs fast.

Usage:
  python3 direct_gpu_experiment.py              # Full run (~100 trials)
  python3 direct_gpu_experiment.py --pilot 10   # Quick pilot (10 trials)
"""

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
import threading
from dataclasses import dataclass, asdict
from pathlib import Path

import pynvml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# NVML Power Monitor — 50 Hz sampling, trapezoidal integration
# ---------------------------------------------------------------------------
@dataclass
class EnergyMeasurement:
    duration_s: float
    energy_joules: float
    avg_power_w: float
    peak_power_w: float
    min_power_w: float
    samples: int
    idle_power_w: float

class NVMLPowerMonitor:
    """Measures real GPU energy via NVML power sampling."""

    def __init__(self, gpu_index: int = 0, sample_hz: float = 50.0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.sample_interval = 1.0 / sample_hz
        self._power_samples: list[tuple[float, float]] = []
        self._running = False
        self._thread = None
        self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
        if isinstance(self.gpu_name, bytes):
            self.gpu_name = self.gpu_name.decode()

    def _sample_loop(self):
        while self._running:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self._power_samples.append((time.monotonic(), power_mw / 1000.0))
            except pynvml.NVMLError:
                pass
            time.sleep(self.sample_interval)

    def measure_idle(self, duration: float = 5.0) -> float:
        """Measure idle power for 5 seconds."""
        samples = []
        end = time.monotonic() + duration
        while time.monotonic() < end:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                samples.append(power_mw / 1000.0)
            except pynvml.NVMLError:
                pass
            time.sleep(self.sample_interval)
        return statistics.mean(samples) if samples else 0.0

    def start(self):
        self._power_samples = []
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> EnergyMeasurement:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        if len(self._power_samples) < 2:
            return EnergyMeasurement(0, 0, 0, 0, 0, 0, 0)

        # Trapezoidal integration
        energy_j = 0.0
        powers = [s[1] for s in self._power_samples]
        for i in range(1, len(self._power_samples)):
            dt = self._power_samples[i][0] - self._power_samples[i - 1][0]
            avg_p = (self._power_samples[i][1] + self._power_samples[i - 1][1]) / 2
            energy_j += avg_p * dt

        duration = self._power_samples[-1][0] - self._power_samples[0][0]
        idle = self.measure_idle(3.0)

        return EnergyMeasurement(
            duration_s=duration,
            energy_joules=energy_j,
            avg_power_w=statistics.mean(powers),
            peak_power_w=max(powers),
            min_power_w=min(powers),
            samples=len(self._power_samples),
            idle_power_w=idle,
        )


# ---------------------------------------------------------------------------
# Prompt compression (simple truncation-based, no LLMLingua dependency)
# ---------------------------------------------------------------------------
def compress_prompt(text: str, ratio: float) -> str:
    """Compress prompt by keeping `ratio` fraction of words."""
    if ratio >= 1.0:
        return text
    words = text.split()
    keep = max(1, int(len(words) * ratio))
    # Keep first and last portions (preserving instruction + context)
    head = keep // 2
    tail = keep - head
    if tail > 0:
        return " ".join(words[:head] + words[-tail:])
    return " ".join(words[:keep])


# ---------------------------------------------------------------------------
# Benchmark prompts
# ---------------------------------------------------------------------------
PROMPTS = [
    # HumanEval-style coding
    "Write a Python function called has_close_elements that takes a list of floating point numbers and a threshold value, and returns True if any two numbers in the list are closer to each other than the given threshold. Include proper edge case handling for empty lists and single-element lists. The function should have O(n log n) time complexity.",

    "Implement a function called separate_paren_groups that takes a string of nested parentheses and separates it into individual balanced groups. Each group should be a complete set of matched parentheses. For example, '(()()) (())' should return ['(()())', '(())'].",

    "Write a Python function truncate_number that takes a positive floating point number and returns the decimal part of it. For instance, for the input 3.5, it should return 0.5. Handle edge cases like very large numbers and numbers very close to zero.",

    # GSM8K-style math
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",

    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take to make the robe? Show your work step by step.",

    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make? Explain your reasoning.",

    # MMLU-style knowledge
    "Explain the key differences between Type I and Type II errors in statistical hypothesis testing. Provide examples of each in the context of medical diagnostic testing, and discuss why controlling for both types of errors simultaneously is challenging.",

    "Describe the process of oxidative phosphorylation in cellular respiration. Include the roles of the electron transport chain, ATP synthase, and the chemiosmotic gradient. What is the theoretical maximum ATP yield from one glucose molecule?",

    "In classical mechanics, derive the expression for the period of a simple pendulum for small oscillations. Explain what assumptions are made and how the period changes if the pendulum is taken to the Moon where gravitational acceleration is approximately 1.6 m/s^2.",

    # Long-context prompts (will show compression effect more clearly)
    "You are an expert software architect reviewing a system design. The system is a real-time collaborative document editor similar to Google Docs. It needs to support the following requirements: 1) Multiple users editing simultaneously with sub-second latency for character-level changes. 2) Offline editing capability with automatic conflict resolution when reconnecting. 3) Version history with the ability to restore any previous state. 4) Rich text formatting including tables, images, and embedded content. 5) Real-time cursor position sharing and user presence indicators. 6) Support for documents up to 10MB in size with up to 100 concurrent editors. Please provide a detailed technical architecture including: data structures for representing document state, conflict resolution algorithm choice and justification, network protocol design, storage architecture, and scalability considerations. Discuss trade-offs between CRDTs and Operational Transform approaches.",

    "Analyze the following economic scenario and provide policy recommendations: A developing country with a population of 50 million is experiencing 12% annual inflation, 15% unemployment, a current account deficit of 8% of GDP, and a fiscal deficit of 6% of GDP. The country's currency has depreciated 30% against the US dollar in the past year. The central bank's foreign reserves cover only 2 months of imports. The country has significant natural resource exports (40% of GDP) but is heavily dependent on imported manufactured goods and food. Discuss the potential policy interventions, their trade-offs, sequencing considerations, and lessons from similar situations in other countries. Consider both orthodox and heterodox approaches.",

    "You are a senior data scientist tasked with building a recommendation system for a streaming music platform with 100 million users and 80 million tracks. The system must handle both explicit feedback (likes, saves, skips) and implicit feedback (play counts, listening duration, playlist additions). Design the complete ML pipeline including: feature engineering from user behavior and audio features, model architecture choices (collaborative filtering, content-based, hybrid approaches, and deep learning methods), training infrastructure considerations, A/B testing methodology for recommendation quality, cold start handling for new users and new tracks, and real-time serving architecture. Discuss how to balance exploration vs exploitation and how to handle popularity bias.",
]

COMPRESSION_RATIOS = [1.0, 0.7, 0.5, 0.3]


# ---------------------------------------------------------------------------
# Proxy formula from paper: E = epsilon * (T_in + omega * T_out) * sqrt(N) * PUE
# ---------------------------------------------------------------------------
def proxy_energy(input_tokens: int, output_tokens: int, model_params_b: float,
                 epsilon: float = 0.0012, omega: float = 8.5, pue: float = 1.2) -> float:
    """Compute proxy energy estimate in joules."""
    import math
    return epsilon * (input_tokens + omega * output_tokens) * math.sqrt(model_params_b) * pue


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(n_trials: int = 100, output_dir: str = "/workspace/results"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Article 4: Direct GPU Energy Proxy Validation")
    print("=" * 70)

    # Load model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_params_b = 1.1  # billion parameters
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Init power monitor
    monitor = NVMLPowerMonitor()
    print(f"GPU: {monitor.gpu_name}")

    # Measure idle power
    print("Measuring idle power (5s)...")
    idle_power = monitor.measure_idle(5.0)
    print(f"Idle power: {idle_power:.1f}W")

    # Build trial matrix
    trials = []
    for prompt_idx, prompt in enumerate(PROMPTS):
        for ratio in COMPRESSION_RATIOS:
            trials.append((prompt_idx, prompt, ratio))

    # Limit to n_trials
    random.seed(42)
    random.shuffle(trials)
    trials = trials[:n_trials]

    print(f"\nRunning {len(trials)} trials...")
    print(f"Prompts: {len(PROMPTS)}, Ratios: {COMPRESSION_RATIOS}")
    print("-" * 70)

    results = []
    for i, (prompt_idx, prompt, ratio) in enumerate(trials):
        compressed = compress_prompt(prompt, ratio)

        # Use TinyLlama chat template for proper generation
        messages = [{"role": "user", "content": compressed}]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(chat_text, return_tensors="pt").to(model.device)
        input_tokens = input_ids.shape[1]

        # Warm up GPU (first trial only)
        if i == 0:
            print("Warming up GPU...")
            with torch.no_grad():
                _ = model.generate(input_ids, max_new_tokens=10)
            time.sleep(2)

        # Measure energy during generation
        monitor.start()
        t0 = time.monotonic()

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                min_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        t1 = time.monotonic()
        measurement = monitor.stop()

        output_tokens = outputs.shape[1] - input_ids.shape[1]
        wall_time = t1 - t0

        # Compute proxy estimate
        proxy_e = proxy_energy(input_tokens, output_tokens, model_params_b)

        # Net energy (subtract idle)
        net_energy = measurement.energy_joules - (idle_power * measurement.duration_s)
        net_energy = max(0.001, net_energy)  # floor at 1mJ

        row = {
            "trial": i + 1,
            "prompt_idx": prompt_idx,
            "compression_ratio": ratio,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "wall_time_s": round(wall_time, 4),
            "duration_s": round(measurement.duration_s, 4),
            "energy_joules": round(measurement.energy_joules, 4),
            "net_energy_joules": round(net_energy, 4),
            "avg_power_w": round(measurement.avg_power_w, 2),
            "peak_power_w": round(measurement.peak_power_w, 2),
            "idle_power_w": round(measurement.idle_power_w, 2),
            "power_samples": measurement.samples,
            "proxy_energy_joules": round(proxy_e, 4),
            "proxy_error_pct": round((proxy_e - net_energy) / net_energy * 100, 2) if net_energy > 0 else 0,
            "gpu_name": monitor.gpu_name,
            "model_name": model_name,
            "model_params_b": model_params_b,
        }
        results.append(row)

        # Progress
        status = "OK"
        if output_tokens < 5:
            status = "SHORT"
        print(f"  [{i+1:3d}/{len(trials)}] ratio={ratio:.1f} in={input_tokens:4d} out={output_tokens:4d} "
              f"E={net_energy:.2f}J proxy={proxy_e:.2f}J err={row['proxy_error_pct']:+.0f}% "
              f"P={measurement.avg_power_w:.0f}W {status}")

        # Small delay between trials for cleaner power measurement
        time.sleep(0.5)

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    csv_path = os.path.join(output_dir, "gpu_energy_validation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {csv_path}")

    json_path = os.path.join(output_dir, "gpu_energy_validation.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {json_path}")

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for ratio in COMPRESSION_RATIOS:
        subset = [r for r in results if r["compression_ratio"] == ratio]
        if not subset:
            continue
        energies = [r["net_energy_joules"] for r in subset]
        proxy_errors = [abs(r["proxy_error_pct"]) for r in subset]
        in_toks = [r["input_tokens"] for r in subset]
        out_toks = [r["output_tokens"] for r in subset]

        print(f"\n  Ratio {ratio:.1f} ({len(subset)} trials):")
        print(f"    Input tokens:  {statistics.mean(in_toks):.0f} avg")
        print(f"    Output tokens: {statistics.mean(out_toks):.0f} avg")
        print(f"    Energy:        {statistics.mean(energies):.3f}J avg (±{statistics.stdev(energies) if len(energies)>1 else 0:.3f})")
        print(f"    Proxy MAPE:    {statistics.mean(proxy_errors):.1f}%")

    # Pearson correlation
    try:
        from scipy import stats
        actual = [r["net_energy_joules"] for r in results]
        predicted = [r["proxy_energy_joules"] for r in results]
        r_val, p_val = stats.pearsonr(actual, predicted)
        mape = statistics.mean([abs(r["proxy_error_pct"]) for r in results])

        print(f"\n  Overall Proxy Validation:")
        print(f"    Pearson r:  {r_val:.4f} (p={p_val:.2e})")
        print(f"    MAPE:       {mape:.1f}%")
        print(f"    N trials:   {len(results)}")

        summary = {
            "gpu": monitor.gpu_name,
            "model": model_name,
            "model_params_b": model_params_b,
            "idle_power_w": idle_power,
            "n_trials": len(results),
            "pearson_r": round(r_val, 4),
            "pearson_p": float(f"{p_val:.2e}"),
            "mape_pct": round(mape, 1),
            "compression_ratios": COMPRESSION_RATIOS,
        }
        summary_path = os.path.join(output_dir, "validation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved: {summary_path}")
    except ImportError:
        print("  (scipy not available — skipping correlation)")

    print("\n" + "=" * 70)
    print("DONE — Remember to download results and terminate the pod!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=0, help="Quick pilot with N trials")
    args = parser.parse_args()

    n = args.pilot if args.pilot > 0 else len(PROMPTS) * len(COMPRESSION_RATIOS)
    run_experiment(n_trials=n)
