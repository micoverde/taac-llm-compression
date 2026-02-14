#!/usr/bin/env python3
"""Workload Generator for LLM Inference Energy Measurement.

This module implements an async workload generator for measuring LLM inference
energy consumption in datacenter simulation experiments. It supports:
- N concurrent clients (1, 4, 8, 16)
- HumanEval benchmark prompts
- Compression ratios: 1.0, 0.7, 0.5, 0.3
- Integration with vLLM OpenAI-compatible endpoint
- PowerMonitor correlation for energy measurement

Design Principles:
1. Async-first with httpx for high concurrency
2. Rate limiting to prevent server overload
3. Comprehensive result collection for energy analysis
4. Time-synchronized event markers for PowerMonitor correlation

Authors: Dr. Rodriguez, Green AI Research Team
Date: January 22, 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default vLLM endpoint (OpenAI-compatible)
DEFAULT_VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
DEFAULT_MODEL = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# Concurrency configurations for experiments
CONCURRENCY_LEVELS = [1, 4, 8, 16]

# Compression ratios to test
COMPRESSION_RATIOS = [1.0, 0.7, 0.5, 0.3]

# Rate limiting defaults
DEFAULT_RATE_LIMIT_RPS = 10.0  # Requests per second
DEFAULT_BURST_SIZE = 5  # Max burst before throttling

# Timeouts
REQUEST_TIMEOUT_SECONDS = 120.0
CONNECT_TIMEOUT_SECONDS = 10.0


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class WorkloadPhase(str, Enum):
    """Phase of workload execution for PowerMonitor correlation."""
    IDLE = "idle"
    WARMUP = "warmup"
    ACTIVE = "active"
    COOLDOWN = "cooldown"


@dataclass
class RequestRecord:
    """Record of a single inference request with all metrics for energy analysis.

    This schema captures everything needed for energy consumption analysis:
    - Token counts for energy proxy formula
    - Precise timing for PowerMonitor correlation
    - Client/batch context for utilization analysis
    """

    # Request identification
    request_id: str
    client_id: int
    batch_id: str

    # Experimental conditions
    compression_ratio: float
    concurrency_level: int

    # Benchmark info
    benchmark: str = "humaneval"
    sample_id: str = ""
    prompt_hash: str = ""

    # Token counts (critical for energy proxy)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    # Original vs compressed tokens
    original_tokens: int = 0
    compressed_tokens: int = 0
    actual_compression_ratio: float = 1.0

    # Timing (nanosecond precision for PowerMonitor correlation)
    start_time_ns: int = 0
    end_time_ns: int = 0
    queue_time_ns: int = 0      # Time waiting in client queue
    ttft_ns: int = 0            # Time to first token

    # Derived timing (milliseconds for human readability)
    latency_ms: float = 0.0
    inference_time_ms: float = 0.0
    compression_time_ms: float = 0.0

    # Throughput metrics
    tokens_per_second: float = 0.0
    input_tokens_per_second: float = 0.0
    output_tokens_per_second: float = 0.0

    # Response metadata
    model: str = ""
    finish_reason: str = ""
    http_status: int = 0

    # Error handling
    error: Optional[str] = None
    error_type: Optional[str] = None
    retries: int = 0

    # PowerMonitor correlation markers
    power_marker_start: str = ""
    power_marker_end: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RequestRecord":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class WorkloadBatch:
    """A batch of requests with aggregate metrics."""

    batch_id: str
    concurrency_level: int
    compression_ratio: float

    # Timing
    start_time_ns: int = 0
    end_time_ns: int = 0
    duration_ms: float = 0.0

    # Aggregate metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Token totals
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # Throughput
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0

    # Latency distribution
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0

    # Records
    records: list[RequestRecord] = field(default_factory=list)

    # PowerMonitor correlation
    power_marker: str = ""


@dataclass
class WorkloadConfig:
    """Configuration for a workload generation run."""

    # Endpoint configuration
    endpoint: str = DEFAULT_VLLM_ENDPOINT
    model: str = DEFAULT_MODEL
    api_key: str = ""

    # Experimental design
    concurrency_levels: list[int] = field(default_factory=lambda: CONCURRENCY_LEVELS.copy())
    compression_ratios: list[float] = field(default_factory=lambda: COMPRESSION_RATIOS.copy())
    samples_per_condition: int = 50
    replicates: int = 3

    # Rate limiting
    rate_limit_rps: float = DEFAULT_RATE_LIMIT_RPS
    burst_size: int = DEFAULT_BURST_SIZE

    # Request parameters
    max_tokens: int = 1024
    temperature: float = 0.0  # Deterministic for reproducibility

    # Warmup and cooldown
    warmup_requests: int = 5
    cooldown_seconds: float = 5.0

    # Output
    output_dir: Path = Path("./results")

    def total_trials(self) -> int:
        """Calculate total number of trials."""
        return (
            len(self.concurrency_levels) *
            len(self.compression_ratios) *
            self.samples_per_condition *
            self.replicates
        )


# =============================================================================
# HUMANEVAL BENCHMARK DATA
# =============================================================================

# HumanEval prompts for code generation benchmarking
# Full dataset: 164 problems, here we include representative samples
HUMANEVAL_PROMPTS = [
    {
        "id": "HumanEval/0",
        "prompt": '''from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
        "canonical_solution": '''    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
''',
        "entry_point": "has_close_elements",
    },
    {
        "id": "HumanEval/1",
        "prompt": '''from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses.
    Your goal is to separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other.
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
''',
        "canonical_solution": '''    result = []
    current_string = []
    current_depth = 0
    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)
            if current_depth == 0:
                result.append(''.join(current_string))
                current_string = []
    return result
''',
        "entry_point": "separate_paren_groups",
    },
    {
        "id": "HumanEval/2",
        "prompt": '''def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).
    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
''',
        "canonical_solution": '''    return number % 1.0
''',
        "entry_point": "truncate_number",
    },
    {
        "id": "HumanEval/3",
        "prompt": '''from typing import List

def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account falls below zero, and
    at that point function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
''',
        "canonical_solution": '''    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False
''',
        "entry_point": "below_zero",
    },
    {
        "id": "HumanEval/4",
        "prompt": '''from typing import List

def mean_absolute_deviation(numbers: List[float]) -> float:
    """ For a given list of input numbers, calculate Mean Absolute Deviation
    around the mean of this dataset.
    Mean Absolute Deviation is the average absolute difference between each
    element and a centerpoint (mean in this case):
    MAD = average | x - x_mean |
    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])
    1.0
    """
''',
        "canonical_solution": '''    mean = sum(numbers) / len(numbers)
    return sum(abs(x - mean) for x in numbers) / len(numbers)
''',
        "entry_point": "mean_absolute_deviation",
    },
    {
        "id": "HumanEval/5",
        "prompt": '''from typing import List

def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
''',
        "canonical_solution": '''    if not numbers:
        return []
    result = []
    for n in numbers[:-1]:
        result.append(n)
        result.append(delimeter)
    result.append(numbers[-1])
    return result
''',
        "entry_point": "intersperse",
    },
    # Add more prompts as needed for full 164-problem benchmark
    # These can be loaded from the HumanEval dataset file
]


def load_humaneval_prompts(data_path: Optional[Path] = None) -> list[dict]:
    """Load HumanEval prompts from file or use built-in samples.

    Args:
        data_path: Optional path to humaneval.jsonl file

    Returns:
        List of prompt dictionaries with id, prompt, canonical_solution, entry_point
    """
    if data_path and data_path.exists():
        prompts = []
        with open(data_path) as f:
            for line in f:
                prompts.append(json.loads(line))
        return prompts
    return HUMANEVAL_PROMPTS


# =============================================================================
# PROMPT COMPRESSION
# =============================================================================

def compress_prompt(prompt: str, target_ratio: float) -> tuple[str, int, float]:
    """Compress a prompt to approximately the target ratio.

    This implements a simplified compression strategy for the workload generator.
    In production, this would integrate with LLMLingua-2 or similar.

    Args:
        prompt: Original prompt text
        target_ratio: Target compression ratio (0.0-1.0, where 1.0 = no compression)

    Returns:
        Tuple of (compressed_prompt, compression_time_ms, actual_ratio)
    """
    if target_ratio >= 1.0:
        return prompt, 0, 1.0

    start_time = time.perf_counter_ns()

    # Estimate tokens (roughly 4 chars per token)
    original_tokens = len(prompt) // 4
    target_tokens = int(original_tokens * target_ratio)

    # Simple token-preserving compression
    # Priority: code structure > docstrings > examples > whitespace
    lines = prompt.split('\n')

    if target_ratio >= 0.7:
        # Light compression: remove excessive whitespace and redundant examples
        compressed_lines = []
        example_count = 0
        for line in lines:
            stripped = line.strip()
            # Keep at most 2 examples
            if stripped.startswith('>>>'):
                example_count += 1
                if example_count <= 2:
                    compressed_lines.append(line)
            else:
                # Remove excessive blank lines
                if stripped or (compressed_lines and compressed_lines[-1].strip()):
                    compressed_lines.append(line)
        compressed = '\n'.join(compressed_lines)

    elif target_ratio >= 0.5:
        # Moderate compression: keep signature and first example only
        compressed_lines = []
        in_docstring = False
        example_seen = False
        for line in lines:
            stripped = line.strip()
            if '"""' in line or "'''" in line:
                in_docstring = not in_docstring
                compressed_lines.append(line)
            elif stripped.startswith('>>>'):
                if not example_seen:
                    compressed_lines.append(line)
                    example_seen = True
            elif stripped.startswith('def ') or stripped.startswith('class '):
                compressed_lines.append(line)
            elif not in_docstring and stripped:
                compressed_lines.append(line)
        compressed = '\n'.join(compressed_lines)

    else:
        # Aggressive compression: signature and brief description only
        compressed_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def ') or stripped.startswith('class '):
                compressed_lines.append(line)
            elif stripped.startswith('"""') or stripped.startswith("'''"):
                # Include first line of docstring only
                compressed_lines.append(line)
                break
        compressed = '\n'.join(compressed_lines)

    compression_time_ns = time.perf_counter_ns() - start_time
    compression_time_ms = compression_time_ns // 1_000_000

    actual_tokens = len(compressed) // 4
    actual_ratio = actual_tokens / max(original_tokens, 1)

    return compressed, compression_time_ms, actual_ratio


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a simple heuristic of ~4 characters per token, which is
    reasonably accurate for code and English text with typical tokenizers.

    For production, use tiktoken or the model's actual tokenizer.
    """
    return max(1, len(text) // 4)


# =============================================================================
# RATE LIMITER
# =============================================================================

class TokenBucketRateLimiter:
    """Token bucket rate limiter for controlling request rate.

    This prevents overwhelming the vLLM server while allowing bursts
    to maximize GPU utilization during concurrent request testing.
    """

    def __init__(self, rate: float, burst: int):
        """Initialize rate limiter.

        Args:
            rate: Sustained rate in requests per second
            burst: Maximum burst size (bucket capacity)
        """
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire (default 1)

        Returns:
            Wait time in seconds (0 if no wait was needed)
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # Refill bucket
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)

            wait_time = 0.0
            if self.tokens < tokens:
                # Calculate required wait time
                deficit = tokens - self.tokens
                wait_time = deficit / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= tokens

            return wait_time

    def available(self) -> float:
        """Return current available tokens."""
        now = time.monotonic()
        elapsed = now - self.last_update
        return min(self.burst, self.tokens + elapsed * self.rate)


# =============================================================================
# ASYNC HTTP CLIENT
# =============================================================================

class VLLMClient:
    """Async HTTP client for vLLM OpenAI-compatible endpoint.

    Handles connection pooling, retries, and comprehensive metrics collection.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_VLLM_ENDPOINT,
        model: str = DEFAULT_MODEL,
        api_key: str = "",
        rate_limiter: Optional[TokenBucketRateLimiter] = None,
        max_retries: int = 3,
    ):
        """Initialize vLLM client.

        Args:
            endpoint: vLLM OpenAI-compatible API endpoint
            model: Model name to use for requests
            api_key: Optional API key
            rate_limiter: Optional rate limiter instance
            max_retries: Maximum retry attempts for failed requests
        """
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries

        # HTTP client with connection pooling
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "VLLMClient":
        """Async context manager entry."""
        timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT_SECONDS,
            read=REQUEST_TIMEOUT_SECONDS,
            write=REQUEST_TIMEOUT_SECONDS,
            pool=REQUEST_TIMEOUT_SECONDS,
        )

        # Connection limits for high concurrency
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            headers=headers,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        client_id: int = 0,
        request_id: Optional[str] = None,
    ) -> RequestRecord:
        """Send completion request and return detailed record.

        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            client_id: Client identifier for tracking
            request_id: Optional request ID (generated if not provided)

        Returns:
            RequestRecord with all timing and token metrics
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Use async context manager.")

        request_id = request_id or str(uuid.uuid4())

        # Rate limiting
        queue_time_ns = 0
        if self.rate_limiter:
            queue_start = time.perf_counter_ns()
            await self.rate_limiter.acquire()
            queue_time_ns = time.perf_counter_ns() - queue_start

        # Prepare request
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,  # Non-streaming for precise timing
        }

        # Generate power marker for correlation
        power_marker_start = f"REQ_START_{request_id}_{time.time_ns()}"

        # Execute request with retries
        retries = 0
        error = None
        response = None

        start_time_ns = time.perf_counter_ns()

        while retries <= self.max_retries:
            try:
                response = await self._client.post(self.endpoint, json=payload)
                if response.status_code == 200:
                    break
                elif response.status_code >= 500:
                    # Server error, retry
                    retries += 1
                    if retries <= self.max_retries:
                        await asyncio.sleep(2 ** retries)  # Exponential backoff
                else:
                    # Client error, don't retry
                    error = f"HTTP {response.status_code}: {response.text[:200]}"
                    break
            except httpx.TimeoutException as e:
                retries += 1
                error = f"Timeout: {str(e)}"
                if retries <= self.max_retries:
                    await asyncio.sleep(2 ** retries)
            except httpx.RequestError as e:
                retries += 1
                error = f"RequestError: {str(e)}"
                if retries <= self.max_retries:
                    await asyncio.sleep(2 ** retries)

        end_time_ns = time.perf_counter_ns()
        power_marker_end = f"REQ_END_{request_id}_{time.time_ns()}"

        # Calculate metrics
        latency_ns = end_time_ns - start_time_ns
        latency_ms = latency_ns / 1_000_000

        # Parse response
        input_tokens = estimate_tokens(prompt)
        output_tokens = 0
        finish_reason = ""
        response_text = ""

        if response and response.status_code == 200:
            try:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    response_text = choices[0].get("message", {}).get("content", "")
                    finish_reason = choices[0].get("finish_reason", "")

                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", input_tokens)
                output_tokens = usage.get("completion_tokens", estimate_tokens(response_text))

            except json.JSONDecodeError:
                error = "Invalid JSON response"

        # Calculate throughput
        inference_time_s = latency_ms / 1000
        tokens_per_second = (input_tokens + output_tokens) / max(inference_time_s, 0.001)
        output_tokens_per_second = output_tokens / max(inference_time_s, 0.001)

        return RequestRecord(
            request_id=request_id,
            client_id=client_id,
            batch_id="",  # Set by caller
            compression_ratio=1.0,  # Set by caller
            concurrency_level=1,  # Set by caller
            benchmark="humaneval",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            queue_time_ns=queue_time_ns,
            latency_ms=latency_ms,
            inference_time_ms=latency_ms,
            tokens_per_second=tokens_per_second,
            output_tokens_per_second=output_tokens_per_second,
            model=self.model,
            finish_reason=finish_reason,
            http_status=response.status_code if response else 0,
            error=error,
            error_type="RetryExhausted" if retries > self.max_retries else None,
            retries=retries,
            power_marker_start=power_marker_start,
            power_marker_end=power_marker_end,
        )


# =============================================================================
# WORKLOAD DISTRIBUTION STRATEGY
# =============================================================================

class WorkloadDistributor:
    """Distributes prompts across concurrent clients.

    Strategies:
    1. Round-robin: Each client gets prompts in rotation
    2. Random: Random assignment for realistic workload simulation
    3. Balanced: Ensures equal distribution of compression ratios per client
    """

    def __init__(
        self,
        prompts: list[dict],
        compression_ratios: list[float],
        strategy: str = "balanced",
    ):
        """Initialize workload distributor.

        Args:
            prompts: List of prompt dictionaries
            compression_ratios: List of compression ratios to test
            strategy: Distribution strategy ('round_robin', 'random', 'balanced')
        """
        self.prompts = prompts
        self.compression_ratios = compression_ratios
        self.strategy = strategy

        # Pre-generate workload items
        self._workload_items: list[tuple[dict, float]] = []
        self._generate_workload_items()

    def _generate_workload_items(self):
        """Generate all workload items (prompt, ratio pairs)."""
        for prompt in self.prompts:
            for ratio in self.compression_ratios:
                self._workload_items.append((prompt, ratio))

        if self.strategy == "random":
            random.shuffle(self._workload_items)

    def distribute(self, num_clients: int) -> list[list[tuple[dict, float]]]:
        """Distribute workload items across clients.

        Args:
            num_clients: Number of concurrent clients

        Returns:
            List of workload lists, one per client
        """
        client_workloads: list[list[tuple[dict, float]]] = [[] for _ in range(num_clients)]

        if self.strategy == "round_robin":
            for i, item in enumerate(self._workload_items):
                client_workloads[i % num_clients].append(item)

        elif self.strategy == "random":
            for item in self._workload_items:
                client_idx = random.randint(0, num_clients - 1)
                client_workloads[client_idx].append(item)

        elif self.strategy == "balanced":
            # Ensure each client gets roughly equal items per compression ratio
            ratio_items: dict[float, list[tuple[dict, float]]] = {
                r: [] for r in self.compression_ratios
            }
            for item in self._workload_items:
                ratio_items[item[1]].append(item)

            for ratio, items in ratio_items.items():
                for i, item in enumerate(items):
                    client_workloads[i % num_clients].append(item)

            # Shuffle each client's workload
            for workload in client_workloads:
                random.shuffle(workload)

        return client_workloads

    def total_items(self) -> int:
        """Return total number of workload items."""
        return len(self._workload_items)


# =============================================================================
# POWER MONITOR INTEGRATION
# =============================================================================

class PowerMonitorCorrelator:
    """Handles correlation between requests and power measurements.

    This class generates synchronized markers that can be correlated
    with external power monitoring tools (e.g., NVIDIA NVML, RAPL, PDU readings).
    """

    def __init__(self, output_dir: Path):
        """Initialize power monitor correlator.

        Args:
            output_dir: Directory for event log files
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.events: list[dict] = []
        self._session_id = str(uuid.uuid4())[:8]

    def mark_phase_start(self, phase: WorkloadPhase, metadata: Optional[dict] = None) -> str:
        """Mark the start of a workload phase.

        Args:
            phase: The workload phase starting
            metadata: Optional additional metadata

        Returns:
            Event marker string
        """
        marker = f"PHASE_START_{phase.value}_{self._session_id}_{time.time_ns()}"
        event = {
            "type": "phase_start",
            "phase": phase.value,
            "marker": marker,
            "timestamp_ns": time.time_ns(),
            "timestamp_iso": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
        }
        self.events.append(event)
        return marker

    def mark_phase_end(self, phase: WorkloadPhase, metadata: Optional[dict] = None) -> str:
        """Mark the end of a workload phase.

        Args:
            phase: The workload phase ending
            metadata: Optional additional metadata

        Returns:
            Event marker string
        """
        marker = f"PHASE_END_{phase.value}_{self._session_id}_{time.time_ns()}"
        event = {
            "type": "phase_end",
            "phase": phase.value,
            "marker": marker,
            "timestamp_ns": time.time_ns(),
            "timestamp_iso": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
        }
        self.events.append(event)
        return marker

    def mark_batch_start(self, batch_id: str, concurrency: int, ratio: float) -> str:
        """Mark the start of a request batch.

        Args:
            batch_id: Unique batch identifier
            concurrency: Number of concurrent clients
            ratio: Compression ratio for this batch

        Returns:
            Event marker string
        """
        marker = f"BATCH_START_{batch_id}_{time.time_ns()}"
        event = {
            "type": "batch_start",
            "batch_id": batch_id,
            "marker": marker,
            "timestamp_ns": time.time_ns(),
            "timestamp_iso": datetime.now(UTC).isoformat(),
            "concurrency": concurrency,
            "compression_ratio": ratio,
        }
        self.events.append(event)
        return marker

    def mark_batch_end(self, batch_id: str, stats: dict) -> str:
        """Mark the end of a request batch.

        Args:
            batch_id: Unique batch identifier
            stats: Batch statistics

        Returns:
            Event marker string
        """
        marker = f"BATCH_END_{batch_id}_{time.time_ns()}"
        event = {
            "type": "batch_end",
            "batch_id": batch_id,
            "marker": marker,
            "timestamp_ns": time.time_ns(),
            "timestamp_iso": datetime.now(UTC).isoformat(),
            "stats": stats,
        }
        self.events.append(event)
        return marker

    def save_event_log(self, filename: str = "power_events.jsonl"):
        """Save event log for PowerMonitor correlation.

        Args:
            filename: Output filename
        """
        path = self.output_dir / filename
        with open(path, "w") as f:
            for event in self.events:
                f.write(json.dumps(event) + "\n")

    def get_batch_timespan(self, batch_id: str) -> tuple[int, int]:
        """Get start/end timestamps for a batch.

        Args:
            batch_id: Batch identifier

        Returns:
            Tuple of (start_timestamp_ns, end_timestamp_ns)
        """
        start_ns = 0
        end_ns = 0
        for event in self.events:
            if event.get("batch_id") == batch_id:
                if event["type"] == "batch_start":
                    start_ns = event["timestamp_ns"]
                elif event["type"] == "batch_end":
                    end_ns = event["timestamp_ns"]
        return start_ns, end_ns


# =============================================================================
# WORKLOAD GENERATOR
# =============================================================================

class WorkloadGenerator:
    """Main workload generator for LLM inference energy measurement.

    Orchestrates concurrent client simulation, request timing, and
    result collection for energy analysis.
    """

    def __init__(self, config: WorkloadConfig):
        """Initialize workload generator.

        Args:
            config: Workload configuration
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.rate_limiter = TokenBucketRateLimiter(
            rate=config.rate_limit_rps,
            burst=config.burst_size,
        )
        self.power_correlator = PowerMonitorCorrelator(config.output_dir)

        # Load prompts
        self.prompts = load_humaneval_prompts()

        # Results storage
        self.all_records: list[RequestRecord] = []
        self.batches: list[WorkloadBatch] = []

    async def _run_client(
        self,
        client_id: int,
        workload: list[tuple[dict, float]],
        batch_id: str,
        concurrency_level: int,
        client: VLLMClient,
        results: list[RequestRecord],
    ):
        """Run a single client's workload.

        Args:
            client_id: Client identifier
            workload: List of (prompt_dict, compression_ratio) tuples
            batch_id: Batch identifier
            concurrency_level: Current concurrency level
            client: VLLMClient instance
            results: Shared results list
        """
        for prompt_dict, ratio in workload:
            prompt_text = prompt_dict["prompt"]
            sample_id = prompt_dict["id"]

            # Compress prompt
            compressed_prompt, compression_time_ms, actual_ratio = compress_prompt(
                prompt_text, ratio
            )

            # Generate request ID
            request_id = f"{batch_id}_{client_id}_{uuid.uuid4().hex[:8]}"

            # Send request
            record = await client.complete(
                prompt=compressed_prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                client_id=client_id,
                request_id=request_id,
            )

            # Add metadata
            record.batch_id = batch_id
            record.compression_ratio = ratio
            record.concurrency_level = concurrency_level
            record.sample_id = sample_id
            record.prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:12]
            record.original_tokens = estimate_tokens(prompt_text)
            record.compressed_tokens = estimate_tokens(compressed_prompt)
            record.actual_compression_ratio = actual_ratio
            record.compression_time_ms = compression_time_ms

            results.append(record)

    async def run_batch(
        self,
        concurrency_level: int,
        compression_ratio: float,
        num_samples: int,
    ) -> WorkloadBatch:
        """Run a batch of requests with specified concurrency.

        Args:
            concurrency_level: Number of concurrent clients
            compression_ratio: Compression ratio to use
            num_samples: Number of samples to process

        Returns:
            WorkloadBatch with results and metrics
        """
        batch_id = f"batch_{concurrency_level}c_{int(compression_ratio*100)}r_{uuid.uuid4().hex[:8]}"

        # Select prompts for this batch
        selected_prompts = self.prompts[:num_samples]

        # Create workload distributor
        distributor = WorkloadDistributor(
            prompts=selected_prompts,
            compression_ratios=[compression_ratio],
            strategy="balanced",
        )
        client_workloads = distributor.distribute(concurrency_level)

        # Mark batch start
        batch_marker = self.power_correlator.mark_batch_start(
            batch_id, concurrency_level, compression_ratio
        )

        start_time_ns = time.perf_counter_ns()

        # Run concurrent clients
        results: list[RequestRecord] = []

        async with VLLMClient(
            endpoint=self.config.endpoint,
            model=self.config.model,
            api_key=self.config.api_key,
            rate_limiter=self.rate_limiter,
        ) as client:
            tasks = []
            for client_id, workload in enumerate(client_workloads):
                if workload:  # Only create task if client has work
                    task = asyncio.create_task(
                        self._run_client(
                            client_id=client_id,
                            workload=workload,
                            batch_id=batch_id,
                            concurrency_level=concurrency_level,
                            client=client,
                            results=results,
                        )
                    )
                    tasks.append(task)

            await asyncio.gather(*tasks)

        end_time_ns = time.perf_counter_ns()

        # Calculate batch metrics
        duration_ms = (end_time_ns - start_time_ns) / 1_000_000
        successful = [r for r in results if r.error is None]

        # Latency percentiles
        latencies = sorted([r.latency_ms for r in successful]) if successful else [0]

        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]

        batch = WorkloadBatch(
            batch_id=batch_id,
            concurrency_level=concurrency_level,
            compression_ratio=compression_ratio,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            duration_ms=duration_ms,
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(results) - len(successful),
            total_input_tokens=sum(r.input_tokens for r in results),
            total_output_tokens=sum(r.output_tokens for r in results),
            requests_per_second=len(results) / max(duration_ms / 1000, 0.001),
            tokens_per_second=sum(r.total_tokens for r in results) / max(duration_ms / 1000, 0.001),
            latency_p50_ms=percentile(latencies, 50),
            latency_p95_ms=percentile(latencies, 95),
            latency_p99_ms=percentile(latencies, 99),
            latency_mean_ms=sum(latencies) / len(latencies) if latencies else 0,
            records=results,
            power_marker=batch_marker,
        )

        # Mark batch end
        self.power_correlator.mark_batch_end(batch_id, {
            "total_requests": batch.total_requests,
            "successful_requests": batch.successful_requests,
            "duration_ms": batch.duration_ms,
            "requests_per_second": batch.requests_per_second,
        })

        return batch

    async def run_experiment(self) -> list[WorkloadBatch]:
        """Run full experimental design across all conditions.

        Returns:
            List of WorkloadBatch results
        """
        print(f"Starting workload generation experiment")
        print(f"  Endpoint: {self.config.endpoint}")
        print(f"  Model: {self.config.model}")
        print(f"  Concurrency levels: {self.config.concurrency_levels}")
        print(f"  Compression ratios: {self.config.compression_ratios}")
        print(f"  Samples per condition: {self.config.samples_per_condition}")
        print(f"  Replicates: {self.config.replicates}")
        print(f"  Total trials: {self.config.total_trials()}")
        print()

        # Warmup phase
        self.power_correlator.mark_phase_start(WorkloadPhase.WARMUP)
        print("Running warmup...")
        warmup_batch = await self.run_batch(
            concurrency_level=1,
            compression_ratio=1.0,
            num_samples=self.config.warmup_requests,
        )
        self.power_correlator.mark_phase_end(WorkloadPhase.WARMUP)
        print(f"  Warmup complete: {warmup_batch.successful_requests} requests")
        print()

        # Active phase
        self.power_correlator.mark_phase_start(WorkloadPhase.ACTIVE)

        for replicate in range(1, self.config.replicates + 1):
            print(f"Replicate {replicate}/{self.config.replicates}")

            for concurrency in self.config.concurrency_levels:
                for ratio in self.config.compression_ratios:
                    print(f"  Running: concurrency={concurrency}, ratio={ratio:.1f}")

                    batch = await self.run_batch(
                        concurrency_level=concurrency,
                        compression_ratio=ratio,
                        num_samples=self.config.samples_per_condition,
                    )

                    self.batches.append(batch)
                    self.all_records.extend(batch.records)

                    print(f"    Completed: {batch.successful_requests}/{batch.total_requests} requests")
                    print(f"    Latency: p50={batch.latency_p50_ms:.1f}ms, p95={batch.latency_p95_ms:.1f}ms")
                    print(f"    Throughput: {batch.requests_per_second:.1f} req/s, {batch.tokens_per_second:.1f} tok/s")

                    # Cooldown between conditions
                    await asyncio.sleep(self.config.cooldown_seconds)

        self.power_correlator.mark_phase_end(WorkloadPhase.ACTIVE)

        # Save results
        self._save_results()

        print()
        print("Experiment complete!")
        print(f"  Total batches: {len(self.batches)}")
        print(f"  Total records: {len(self.all_records)}")
        print(f"  Results saved to: {self.config.output_dir}")

        return self.batches

    def _save_results(self):
        """Save all results to output directory."""
        # Save individual records
        records_path = self.config.output_dir / "request_records.jsonl"
        with open(records_path, "w") as f:
            for record in self.all_records:
                f.write(json.dumps(record.to_dict()) + "\n")

        # Save batch summaries
        batches_path = self.config.output_dir / "batch_summaries.jsonl"
        with open(batches_path, "w") as f:
            for batch in self.batches:
                summary = {
                    "batch_id": batch.batch_id,
                    "concurrency_level": batch.concurrency_level,
                    "compression_ratio": batch.compression_ratio,
                    "duration_ms": batch.duration_ms,
                    "total_requests": batch.total_requests,
                    "successful_requests": batch.successful_requests,
                    "failed_requests": batch.failed_requests,
                    "total_input_tokens": batch.total_input_tokens,
                    "total_output_tokens": batch.total_output_tokens,
                    "requests_per_second": batch.requests_per_second,
                    "tokens_per_second": batch.tokens_per_second,
                    "latency_p50_ms": batch.latency_p50_ms,
                    "latency_p95_ms": batch.latency_p95_ms,
                    "latency_p99_ms": batch.latency_p99_ms,
                    "latency_mean_ms": batch.latency_mean_ms,
                    "power_marker": batch.power_marker,
                }
                f.write(json.dumps(summary) + "\n")

        # Save power monitor event log
        self.power_correlator.save_event_log()

        # Save experiment configuration
        config_path = self.config.output_dir / "experiment_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "endpoint": self.config.endpoint,
                "model": self.config.model,
                "concurrency_levels": self.config.concurrency_levels,
                "compression_ratios": self.config.compression_ratios,
                "samples_per_condition": self.config.samples_per_condition,
                "replicates": self.config.replicates,
                "rate_limit_rps": self.config.rate_limit_rps,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "total_trials": self.config.total_trials(),
                "timestamp": datetime.now(UTC).isoformat(),
            }, f, indent=2)


# =============================================================================
# RESULT COLLECTION SCHEMA
# =============================================================================

RESULT_SCHEMA = """
Result Collection Schema for Energy Analysis
============================================

1. RequestRecord (per-request metrics):
   - request_id: Unique identifier
   - client_id: Client number (0 to N-1)
   - batch_id: Batch identifier
   - compression_ratio: Target compression ratio
   - concurrency_level: Number of concurrent clients
   - benchmark: Benchmark name (humaneval)
   - sample_id: Problem identifier

   Token Metrics (critical for energy proxy):
   - input_tokens: Actual input tokens from API
   - output_tokens: Actual output tokens from API
   - total_tokens: input + output
   - original_tokens: Tokens before compression
   - compressed_tokens: Tokens after compression
   - actual_compression_ratio: compressed/original

   Timing (nanosecond precision):
   - start_time_ns: Request start timestamp
   - end_time_ns: Request end timestamp
   - queue_time_ns: Time waiting for rate limiter
   - latency_ms: Total latency
   - inference_time_ms: Model inference time
   - compression_time_ms: Prompt compression time

   Throughput:
   - tokens_per_second: Total throughput
   - output_tokens_per_second: Generation speed

   PowerMonitor Correlation:
   - power_marker_start: Start marker for power logs
   - power_marker_end: End marker for power logs

2. WorkloadBatch (aggregate metrics):
   - batch_id: Unique identifier
   - concurrency_level: N clients
   - compression_ratio: Ratio tested
   - duration_ms: Total batch duration
   - total_requests: Number of requests
   - successful_requests: Completed without error
   - total_input_tokens: Sum of inputs
   - total_output_tokens: Sum of outputs
   - requests_per_second: Throughput
   - latency_p50/p95/p99_ms: Percentiles
   - power_marker: Batch marker for power logs

3. PowerMonitor Event Log:
   - Phase markers (warmup, active, cooldown)
   - Batch start/end with timestamps
   - Nanosecond precision for correlation
"""


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """Example usage of the workload generator."""

    # Create configuration
    config = WorkloadConfig(
        endpoint="http://localhost:8000/v1/chat/completions",
        model="meta-llama/Llama-3.1-8B-Instruct",
        concurrency_levels=[1, 4, 8, 16],
        compression_ratios=[1.0, 0.7, 0.5, 0.3],
        samples_per_condition=10,  # Use 50+ for real experiments
        replicates=1,  # Use 3 for real experiments
        rate_limit_rps=10.0,
        max_tokens=512,
        temperature=0.0,
        warmup_requests=3,
        cooldown_seconds=2.0,
        output_dir=Path("./workload_results"),
    )

    # Run experiment
    generator = WorkloadGenerator(config)
    batches = await generator.run_experiment()

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for batch in batches:
        print(f"\nBatch: concurrency={batch.concurrency_level}, ratio={batch.compression_ratio}")
        print(f"  Requests: {batch.successful_requests}/{batch.total_requests}")
        print(f"  Input tokens: {batch.total_input_tokens}")
        print(f"  Output tokens: {batch.total_output_tokens}")
        print(f"  Latency p50: {batch.latency_p50_ms:.1f}ms")
        print(f"  Throughput: {batch.requests_per_second:.1f} req/s")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Workload Generator for LLM Inference Energy Measurement"
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_VLLM_ENDPOINT,
        help="vLLM OpenAI-compatible endpoint URL",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name to use",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=CONCURRENCY_LEVELS,
        help="Concurrency levels to test (e.g., 1 4 8 16)",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=COMPRESSION_RATIOS,
        help="Compression ratios to test (e.g., 1.0 0.7 0.5 0.3)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Samples per condition",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="Number of replicates",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=DEFAULT_RATE_LIMIT_RPS,
        help="Rate limit in requests per second",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./workload_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration without running",
    )

    args = parser.parse_args()

    config = WorkloadConfig(
        endpoint=args.endpoint,
        model=args.model,
        api_key=args.api_key,
        concurrency_levels=args.concurrency,
        compression_ratios=args.ratios,
        samples_per_condition=args.samples,
        replicates=args.replicates,
        rate_limit_rps=args.rate_limit,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )

    if args.dry_run:
        print("Configuration:")
        print(f"  Endpoint: {config.endpoint}")
        print(f"  Model: {config.model}")
        print(f"  Concurrency levels: {config.concurrency_levels}")
        print(f"  Compression ratios: {config.compression_ratios}")
        print(f"  Samples per condition: {config.samples_per_condition}")
        print(f"  Replicates: {config.replicates}")
        print(f"  Rate limit: {config.rate_limit_rps} RPS")
        print(f"  Total trials: {config.total_trials()}")
        print(f"  Output directory: {config.output_dir}")
        return

    generator = WorkloadGenerator(config)
    asyncio.run(generator.run_experiment())


if __name__ == "__main__":
    main()
