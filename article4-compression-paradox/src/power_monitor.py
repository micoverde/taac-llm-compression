"""
Power Monitoring System for Green AI Datacenter Simulation
===========================================================

Dr. Park - Systems Engineering Design
Article 4: Green AI Research

This module provides high-frequency GPU power monitoring using NVIDIA NVML,
with thread-safe access and accurate energy attribution for overlapping requests.

Requirements:
- NVIDIA GPU with NVML support
- pynvml library (pip install pynvml)
- Python 3.8+
"""

import csv
import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logger.warning("pynvml not available - using mock mode for testing")


class MonitorState(Enum):
    """Power monitor operational states."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class PowerSample:
    """Single power measurement sample."""
    timestamp: float  # Unix timestamp with microsecond precision
    power_watts: float  # Instantaneous power draw
    gpu_utilization: float  # GPU utilization percentage (0-100)
    memory_used_bytes: int  # GPU memory used in bytes
    memory_total_bytes: int  # Total GPU memory in bytes
    temperature_c: float  # GPU temperature in Celsius
    gpu_index: int = 0  # GPU device index

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "power_watts": round(self.power_watts, 2),
            "gpu_utilization": round(self.gpu_utilization, 1),
            "memory_used_mb": round(self.memory_used_bytes / (1024 * 1024), 2),
            "memory_total_mb": round(self.memory_total_bytes / (1024 * 1024), 2),
            "memory_utilization": round(100 * self.memory_used_bytes / max(self.memory_total_bytes, 1), 1),
            "temperature_c": round(self.temperature_c, 1),
            "gpu_index": self.gpu_index
        }


@dataclass
class RequestEnergyAttribution:
    """Energy attribution for a single inference request."""
    request_id: str
    start_time: float
    end_time: float
    duration_seconds: float

    # Energy metrics
    total_energy_joules: float  # Total energy consumed
    attributed_energy_joules: float  # Energy attributed to this request
    idle_baseline_watts: float  # Estimated idle power
    active_energy_joules: float  # Energy above idle baseline

    # Resource utilization
    avg_power_watts: float
    peak_power_watts: float
    avg_gpu_utilization: float
    avg_memory_utilization: float

    # Attribution metadata
    overlapping_requests: int  # Number of concurrent requests
    attribution_method: str  # Algorithm used for attribution
    confidence_score: float  # Confidence in attribution (0-1)

    sample_count: int = 0  # Number of power samples in window

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActiveRequest:
    """Tracks an active inference request for energy attribution."""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    metadata: dict = field(default_factory=dict)


class EnergyAttributionAlgorithm(Enum):
    """Available energy attribution algorithms."""
    EQUAL_SHARE = "equal_share"  # Divide energy equally among concurrent requests
    TIME_WEIGHTED = "time_weighted"  # Weight by request duration overlap
    UTILIZATION_WEIGHTED = "utilization_weighted"  # Weight by GPU utilization delta
    MARGINAL_COST = "marginal_cost"  # Attribute marginal energy above baseline


class PowerMonitor:
    """
    High-frequency GPU power monitor with thread-safe access
    and energy attribution for overlapping requests.

    Usage:
        monitor = PowerMonitor(sample_rate_hz=10)
        monitor.start()

        # Track a request
        request_id = monitor.begin_request(metadata={"model": "llama-7b"})
        # ... inference happens ...
        attribution = monitor.end_request(request_id)

        # Get cumulative energy
        total_energy = monitor.get_total_energy()

        monitor.stop()
        monitor.export_csv("power_data.csv")
    """

    # Class constants
    DEFAULT_SAMPLE_RATE_HZ = 10  # 100ms intervals
    MAX_BUFFER_SIZE = 36000  # 1 hour at 10Hz
    IDLE_CALIBRATION_SECONDS = 5  # Time to measure idle baseline
    MIN_IDLE_SAMPLES = 30  # Minimum samples for idle calibration

    def __init__(
        self,
        sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
        gpu_index: int = 0,
        buffer_size: int = MAX_BUFFER_SIZE,
        attribution_algorithm: EnergyAttributionAlgorithm = EnergyAttributionAlgorithm.UTILIZATION_WEIGHTED,
        auto_calibrate_idle: bool = True
    ):
        """
        Initialize the power monitor.

        Args:
            sample_rate_hz: Sampling frequency (default 10Hz = 100ms intervals)
            gpu_index: NVIDIA GPU device index to monitor
            buffer_size: Maximum samples to keep in memory
            attribution_algorithm: Algorithm for energy attribution
            auto_calibrate_idle: Whether to calibrate idle power on start
        """
        self.sample_rate_hz = sample_rate_hz
        self.sample_interval = 1.0 / sample_rate_hz
        self.gpu_index = gpu_index
        self.buffer_size = buffer_size
        self.attribution_algorithm = attribution_algorithm
        self.auto_calibrate_idle = auto_calibrate_idle

        # Thread synchronization
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._sample_thread: Optional[threading.Thread] = None

        # State
        self._state = MonitorState.STOPPED
        self._nvml_handle = None

        # Data storage (thread-safe via lock)
        self._samples: deque = deque(maxlen=buffer_size)
        self._active_requests: Dict[str, ActiveRequest] = {}
        self._completed_attributions: List[RequestEnergyAttribution] = []

        # Calibration data
        self._idle_power_watts: float = 0.0
        self._idle_calibrated: bool = False

        # Statistics
        self._total_samples: int = 0
        self._missed_samples: int = 0
        self._start_time: Optional[float] = None

    def start(self) -> bool:
        """
        Start the power monitoring thread.

        Returns:
            True if started successfully, False otherwise.
        """
        with self._lock:
            if self._state == MonitorState.RUNNING:
                logger.warning("PowerMonitor already running")
                return True

            try:
                # Initialize NVML
                if NVML_AVAILABLE:
                    pynvml.nvmlInit()
                    self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)

                    # Log GPU info
                    gpu_name = pynvml.nvmlDeviceGetName(self._nvml_handle)
                    logger.info(f"Initialized NVML for GPU {self.gpu_index}: {gpu_name}")
                else:
                    logger.warning("Running in mock mode - no real GPU measurements")

                # Start sampling thread
                self._stop_event.clear()
                self._start_time = time.time()
                self._sample_thread = threading.Thread(
                    target=self._sampling_loop,
                    name="PowerMonitor-Sampler",
                    daemon=True
                )
                self._sample_thread.start()
                self._state = MonitorState.RUNNING

                # Calibrate idle power if requested
                if self.auto_calibrate_idle and not self._idle_calibrated:
                    self._calibrate_idle_power()

                logger.info(f"PowerMonitor started at {self.sample_rate_hz}Hz")
                return True

            except Exception as e:
                logger.error(f"Failed to start PowerMonitor: {e}")
                self._state = MonitorState.ERROR
                return False

    def stop(self) -> None:
        """Stop the power monitoring thread."""
        with self._lock:
            if self._state != MonitorState.RUNNING:
                return

            self._stop_event.set()

        # Wait for thread outside lock to avoid deadlock
        if self._sample_thread and self._sample_thread.is_alive():
            self._sample_thread.join(timeout=2.0)

        with self._lock:
            # Shutdown NVML
            if NVML_AVAILABLE:
                try:
                    pynvml.nvmlShutdown()
                except Exception as e:
                    logger.warning(f"Error shutting down NVML: {e}")

            self._state = MonitorState.STOPPED
            self._nvml_handle = None

            # Log final statistics
            runtime = time.time() - self._start_time if self._start_time else 0
            logger.info(
                f"PowerMonitor stopped. Runtime: {runtime:.1f}s, "
                f"Samples: {self._total_samples}, Missed: {self._missed_samples}"
            )

    def _sampling_loop(self) -> None:
        """Main sampling loop running in dedicated thread."""
        next_sample_time = time.time()

        while not self._stop_event.is_set():
            current_time = time.time()

            # Check if we missed samples
            if current_time > next_sample_time + self.sample_interval:
                missed = int((current_time - next_sample_time) / self.sample_interval)
                with self._lock:
                    self._missed_samples += missed
                logger.warning(f"Missed {missed} samples due to timing drift")
                next_sample_time = current_time

            # Take sample
            try:
                sample = self._take_sample()
                if sample:
                    with self._lock:
                        self._samples.append(sample)
                        self._total_samples += 1
            except Exception as e:
                logger.error(f"Sampling error: {e}")

            # Schedule next sample
            next_sample_time += self.sample_interval
            sleep_time = next_sample_time - time.time()

            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

    def _take_sample(self) -> Optional[PowerSample]:
        """Take a single power measurement."""
        timestamp = time.time()

        if NVML_AVAILABLE and self._nvml_handle:
            try:
                # Get power
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)
                power_watts = power_mw / 1000.0

                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu_util = util.gpu

                # Get memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)

                # Get temperature
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle,
                    pynvml.NVML_TEMPERATURE_GPU
                )

                return PowerSample(
                    timestamp=timestamp,
                    power_watts=power_watts,
                    gpu_utilization=gpu_util,
                    memory_used_bytes=mem_info.used,
                    memory_total_bytes=mem_info.total,
                    temperature_c=temp,
                    gpu_index=self.gpu_index
                )
            except pynvml.NVMLError as e:
                logger.error(f"NVML error: {e}")
                return None
        else:
            # Mock mode for testing
            return self._generate_mock_sample(timestamp)

    def _generate_mock_sample(self, timestamp: float) -> PowerSample:
        """Generate mock sample for testing without GPU."""
        import random

        # Simulate realistic power patterns
        base_power = 50.0  # Idle power
        active_requests = len(self._active_requests)

        # Add power per active request
        power_per_request = 30.0 + random.uniform(-5, 5)
        power = base_power + (active_requests * power_per_request)
        power += random.uniform(-2, 2)  # Noise

        gpu_util = min(100, active_requests * 25 + random.uniform(-5, 5))

        return PowerSample(
            timestamp=timestamp,
            power_watts=power,
            gpu_utilization=max(0, gpu_util),
            memory_used_bytes=int(2e9 + active_requests * 1e9),
            memory_total_bytes=int(16e9),
            temperature_c=45 + active_requests * 5 + random.uniform(-2, 2),
            gpu_index=self.gpu_index
        )

    def _calibrate_idle_power(self) -> None:
        """Calibrate idle power baseline."""
        logger.info("Calibrating idle power baseline...")

        # Wait for calibration samples
        time.sleep(self.IDLE_CALIBRATION_SECONDS)

        with self._lock:
            if len(self._samples) < self.MIN_IDLE_SAMPLES:
                logger.warning("Insufficient samples for idle calibration")
                self._idle_power_watts = 50.0  # Default estimate
                return

            # Use lowest 10th percentile as idle baseline
            recent_samples = list(self._samples)[-self.MIN_IDLE_SAMPLES:]
            power_values = sorted([s.power_watts for s in recent_samples])
            percentile_10_idx = len(power_values) // 10
            self._idle_power_watts = power_values[percentile_10_idx]
            self._idle_calibrated = True

            logger.info(f"Idle power calibrated: {self._idle_power_watts:.1f}W")

    # =========================================================================
    # Request Tracking and Energy Attribution
    # =========================================================================

    def begin_request(self, request_id: Optional[str] = None, metadata: dict = None) -> str:
        """
        Begin tracking a new inference request.

        Args:
            request_id: Optional custom request ID (auto-generated if None)
            metadata: Optional metadata about the request (model, batch_size, etc.)

        Returns:
            The request ID for later reference.
        """
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]

        with self._lock:
            if request_id in self._active_requests:
                raise ValueError(f"Request {request_id} already active")

            self._active_requests[request_id] = ActiveRequest(
                request_id=request_id,
                start_time=time.time(),
                metadata=metadata or {}
            )

            logger.debug(f"Started tracking request {request_id}")
            return request_id

    def end_request(self, request_id: str) -> RequestEnergyAttribution:
        """
        End tracking a request and compute energy attribution.

        Args:
            request_id: The request ID returned by begin_request()

        Returns:
            Energy attribution for the request.
        """
        end_time = time.time()

        with self._lock:
            if request_id not in self._active_requests:
                raise ValueError(f"Request {request_id} not found")

            request = self._active_requests.pop(request_id)
            request.end_time = end_time

            # Compute attribution
            attribution = self._compute_attribution(request)
            self._completed_attributions.append(attribution)

            logger.debug(
                f"Request {request_id} completed: "
                f"{attribution.attributed_energy_joules:.2f}J over {attribution.duration_seconds:.3f}s"
            )

            return attribution

    def _compute_attribution(self, request: ActiveRequest) -> RequestEnergyAttribution:
        """
        Compute energy attribution for a completed request.

        This implements multiple attribution algorithms for handling
        overlapping concurrent requests.
        """
        start_time = request.start_time
        end_time = request.end_time
        duration = end_time - start_time

        # Get samples within request window
        samples = [
            s for s in self._samples
            if start_time <= s.timestamp <= end_time
        ]

        if not samples:
            # No samples in window - estimate based on duration
            logger.warning(f"No samples for request {request.request_id}")
            return RequestEnergyAttribution(
                request_id=request.request_id,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                total_energy_joules=0,
                attributed_energy_joules=0,
                idle_baseline_watts=self._idle_power_watts,
                active_energy_joules=0,
                avg_power_watts=0,
                peak_power_watts=0,
                avg_gpu_utilization=0,
                avg_memory_utilization=0,
                overlapping_requests=0,
                attribution_method=self.attribution_algorithm.value,
                confidence_score=0.0,
                sample_count=0
            )

        # Calculate total energy using trapezoidal integration
        total_energy = self._integrate_energy(samples)

        # Calculate statistics
        power_values = [s.power_watts for s in samples]
        gpu_utils = [s.gpu_utilization for s in samples]
        mem_utils = [100 * s.memory_used_bytes / s.memory_total_bytes for s in samples]

        avg_power = sum(power_values) / len(power_values)
        peak_power = max(power_values)
        avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
        avg_mem_util = sum(mem_utils) / len(mem_utils)

        # Energy above idle baseline
        active_energy = max(0, total_energy - (self._idle_power_watts * duration))

        # Count overlapping requests during this window
        overlapping = self._count_overlapping_requests(start_time, end_time, request.request_id)

        # Apply attribution algorithm
        attributed_energy, confidence = self._apply_attribution(
            request=request,
            samples=samples,
            total_energy=total_energy,
            active_energy=active_energy,
            overlapping_count=overlapping
        )

        return RequestEnergyAttribution(
            request_id=request.request_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_energy_joules=total_energy,
            attributed_energy_joules=attributed_energy,
            idle_baseline_watts=self._idle_power_watts,
            active_energy_joules=active_energy,
            avg_power_watts=avg_power,
            peak_power_watts=peak_power,
            avg_gpu_utilization=avg_gpu_util,
            avg_memory_utilization=avg_mem_util,
            overlapping_requests=overlapping,
            attribution_method=self.attribution_algorithm.value,
            confidence_score=confidence,
            sample_count=len(samples)
        )

    def _integrate_energy(self, samples: List[PowerSample]) -> float:
        """
        Integrate power over time using trapezoidal rule.

        Energy (Joules) = integral of Power (Watts) over Time (seconds)
        """
        if len(samples) < 2:
            if samples:
                # Single sample - estimate based on sample interval
                return samples[0].power_watts * self.sample_interval
            return 0.0

        energy = 0.0
        for i in range(1, len(samples)):
            dt = samples[i].timestamp - samples[i-1].timestamp
            avg_power = (samples[i].power_watts + samples[i-1].power_watts) / 2
            energy += avg_power * dt

        return energy

    def _count_overlapping_requests(
        self,
        start_time: float,
        end_time: float,
        exclude_id: str
    ) -> int:
        """Count requests that overlap with the given time window."""
        count = 0

        # Check active requests
        for req_id, req in self._active_requests.items():
            if req_id == exclude_id:
                continue
            # Request overlaps if it started before end and hasn't ended before start
            if req.start_time < end_time:
                count += 1

        # Check recently completed requests
        for attr in self._completed_attributions[-100:]:  # Check recent completions
            if attr.request_id == exclude_id:
                continue
            # Check for time overlap
            if attr.start_time < end_time and attr.end_time > start_time:
                count += 1

        return count

    def _apply_attribution(
        self,
        request: ActiveRequest,
        samples: List[PowerSample],
        total_energy: float,
        active_energy: float,
        overlapping_count: int
    ) -> Tuple[float, float]:
        """
        Apply the configured attribution algorithm.

        Returns:
            Tuple of (attributed_energy, confidence_score)
        """
        if overlapping_count == 0:
            # No overlap - full attribution with high confidence
            return total_energy, 1.0

        n_concurrent = overlapping_count + 1  # Include this request

        if self.attribution_algorithm == EnergyAttributionAlgorithm.EQUAL_SHARE:
            # Simple equal division
            attributed = total_energy / n_concurrent
            # Lower confidence with more concurrent requests
            confidence = 1.0 / n_concurrent
            return attributed, confidence

        elif self.attribution_algorithm == EnergyAttributionAlgorithm.TIME_WEIGHTED:
            # Weight by duration relative to total concurrent time
            # For now, simplified equal share (full implementation would track all overlaps)
            attributed = total_energy / n_concurrent
            confidence = 0.7 / n_concurrent
            return attributed, confidence

        elif self.attribution_algorithm == EnergyAttributionAlgorithm.UTILIZATION_WEIGHTED:
            # Attribute based on GPU utilization contribution
            # Assumes each request contributes proportionally to utilization
            avg_util = sum(s.gpu_utilization for s in samples) / len(samples)

            if avg_util > 0:
                # Estimate this request's share based on per-request utilization
                est_util_per_request = avg_util / n_concurrent
                share = est_util_per_request / avg_util
                attributed = active_energy * share + (self._idle_power_watts * (request.end_time - request.start_time) / n_concurrent)
            else:
                attributed = total_energy / n_concurrent

            confidence = 0.8 / n_concurrent
            return attributed, confidence

        elif self.attribution_algorithm == EnergyAttributionAlgorithm.MARGINAL_COST:
            # Attribute marginal energy above baseline, divided among active requests
            # Plus fair share of idle/baseline power
            duration = request.end_time - request.start_time
            idle_share = (self._idle_power_watts * duration) / n_concurrent
            active_share = active_energy / n_concurrent
            attributed = idle_share + active_share
            confidence = 0.85 / n_concurrent
            return attributed, confidence

        # Default fallback
        return total_energy / n_concurrent, 0.5

    # =========================================================================
    # Data Access Methods
    # =========================================================================

    def get_total_energy(self) -> float:
        """Get total energy consumed since monitoring started (Joules)."""
        with self._lock:
            samples = list(self._samples)
            return self._integrate_energy(samples)

    def get_current_power(self) -> Optional[float]:
        """Get most recent power reading (Watts)."""
        with self._lock:
            if self._samples:
                return self._samples[-1].power_watts
            return None

    def get_samples(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[PowerSample]:
        """Get power samples, optionally filtered by time range."""
        with self._lock:
            samples = list(self._samples)

        if start_time is not None:
            samples = [s for s in samples if s.timestamp >= start_time]
        if end_time is not None:
            samples = [s for s in samples if s.timestamp <= end_time]

        return samples

    def get_statistics(self) -> dict:
        """Get monitoring statistics."""
        with self._lock:
            samples = list(self._samples)
            runtime = time.time() - self._start_time if self._start_time else 0

            if samples:
                power_values = [s.power_watts for s in samples]
                return {
                    "state": self._state.value,
                    "runtime_seconds": runtime,
                    "total_samples": self._total_samples,
                    "missed_samples": self._missed_samples,
                    "sample_rate_hz": self.sample_rate_hz,
                    "buffer_utilization": len(self._samples) / self.buffer_size,
                    "idle_power_watts": self._idle_power_watts,
                    "total_energy_joules": self._integrate_energy(samples),
                    "avg_power_watts": sum(power_values) / len(power_values),
                    "min_power_watts": min(power_values),
                    "max_power_watts": max(power_values),
                    "active_requests": len(self._active_requests),
                    "completed_requests": len(self._completed_attributions)
                }
            else:
                return {
                    "state": self._state.value,
                    "runtime_seconds": runtime,
                    "total_samples": 0,
                    "active_requests": len(self._active_requests)
                }

    def get_attributions(self) -> List[RequestEnergyAttribution]:
        """Get all completed request attributions."""
        with self._lock:
            return list(self._completed_attributions)

    # =========================================================================
    # Data Export
    # =========================================================================

    def export_csv(self, filepath: str) -> None:
        """
        Export power samples to CSV file.

        CSV Schema:
        - timestamp: Unix timestamp (float)
        - timestamp_iso: ISO 8601 formatted timestamp
        - power_watts: Power draw in watts
        - gpu_utilization: GPU utilization percentage
        - memory_used_mb: Memory used in megabytes
        - memory_total_mb: Total memory in megabytes
        - memory_utilization: Memory utilization percentage
        - temperature_c: GPU temperature in Celsius
        - gpu_index: GPU device index
        """
        with self._lock:
            samples = list(self._samples)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            if not samples:
                logger.warning("No samples to export")
                return

            writer = csv.DictWriter(f, fieldnames=samples[0].to_dict().keys())
            writer.writeheader()
            for sample in samples:
                writer.writerow(sample.to_dict())

        logger.info(f"Exported {len(samples)} samples to {filepath}")

    def export_json(self, filepath: str, include_attributions: bool = True) -> None:
        """
        Export all data to JSON file.

        JSON Schema:
        {
            "metadata": {
                "export_timestamp": ISO timestamp,
                "monitoring_duration_seconds": float,
                "sample_rate_hz": float,
                "gpu_index": int,
                "idle_power_watts": float,
                "attribution_algorithm": string
            },
            "statistics": { ... },
            "samples": [ PowerSample objects ],
            "attributions": [ RequestEnergyAttribution objects ]  // if include_attributions
        }
        """
        with self._lock:
            samples = list(self._samples)
            attributions = list(self._completed_attributions) if include_attributions else []
            stats = self.get_statistics()

        data = {
            "metadata": {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "monitoring_duration_seconds": stats.get("runtime_seconds", 0),
                "sample_rate_hz": self.sample_rate_hz,
                "gpu_index": self.gpu_index,
                "idle_power_watts": self._idle_power_watts,
                "attribution_algorithm": self.attribution_algorithm.value
            },
            "statistics": stats,
            "samples": [s.to_dict() for s in samples]
        }

        if include_attributions:
            data["attributions"] = [a.to_dict() for a in attributions]

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported data to {filepath}")

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    def __enter__(self) -> 'PowerMonitor':
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


# =============================================================================
# Validation and Testing Utilities
# =============================================================================

class PowerMonitorValidator:
    """
    Validation utilities for verifying power measurement accuracy.
    """

    @staticmethod
    def validate_sample_timing(
        monitor: PowerMonitor,
        duration_seconds: float = 10.0,
        tolerance_percent: float = 5.0
    ) -> dict:
        """
        Validate that samples are taken at the expected rate.

        Args:
            monitor: Active PowerMonitor instance
            duration_seconds: How long to collect samples
            tolerance_percent: Acceptable deviation from target rate

        Returns:
            Validation results dict
        """
        start_time = time.time()
        time.sleep(duration_seconds)
        end_time = time.time()

        samples = monitor.get_samples(start_time, end_time)
        actual_duration = end_time - start_time
        expected_samples = int(actual_duration * monitor.sample_rate_hz)
        actual_samples = len(samples)

        # Check sample rate accuracy
        rate_error_percent = abs(actual_samples - expected_samples) / expected_samples * 100

        # Check sample interval consistency
        if len(samples) >= 2:
            intervals = [
                samples[i].timestamp - samples[i-1].timestamp
                for i in range(1, len(samples))
            ]
            avg_interval = sum(intervals) / len(intervals)
            expected_interval = 1.0 / monitor.sample_rate_hz
            interval_error_percent = abs(avg_interval - expected_interval) / expected_interval * 100
        else:
            interval_error_percent = 100.0

        passed = rate_error_percent <= tolerance_percent and interval_error_percent <= tolerance_percent

        return {
            "passed": passed,
            "duration_seconds": actual_duration,
            "expected_samples": expected_samples,
            "actual_samples": actual_samples,
            "rate_error_percent": rate_error_percent,
            "interval_error_percent": interval_error_percent,
            "tolerance_percent": tolerance_percent
        }

    @staticmethod
    def validate_energy_conservation(
        monitor: PowerMonitor,
        attributions: List[RequestEnergyAttribution]
    ) -> dict:
        """
        Validate that attributed energy doesn't exceed total measured energy.

        This is a sanity check - attributed energy should be <= total energy.
        """
        if not attributions:
            return {"passed": True, "message": "No attributions to validate"}

        # Find time range covered by attributions
        start_time = min(a.start_time for a in attributions)
        end_time = max(a.end_time for a in attributions)

        # Get total energy in that window
        samples = monitor.get_samples(start_time, end_time)
        total_energy = monitor._integrate_energy(samples)

        # Sum attributed energy
        total_attributed = sum(a.attributed_energy_joules for a in attributions)

        # Attributed should not exceed total (with small tolerance for rounding)
        tolerance = 0.01 * total_energy  # 1% tolerance
        passed = total_attributed <= total_energy + tolerance

        return {
            "passed": passed,
            "total_measured_joules": total_energy,
            "total_attributed_joules": total_attributed,
            "attribution_ratio": total_attributed / total_energy if total_energy > 0 else 0,
            "time_range_seconds": end_time - start_time
        }

    @staticmethod
    def validate_idle_calibration(
        monitor: PowerMonitor,
        expected_idle_range: Tuple[float, float] = (30.0, 100.0)
    ) -> dict:
        """
        Validate that idle power calibration is reasonable.

        Args:
            monitor: PowerMonitor with calibrated idle power
            expected_idle_range: Expected range for idle power (min, max) in watts
        """
        idle_power = monitor._idle_power_watts
        min_expected, max_expected = expected_idle_range

        passed = min_expected <= idle_power <= max_expected

        return {
            "passed": passed,
            "idle_power_watts": idle_power,
            "expected_range": expected_idle_range,
            "calibrated": monitor._idle_calibrated
        }

    @staticmethod
    def cross_validate_with_nvidia_smi(
        monitor: PowerMonitor,
        duration_seconds: float = 5.0
    ) -> dict:
        """
        Cross-validate power readings against nvidia-smi.

        This runs nvidia-smi in parallel and compares readings.
        """
        import subprocess

        try:
            # Get current reading from monitor
            monitor_reading = monitor.get_current_power()

            # Get reading from nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return {"passed": False, "error": "nvidia-smi failed"}

            smi_reading = float(result.stdout.strip().split('\n')[monitor.gpu_index])

            # Compare (allow 5% tolerance)
            if monitor_reading:
                error_percent = abs(monitor_reading - smi_reading) / smi_reading * 100
                passed = error_percent < 5.0
            else:
                error_percent = 100.0
                passed = False

            return {
                "passed": passed,
                "monitor_reading_watts": monitor_reading,
                "nvidia_smi_reading_watts": smi_reading,
                "error_percent": error_percent
            }

        except Exception as e:
            return {"passed": False, "error": str(e)}


# =============================================================================
# Example Usage and Demo
# =============================================================================

def demo_power_monitoring():
    """Demonstrate power monitoring capabilities."""

    print("=" * 60)
    print("Green AI Power Monitoring System Demo")
    print("=" * 60)

    # Create monitor with 10Hz sampling
    monitor = PowerMonitor(
        sample_rate_hz=10,
        attribution_algorithm=EnergyAttributionAlgorithm.UTILIZATION_WEIGHTED,
        auto_calibrate_idle=True
    )

    try:
        # Start monitoring
        monitor.start()
        print("\nMonitor started. Calibrating idle power...")
        time.sleep(2)

        # Simulate some inference requests
        print("\nSimulating inference requests...")

        # Request 1: Short request
        req1 = monitor.begin_request(metadata={"model": "llama-7b", "batch_size": 1})
        time.sleep(0.5)
        attr1 = monitor.end_request(req1)
        print(f"Request 1: {attr1.attributed_energy_joules:.3f}J over {attr1.duration_seconds:.3f}s")

        # Request 2 and 3: Overlapping requests
        req2 = monitor.begin_request(metadata={"model": "llama-7b", "batch_size": 4})
        time.sleep(0.2)
        req3 = monitor.begin_request(metadata={"model": "llama-7b", "batch_size": 2})
        time.sleep(0.3)
        attr2 = monitor.end_request(req2)
        time.sleep(0.2)
        attr3 = monitor.end_request(req3)

        print(f"Request 2 (overlapped): {attr2.attributed_energy_joules:.3f}J, "
              f"confidence: {attr2.confidence_score:.2f}")
        print(f"Request 3 (overlapped): {attr3.attributed_energy_joules:.3f}J, "
              f"confidence: {attr3.confidence_score:.2f}")

        # Get statistics
        print("\n" + "=" * 40)
        print("Monitoring Statistics:")
        print("=" * 40)
        stats = monitor.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")

        # Export data
        print("\nExporting data...")
        monitor.export_csv("/tmp/power_samples.csv")
        monitor.export_json("/tmp/power_data.json")

        # Run validation
        print("\n" + "=" * 40)
        print("Running Validation:")
        print("=" * 40)

        validator = PowerMonitorValidator()

        timing_result = validator.validate_sample_timing(monitor, duration_seconds=3.0)
        print(f"Sample timing validation: {'PASSED' if timing_result['passed'] else 'FAILED'}")
        print(f"  Rate error: {timing_result['rate_error_percent']:.1f}%")

        conservation_result = validator.validate_energy_conservation(
            monitor, monitor.get_attributions()
        )
        print(f"Energy conservation: {'PASSED' if conservation_result['passed'] else 'FAILED'}")
        print(f"  Attribution ratio: {conservation_result['attribution_ratio']:.2f}")

        idle_result = validator.validate_idle_calibration(monitor)
        print(f"Idle calibration: {'PASSED' if idle_result['passed'] else 'FAILED'}")
        print(f"  Idle power: {idle_result['idle_power_watts']:.1f}W")

    finally:
        monitor.stop()
        print("\nMonitor stopped.")


if __name__ == "__main__":
    demo_power_monitoring()
