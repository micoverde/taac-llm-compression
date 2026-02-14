"""
Edge Case Handling for Power Monitoring System
===============================================

Dr. Park - Systems Engineering Design
Article 4: Green AI Research

This module handles edge cases in power monitoring:
1. GPU idle power baseline tracking
2. Overlapping request attribution
3. Measurement gaps and recovery
4. Thermal throttling detection
5. Power limit capping
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
from collections import deque
import statistics

logger = logging.getLogger(__name__)


class PowerState(Enum):
    """GPU power states for edge case handling."""
    IDLE = "idle"
    ACTIVE = "active"
    THERMAL_THROTTLED = "thermal_throttled"
    POWER_CAPPED = "power_capped"
    UNKNOWN = "unknown"


@dataclass
class IdlePowerTracker:
    """
    Tracks GPU idle power baseline over time.

    Edge Case 1: GPU Idle Power

    Problem:
    - GPU idle power varies with temperature, driver state, and background processes
    - Static idle baseline becomes inaccurate over time
    - Need to distinguish "true idle" from "light workload"

    Solution:
    - Maintain rolling window of low-utilization samples
    - Use statistical methods to estimate current idle baseline
    - Adapt baseline as conditions change
    """

    window_size: int = 1000  # Samples to consider for idle estimation
    utilization_threshold: float = 5.0  # Max GPU util % to consider "idle"
    update_interval_seconds: float = 60.0  # How often to recalculate

    # Internal state
    _idle_samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    _current_baseline: float = 50.0  # Default estimate
    _last_update: float = 0.0
    _baseline_history: List[Tuple[float, float]] = field(default_factory=list)

    def add_sample(self, power_watts: float, gpu_utilization: float, timestamp: float) -> None:
        """Add a power sample, updating idle tracking if low utilization."""
        if gpu_utilization <= self.utilization_threshold:
            self._idle_samples.append((timestamp, power_watts))

            # Periodically recalculate baseline
            if timestamp - self._last_update >= self.update_interval_seconds:
                self._recalculate_baseline(timestamp)

    def _recalculate_baseline(self, current_time: float) -> None:
        """Recalculate idle power baseline from recent samples."""
        if len(self._idle_samples) < 10:
            return  # Not enough data

        # Use samples from last 5 minutes
        cutoff_time = current_time - 300
        recent_samples = [
            power for ts, power in self._idle_samples
            if ts >= cutoff_time
        ]

        if len(recent_samples) >= 10:
            # Use 10th percentile to filter out brief activity spikes
            sorted_samples = sorted(recent_samples)
            percentile_idx = max(1, len(sorted_samples) // 10)
            new_baseline = statistics.mean(sorted_samples[:percentile_idx * 2])

            # Smooth transition to avoid jumps
            alpha = 0.3  # Smoothing factor
            self._current_baseline = (
                alpha * new_baseline + (1 - alpha) * self._current_baseline
            )

            self._baseline_history.append((current_time, self._current_baseline))
            self._last_update = current_time

            logger.debug(f"Updated idle baseline: {self._current_baseline:.1f}W")

    def get_baseline(self) -> float:
        """Get current idle power baseline estimate."""
        return self._current_baseline

    def get_baseline_at_time(self, timestamp: float) -> float:
        """Get historical idle baseline for a specific time."""
        if not self._baseline_history:
            return self._current_baseline

        # Find closest historical baseline
        for i, (ts, baseline) in enumerate(reversed(self._baseline_history)):
            if ts <= timestamp:
                return baseline

        return self._baseline_history[0][1] if self._baseline_history else self._current_baseline


@dataclass
class OverlapResolver:
    """
    Resolves energy attribution for overlapping requests.

    Edge Case 2: Overlapping Requests

    Problem:
    - Multiple concurrent inference requests share GPU resources
    - Power consumption is not simply additive
    - Need fair attribution that sums to actual consumption

    Solution:
    - Track all active requests with timestamps
    - Use time-slice based attribution
    - Apply weighting based on resource usage indicators
    """

    @staticmethod
    def resolve_time_sliced(
        target_request_id: str,
        target_start: float,
        target_end: float,
        all_requests: Dict[str, Tuple[float, float]],  # id -> (start, end)
        samples: List[Tuple[float, float]],  # (timestamp, power)
        idle_power: float
    ) -> Tuple[float, float, float]:
        """
        Resolve energy attribution using time-slicing.

        For each time interval between samples:
        1. Count active requests
        2. Divide power equally among active requests
        3. Sum attributed power for target request

        Returns:
            Tuple of (attributed_energy, confidence, overlap_factor)
        """
        if not samples or len(samples) < 2:
            return 0.0, 0.0, 0.0

        attributed_energy = 0.0
        total_energy = 0.0
        max_concurrent = 1

        for i in range(1, len(samples)):
            t1, p1 = samples[i-1]
            t2, p2 = samples[i]

            # Time midpoint for concurrency check
            t_mid = (t1 + t2) / 2
            dt = t2 - t1
            avg_power = (p1 + p2) / 2

            # Count concurrent requests at this time
            concurrent_count = 0
            target_active = False

            for req_id, (req_start, req_end) in all_requests.items():
                if req_start <= t_mid <= req_end:
                    concurrent_count += 1
                    if req_id == target_request_id:
                        target_active = True

            max_concurrent = max(max_concurrent, concurrent_count)

            # Calculate energy for this interval
            interval_energy = avg_power * dt
            total_energy += interval_energy

            # Attribute to target if active
            if target_active and concurrent_count > 0:
                # Split active power (above idle) among concurrent requests
                active_power = max(0, avg_power - idle_power)
                idle_share = idle_power / concurrent_count
                active_share = active_power / concurrent_count
                attributed_energy += (idle_share + active_share) * dt

        # Confidence decreases with more concurrent requests
        confidence = 1.0 / max_concurrent if max_concurrent > 0 else 0.0

        # Overlap factor is average concurrency
        overlap_factor = max_concurrent

        return attributed_energy, confidence, overlap_factor

    @staticmethod
    def resolve_utilization_weighted(
        target_request_id: str,
        target_start: float,
        target_end: float,
        all_requests: Dict[str, Tuple[float, float, float]],  # id -> (start, end, relative_weight)
        samples: List[Tuple[float, float]],
        idle_power: float
    ) -> Tuple[float, float]:
        """
        Resolve using utilization-based weighting.

        Each request has a relative weight (e.g., based on model size, batch size).
        Energy is attributed proportionally to these weights.
        """
        if not samples:
            return 0.0, 0.0

        # Calculate total energy
        total_energy = 0.0
        for i in range(1, len(samples)):
            dt = samples[i][0] - samples[i-1][0]
            avg_power = (samples[i][1] + samples[i-1][1]) / 2
            total_energy += avg_power * dt

        # Get target weight
        if target_request_id not in all_requests:
            return 0.0, 0.0

        target_start, target_end, target_weight = all_requests[target_request_id]

        # Calculate total weight of concurrent requests
        total_weight = 0.0
        for req_id, (start, end, weight) in all_requests.items():
            # Check if overlaps with target
            if start < target_end and end > target_start:
                # Weight by overlap duration
                overlap_start = max(start, target_start)
                overlap_end = min(end, target_end)
                overlap_duration = overlap_end - overlap_start
                target_duration = target_end - target_start
                overlap_fraction = overlap_duration / target_duration if target_duration > 0 else 0
                total_weight += weight * overlap_fraction

        if total_weight <= 0:
            return total_energy, 1.0

        # Attribute proportionally
        attributed = total_energy * (target_weight / total_weight)
        confidence = target_weight / total_weight

        return attributed, confidence


@dataclass
class MeasurementGapHandler:
    """
    Handles gaps in power measurements.

    Edge Case 3: Measurement Gaps

    Problem:
    - Sampling thread may be delayed (GC, OS scheduling)
    - GPU driver may not respond immediately
    - System may enter sleep/hibernate
    - Data corruption or sensor failures

    Solution:
    - Detect gaps in sample timestamps
    - Interpolate or extrapolate for small gaps
    - Mark large gaps as "unknown" energy
    - Log warnings for analysis
    """

    max_acceptable_gap_seconds: float = 0.5  # Gaps larger than this are problematic
    interpolation_limit_seconds: float = 1.0  # Maximum gap to interpolate across

    def detect_gaps(
        self,
        samples: List[Tuple[float, float]],
        expected_interval: float
    ) -> List[Dict]:
        """
        Detect gaps in the sample sequence.

        Returns list of gap events with metadata.
        """
        gaps = []
        tolerance = expected_interval * 1.5  # 50% tolerance

        for i in range(1, len(samples)):
            dt = samples[i][0] - samples[i-1][0]

            if dt > tolerance:
                gap_info = {
                    "index": i,
                    "start_time": samples[i-1][0],
                    "end_time": samples[i][0],
                    "duration_seconds": dt,
                    "expected_samples": int(dt / expected_interval) - 1,
                    "severity": "warning" if dt < self.interpolation_limit_seconds else "error"
                }
                gaps.append(gap_info)

        return gaps

    def interpolate_gap(
        self,
        before_sample: Tuple[float, float],
        after_sample: Tuple[float, float],
        expected_interval: float
    ) -> List[Tuple[float, float]]:
        """
        Generate interpolated samples to fill a gap.

        Uses linear interpolation between boundary samples.
        """
        t1, p1 = before_sample
        t2, p2 = after_sample
        gap_duration = t2 - t1

        if gap_duration > self.interpolation_limit_seconds:
            logger.warning(
                f"Gap too large to interpolate: {gap_duration:.2f}s > {self.interpolation_limit_seconds}s"
            )
            return []

        interpolated = []
        num_samples = int(gap_duration / expected_interval) - 1

        for i in range(1, num_samples + 1):
            fraction = i / (num_samples + 1)
            t = t1 + fraction * gap_duration
            p = p1 + fraction * (p2 - p1)
            interpolated.append((t, p))

        return interpolated

    def estimate_gap_energy(
        self,
        before_sample: Tuple[float, float],
        after_sample: Tuple[float, float],
        method: str = "linear"
    ) -> Tuple[float, float]:
        """
        Estimate energy consumed during a measurement gap.

        Returns:
            Tuple of (estimated_energy, uncertainty)
        """
        t1, p1 = before_sample
        t2, p2 = after_sample
        duration = t2 - t1

        if method == "linear":
            # Linear interpolation
            avg_power = (p1 + p2) / 2
            energy = avg_power * duration
            # Uncertainty increases with gap size and power difference
            power_uncertainty = abs(p2 - p1) / 2
            uncertainty = power_uncertainty * duration

        elif method == "conservative":
            # Use lower bound (minimum power)
            min_power = min(p1, p2)
            energy = min_power * duration
            uncertainty = abs(p2 - p1) * duration / 2

        elif method == "aggressive":
            # Use upper bound (maximum power)
            max_power = max(p1, p2)
            energy = max_power * duration
            uncertainty = abs(p2 - p1) * duration / 2

        else:
            raise ValueError(f"Unknown method: {method}")

        return energy, uncertainty


@dataclass
class ThermalThrottleDetector:
    """
    Detects GPU thermal throttling.

    Edge Case 4: Thermal Throttling

    Problem:
    - GPU may reduce performance when hot
    - Power readings may drop unexpectedly
    - Energy attribution becomes inaccurate

    Solution:
    - Monitor temperature alongside power
    - Detect correlation between temp increase and power decrease
    - Flag affected measurements
    """

    throttle_temp_threshold_c: float = 80.0
    power_drop_threshold_percent: float = 15.0

    _temp_history: deque = field(default_factory=lambda: deque(maxlen=100))
    _power_history: deque = field(default_factory=lambda: deque(maxlen=100))
    _throttle_events: List[Dict] = field(default_factory=list)

    def add_sample(
        self,
        timestamp: float,
        temperature_c: float,
        power_watts: float
    ) -> Optional[Dict]:
        """
        Add a sample and check for throttling.

        Returns throttle event dict if detected, None otherwise.
        """
        self._temp_history.append((timestamp, temperature_c))
        self._power_history.append((timestamp, power_watts))

        # Check for throttling conditions
        if temperature_c >= self.throttle_temp_threshold_c:
            # Check if power dropped
            if len(self._power_history) >= 10:
                recent_powers = [p for _, p in list(self._power_history)[-10:]]
                older_powers = [p for _, p in list(self._power_history)[-20:-10]]

                if older_powers:
                    recent_avg = sum(recent_powers) / len(recent_powers)
                    older_avg = sum(older_powers) / len(older_powers)

                    if older_avg > 0:
                        drop_percent = (older_avg - recent_avg) / older_avg * 100

                        if drop_percent >= self.power_drop_threshold_percent:
                            event = {
                                "timestamp": timestamp,
                                "temperature_c": temperature_c,
                                "power_drop_percent": drop_percent,
                                "power_before": older_avg,
                                "power_after": recent_avg,
                                "state": PowerState.THERMAL_THROTTLED.value
                            }
                            self._throttle_events.append(event)
                            logger.warning(
                                f"Thermal throttling detected: {temperature_c:.1f}C, "
                                f"power dropped {drop_percent:.1f}%"
                            )
                            return event

        return None

    def is_throttling(self) -> bool:
        """Check if GPU is currently throttling."""
        if not self._temp_history:
            return False

        _, latest_temp = self._temp_history[-1]
        return latest_temp >= self.throttle_temp_threshold_c

    def get_throttle_events(self) -> List[Dict]:
        """Get all recorded throttle events."""
        return self._throttle_events.copy()


@dataclass
class PowerLimitDetector:
    """
    Detects when GPU hits power limit.

    Edge Case 5: Power Limit Capping

    Problem:
    - GPUs have configurable power limits
    - Hitting the limit caps performance
    - Energy measurements may plateau unexpectedly

    Solution:
    - Query GPU power limit via NVML
    - Detect sustained readings near the limit
    - Flag affected measurements
    """

    power_limit_watts: float = 300.0  # Default, should be queried from GPU
    proximity_threshold_percent: float = 5.0  # How close to limit to flag

    _near_limit_samples: int = 0
    _total_samples: int = 0

    def set_power_limit(self, power_limit_watts: float) -> None:
        """Set the GPU power limit (should be queried from NVML)."""
        self.power_limit_watts = power_limit_watts
        logger.info(f"GPU power limit set to {power_limit_watts:.1f}W")

    def add_sample(self, power_watts: float) -> bool:
        """
        Add a power sample and check if near limit.

        Returns True if sample is near the power limit.
        """
        self._total_samples += 1

        threshold = self.power_limit_watts * (1 - self.proximity_threshold_percent / 100)

        if power_watts >= threshold:
            self._near_limit_samples += 1
            return True

        return False

    def get_limit_hit_ratio(self) -> float:
        """Get ratio of samples that hit the power limit."""
        if self._total_samples == 0:
            return 0.0
        return self._near_limit_samples / self._total_samples

    def is_power_limited(self, lookback_samples: int = 10) -> bool:
        """
        Check if GPU is currently power-limited.

        Returns True if most recent samples are near the limit.
        """
        # This would need access to recent sample history
        # For now, return based on overall ratio
        return self.get_limit_hit_ratio() > 0.5


class EdgeCaseManager:
    """
    Unified manager for all edge case handlers.

    Integrates:
    - Idle power tracking
    - Overlap resolution
    - Measurement gap handling
    - Thermal throttle detection
    - Power limit detection
    """

    def __init__(
        self,
        sample_interval: float = 0.1,
        power_limit_watts: Optional[float] = None
    ):
        self.sample_interval = sample_interval

        self.idle_tracker = IdlePowerTracker()
        self.overlap_resolver = OverlapResolver()
        self.gap_handler = MeasurementGapHandler()
        self.thermal_detector = ThermalThrottleDetector()
        self.power_limit_detector = PowerLimitDetector()

        if power_limit_watts:
            self.power_limit_detector.set_power_limit(power_limit_watts)

        self._samples: List[Tuple[float, float, float, float]] = []  # ts, power, gpu_util, temp

    def process_sample(
        self,
        timestamp: float,
        power_watts: float,
        gpu_utilization: float,
        temperature_c: float
    ) -> Dict:
        """
        Process a power sample through all edge case handlers.

        Returns status dict with any detected issues.
        """
        status = {
            "timestamp": timestamp,
            "issues": [],
            "state": PowerState.ACTIVE if gpu_utilization > 5 else PowerState.IDLE
        }

        # Update idle tracker
        self.idle_tracker.add_sample(power_watts, gpu_utilization, timestamp)

        # Check for gaps
        if self._samples:
            last_ts = self._samples[-1][0]
            gap = timestamp - last_ts

            if gap > self.sample_interval * 1.5:
                status["issues"].append({
                    "type": "measurement_gap",
                    "duration_seconds": gap,
                    "missed_samples": int(gap / self.sample_interval) - 1
                })

        # Check thermal throttling
        throttle_event = self.thermal_detector.add_sample(
            timestamp, temperature_c, power_watts
        )
        if throttle_event:
            status["issues"].append({
                "type": "thermal_throttle",
                "event": throttle_event
            })
            status["state"] = PowerState.THERMAL_THROTTLED

        # Check power limit
        if self.power_limit_detector.add_sample(power_watts):
            status["issues"].append({
                "type": "power_limit",
                "power_watts": power_watts,
                "limit_watts": self.power_limit_detector.power_limit_watts
            })
            if status["state"] != PowerState.THERMAL_THROTTLED:
                status["state"] = PowerState.POWER_CAPPED

        # Store sample
        self._samples.append((timestamp, power_watts, gpu_utilization, temperature_c))

        return status

    def get_adjusted_energy(
        self,
        start_time: float,
        end_time: float,
        include_gap_estimates: bool = True
    ) -> Dict:
        """
        Calculate energy with edge case adjustments.

        Returns dict with energy and uncertainty estimates.
        """
        # Filter samples in time range
        samples = [
            (ts, power) for ts, power, _, _ in self._samples
            if start_time <= ts <= end_time
        ]

        if len(samples) < 2:
            return {
                "energy_joules": 0.0,
                "uncertainty_joules": 0.0,
                "gap_energy_joules": 0.0,
                "sample_count": len(samples)
            }

        # Calculate base energy
        base_energy = 0.0
        for i in range(1, len(samples)):
            dt = samples[i][0] - samples[i-1][0]
            avg_power = (samples[i][1] + samples[i-1][1]) / 2
            base_energy += avg_power * dt

        # Detect and handle gaps
        gaps = self.gap_handler.detect_gaps(samples, self.sample_interval)
        gap_energy = 0.0
        gap_uncertainty = 0.0

        if include_gap_estimates:
            for gap in gaps:
                idx = gap["index"]
                before = samples[idx - 1]
                after = samples[idx]
                energy, uncertainty = self.gap_handler.estimate_gap_energy(before, after)
                gap_energy += energy
                gap_uncertainty += uncertainty

        return {
            "energy_joules": base_energy + gap_energy,
            "base_energy_joules": base_energy,
            "gap_energy_joules": gap_energy,
            "uncertainty_joules": gap_uncertainty,
            "gap_count": len(gaps),
            "sample_count": len(samples),
            "idle_baseline_watts": self.idle_tracker.get_baseline()
        }

    def get_health_report(self) -> Dict:
        """Generate a health report for the monitoring system."""
        total_samples = len(self._samples)

        if total_samples < 10:
            return {"status": "insufficient_data", "sample_count": total_samples}

        # Calculate gap statistics
        gaps = self.gap_handler.detect_gaps(
            [(ts, power) for ts, power, _, _ in self._samples],
            self.sample_interval
        )

        return {
            "status": "healthy" if len(gaps) < total_samples * 0.01 else "degraded",
            "sample_count": total_samples,
            "gap_count": len(gaps),
            "gap_ratio": len(gaps) / total_samples if total_samples > 0 else 0,
            "idle_power_watts": self.idle_tracker.get_baseline(),
            "throttle_events": len(self.thermal_detector.get_throttle_events()),
            "power_limit_hit_ratio": self.power_limit_detector.get_limit_hit_ratio(),
            "is_currently_throttling": self.thermal_detector.is_throttling()
        }


# =============================================================================
# Demo and Testing
# =============================================================================

def demo_edge_cases():
    """Demonstrate edge case handling."""
    import random

    print("=" * 60)
    print("Edge Case Handling Demo")
    print("=" * 60)

    manager = EdgeCaseManager(sample_interval=0.1, power_limit_watts=250.0)

    # Simulate 30 seconds of data with various edge cases
    current_time = time.time()

    for i in range(300):
        # Simulate timestamp (with occasional gaps)
        if i == 100:
            # Simulate a gap
            current_time += 0.5
        else:
            current_time += 0.1

        # Simulate power (with occasional spikes)
        base_power = 100 + random.gauss(0, 5)
        if 150 <= i <= 180:
            # Simulate high load
            base_power += 120

        # Simulate temperature
        temp = 50 + (base_power - 100) * 0.2 + random.gauss(0, 2)

        # Simulate thermal throttling scenario
        if i > 200 and temp > 75:
            temp = 82
            base_power *= 0.8

        # Simulate GPU utilization
        gpu_util = max(0, min(100, (base_power - 80) / 1.5))

        status = manager.process_sample(
            timestamp=current_time,
            power_watts=base_power,
            gpu_utilization=gpu_util,
            temperature_c=temp
        )

        if status["issues"]:
            print(f"Sample {i}: {status['issues']}")

    # Get final report
    print("\n" + "=" * 40)
    print("Health Report:")
    print("=" * 40)
    report = manager.get_health_report()
    for key, value in report.items():
        print(f"  {key}: {value}")

    # Get energy calculation
    print("\n" + "=" * 40)
    print("Energy Calculation:")
    print("=" * 40)
    samples = manager._samples
    if samples:
        energy = manager.get_adjusted_energy(samples[0][0], samples[-1][0])
        for key, value in energy.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    demo_edge_cases()
