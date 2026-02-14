"""
Data Export Schemas for Power Monitoring System
================================================

Dr. Park - Systems Engineering Design
Article 4: Green AI Research

This module defines the data schemas for CSV and JSON exports,
enabling consistent data analysis across experiments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import json


# =============================================================================
# CSV Schema Definition
# =============================================================================

CSV_POWER_SAMPLES_SCHEMA = """
Power Samples CSV Schema
========================

File naming convention: power_samples_YYYYMMDD_HHMMSS.csv

Columns:
--------
timestamp           float       Unix timestamp with microsecond precision
timestamp_iso       string      ISO 8601 formatted timestamp (UTC)
power_watts         float       Instantaneous power draw in watts (2 decimal places)
gpu_utilization     float       GPU compute utilization percentage (0-100, 1 decimal)
memory_used_mb      float       GPU memory used in megabytes (2 decimal places)
memory_total_mb     float       Total GPU memory in megabytes (2 decimal places)
memory_utilization  float       Memory utilization percentage (0-100, 1 decimal)
temperature_c       float       GPU temperature in Celsius (1 decimal place)
gpu_index           int         GPU device index (0-based)

Example:
--------
timestamp,timestamp_iso,power_watts,gpu_utilization,memory_used_mb,memory_total_mb,memory_utilization,temperature_c,gpu_index
1705936800.123456,2024-01-22T14:00:00.123456+00:00,152.50,75.0,8192.00,16384.00,50.0,65.0,0
1705936800.223456,2024-01-22T14:00:00.223456+00:00,155.25,78.0,8256.00,16384.00,50.4,66.0,0
"""

CSV_ATTRIBUTIONS_SCHEMA = """
Request Attributions CSV Schema
===============================

File naming convention: request_attributions_YYYYMMDD_HHMMSS.csv

Columns:
--------
request_id                  string      Unique request identifier
start_time                  float       Request start Unix timestamp
end_time                    float       Request end Unix timestamp
duration_seconds            float       Request duration in seconds
total_energy_joules         float       Total energy in window (Joules)
attributed_energy_joules    float       Energy attributed to this request
idle_baseline_watts         float       Estimated idle power during request
active_energy_joules        float       Energy above idle baseline
avg_power_watts             float       Average power during request
peak_power_watts            float       Peak power during request
avg_gpu_utilization         float       Average GPU utilization (%)
avg_memory_utilization      float       Average memory utilization (%)
overlapping_requests        int         Number of concurrent requests
attribution_method          string      Algorithm used for attribution
confidence_score            float       Attribution confidence (0-1)
sample_count                int         Number of power samples in window

Example:
--------
request_id,start_time,end_time,duration_seconds,total_energy_joules,...
abc12345,1705936800.0,1705936801.5,1.5,225.75,180.60,50.0,175.60,150.50,165.25,75.0,45.0,2,utilization_weighted,0.65,15
"""


# =============================================================================
# JSON Schema Definition
# =============================================================================

@dataclass
class JSONExportMetadata:
    """Metadata section of JSON export."""
    export_timestamp: str  # ISO 8601 format
    monitoring_duration_seconds: float
    sample_rate_hz: float
    gpu_index: int
    idle_power_watts: float
    attribution_algorithm: str
    gpu_name: Optional[str] = None
    driver_version: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_notes: Optional[str] = None


@dataclass
class JSONExportStatistics:
    """Statistics section of JSON export."""
    state: str
    runtime_seconds: float
    total_samples: int
    missed_samples: int
    sample_rate_hz: float
    buffer_utilization: float
    idle_power_watts: float
    total_energy_joules: float
    avg_power_watts: float
    min_power_watts: float
    max_power_watts: float
    active_requests: int
    completed_requests: int


@dataclass
class JSONExportSample:
    """Single sample in JSON export."""
    timestamp: float
    timestamp_iso: str
    power_watts: float
    gpu_utilization: float
    memory_used_mb: float
    memory_total_mb: float
    memory_utilization: float
    temperature_c: float
    gpu_index: int


@dataclass
class JSONExportAttribution:
    """Attribution record in JSON export."""
    request_id: str
    start_time: float
    end_time: float
    duration_seconds: float
    total_energy_joules: float
    attributed_energy_joules: float
    idle_baseline_watts: float
    active_energy_joules: float
    avg_power_watts: float
    peak_power_watts: float
    avg_gpu_utilization: float
    avg_memory_utilization: float
    overlapping_requests: int
    attribution_method: str
    confidence_score: float
    sample_count: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class JSONExportRoot:
    """Root structure of JSON export."""
    metadata: JSONExportMetadata
    statistics: JSONExportStatistics
    samples: List[JSONExportSample]
    attributions: List[JSONExportAttribution] = field(default_factory=list)


JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Power Monitoring Export",
    "description": "Green AI power monitoring data export format",
    "type": "object",
    "required": ["metadata", "statistics", "samples"],
    "properties": {
        "metadata": {
            "type": "object",
            "required": [
                "export_timestamp",
                "monitoring_duration_seconds",
                "sample_rate_hz",
                "gpu_index",
                "idle_power_watts",
                "attribution_algorithm"
            ],
            "properties": {
                "export_timestamp": {
                    "type": "string",
                    "format": "date-time",
                    "description": "ISO 8601 timestamp of export"
                },
                "monitoring_duration_seconds": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Total monitoring duration in seconds"
                },
                "sample_rate_hz": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 1000,
                    "description": "Sampling frequency in Hz"
                },
                "gpu_index": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "GPU device index"
                },
                "idle_power_watts": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Calibrated idle power baseline in watts"
                },
                "attribution_algorithm": {
                    "type": "string",
                    "enum": ["equal_share", "time_weighted", "utilization_weighted", "marginal_cost"],
                    "description": "Energy attribution algorithm used"
                },
                "gpu_name": {
                    "type": "string",
                    "description": "GPU model name"
                },
                "driver_version": {
                    "type": "string",
                    "description": "NVIDIA driver version"
                },
                "experiment_id": {
                    "type": "string",
                    "description": "Experiment identifier"
                },
                "experiment_notes": {
                    "type": "string",
                    "description": "Free-form experiment notes"
                }
            }
        },
        "statistics": {
            "type": "object",
            "required": [
                "state",
                "runtime_seconds",
                "total_samples",
                "total_energy_joules"
            ],
            "properties": {
                "state": {
                    "type": "string",
                    "enum": ["stopped", "running", "paused", "error"],
                    "description": "Monitor state at export time"
                },
                "runtime_seconds": {
                    "type": "number",
                    "minimum": 0
                },
                "total_samples": {
                    "type": "integer",
                    "minimum": 0
                },
                "missed_samples": {
                    "type": "integer",
                    "minimum": 0
                },
                "sample_rate_hz": {
                    "type": "number",
                    "minimum": 0
                },
                "buffer_utilization": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "idle_power_watts": {
                    "type": "number",
                    "minimum": 0
                },
                "total_energy_joules": {
                    "type": "number",
                    "minimum": 0
                },
                "avg_power_watts": {
                    "type": "number",
                    "minimum": 0
                },
                "min_power_watts": {
                    "type": "number",
                    "minimum": 0
                },
                "max_power_watts": {
                    "type": "number",
                    "minimum": 0
                },
                "active_requests": {
                    "type": "integer",
                    "minimum": 0
                },
                "completed_requests": {
                    "type": "integer",
                    "minimum": 0
                }
            }
        },
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "timestamp",
                    "power_watts",
                    "gpu_utilization"
                ],
                "properties": {
                    "timestamp": {
                        "type": "number",
                        "description": "Unix timestamp"
                    },
                    "timestamp_iso": {
                        "type": "string",
                        "format": "date-time"
                    },
                    "power_watts": {
                        "type": "number",
                        "minimum": 0
                    },
                    "gpu_utilization": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "memory_used_mb": {
                        "type": "number",
                        "minimum": 0
                    },
                    "memory_total_mb": {
                        "type": "number",
                        "minimum": 0
                    },
                    "memory_utilization": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "temperature_c": {
                        "type": "number"
                    },
                    "gpu_index": {
                        "type": "integer",
                        "minimum": 0
                    }
                }
            }
        },
        "attributions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "request_id",
                    "start_time",
                    "end_time",
                    "attributed_energy_joules"
                ],
                "properties": {
                    "request_id": {
                        "type": "string"
                    },
                    "start_time": {
                        "type": "number"
                    },
                    "end_time": {
                        "type": "number"
                    },
                    "duration_seconds": {
                        "type": "number",
                        "minimum": 0
                    },
                    "total_energy_joules": {
                        "type": "number",
                        "minimum": 0
                    },
                    "attributed_energy_joules": {
                        "type": "number",
                        "minimum": 0
                    },
                    "idle_baseline_watts": {
                        "type": "number",
                        "minimum": 0
                    },
                    "active_energy_joules": {
                        "type": "number",
                        "minimum": 0
                    },
                    "avg_power_watts": {
                        "type": "number",
                        "minimum": 0
                    },
                    "peak_power_watts": {
                        "type": "number",
                        "minimum": 0
                    },
                    "avg_gpu_utilization": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "avg_memory_utilization": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "overlapping_requests": {
                        "type": "integer",
                        "minimum": 0
                    },
                    "attribution_method": {
                        "type": "string"
                    },
                    "confidence_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    },
                    "sample_count": {
                        "type": "integer",
                        "minimum": 0
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Request-specific metadata"
                    }
                }
            }
        }
    }
}


def get_json_schema() -> dict:
    """Return the JSON schema for validation."""
    return JSON_SCHEMA


def validate_json_export(data: dict) -> tuple[bool, list[str]]:
    """
    Validate JSON export data against schema.

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    # Check required top-level keys
    for key in ["metadata", "statistics", "samples"]:
        if key not in data:
            errors.append(f"Missing required key: {key}")

    if errors:
        return False, errors

    # Validate metadata
    metadata = data.get("metadata", {})
    required_metadata = [
        "export_timestamp",
        "monitoring_duration_seconds",
        "sample_rate_hz",
        "gpu_index",
        "idle_power_watts",
        "attribution_algorithm"
    ]
    for key in required_metadata:
        if key not in metadata:
            errors.append(f"Missing required metadata field: {key}")

    # Validate samples
    samples = data.get("samples", [])
    for i, sample in enumerate(samples):
        if "timestamp" not in sample:
            errors.append(f"Sample {i}: missing timestamp")
        if "power_watts" not in sample:
            errors.append(f"Sample {i}: missing power_watts")

    # Validate attributions if present
    attributions = data.get("attributions", [])
    for i, attr in enumerate(attributions):
        if "request_id" not in attr:
            errors.append(f"Attribution {i}: missing request_id")
        if "attributed_energy_joules" not in attr:
            errors.append(f"Attribution {i}: missing attributed_energy_joules")

    return len(errors) == 0, errors


def export_schema_documentation(filepath: str) -> None:
    """Export schema documentation to a file."""
    doc = f"""
# Power Monitoring Data Export Schemas

Generated: {datetime.now().isoformat()}

## Overview

This document describes the data export formats for the Green AI power monitoring system.
Two formats are supported: CSV for tabular data analysis and JSON for rich structured data.

---

{CSV_POWER_SAMPLES_SCHEMA}

---

{CSV_ATTRIBUTIONS_SCHEMA}

---

## JSON Schema

The JSON export format provides comprehensive data including metadata, statistics,
power samples, and energy attributions.

```json
{json.dumps(JSON_SCHEMA, indent=2)}
```

---

## Data Quality Notes

### Timestamp Precision
- Timestamps are recorded with microsecond precision
- Both Unix timestamp (float) and ISO 8601 (string) are provided
- All timestamps are in UTC

### Power Measurement
- Power values from NVML are in milliwatts, converted to watts
- Sampling at 10Hz provides good balance of resolution and overhead
- Trapezoidal integration is used for energy calculations

### Energy Attribution
- Multiple algorithms available for handling overlapping requests
- Confidence scores indicate attribution reliability
- Lower confidence with more concurrent requests

### Known Limitations
- GPU idle power varies with temperature and driver state
- NVML may have up to 10ms latency on power readings
- Some GPU models have limited power reporting resolution
"""

    with open(filepath, 'w') as f:
        f.write(doc)


if __name__ == "__main__":
    # Generate schema documentation
    export_schema_documentation("/tmp/power_monitoring_schemas.md")
    print("Schema documentation exported to /tmp/power_monitoring_schemas.md")
