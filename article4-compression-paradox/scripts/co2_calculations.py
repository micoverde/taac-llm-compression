#!/usr/bin/env python3
"""
CO2 Savings Calculator for AI Inference Optimization

Calculates datacenter-scale CO2 savings from:
1. Prompt compression (reducing token count)
2. Intelligent routing (directing queries to appropriately-sized models)

Based on Phase 2 experimental data and IEA energy estimates.

References:
- IEA World Energy Outlook 2024: AI datacenter energy projections
- Phase 2 compression experiments: 4.89% energy reduction
- Carbon intensity: EPA eGRID, IEA global averages
"""

import json
import argparse
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


# =============================================================================
# Constants and Reference Data
# =============================================================================

# Global AI inference energy consumption estimates (TWh/year)
# Source: IEA World Energy Outlook 2024, various industry analyses
AI_ENERGY_ESTIMATES = {
    "low": 50,      # Conservative: AI inference only
    "medium": 75,   # Mid-range estimate
    "high": 100,    # Optimistic: Including training spillover
}

# Carbon intensity (gCO2/kWh) by region/scenario
# Source: IEA, EPA eGRID, national grid data
CARBON_INTENSITY = {
    "low": 300,           # Renewable-heavy grids (Nordic, France)
    "global_average": 436, # IEA global average 2024
    "medium": 450,        # Mixed grid (US average)
    "high": 600,          # Coal-heavy grids (China, India avg)
}

# Our experimental results from Phase 2
COMPRESSION_SAVINGS_PERCENT = 4.89  # Energy reduction from minimal vs full prompts

# Routing assumptions
ROUTING_ASSUMPTIONS = {
    "conservative": {
        "routable_fraction": 0.30,  # 30% of queries can use smaller models
        "energy_ratio": 0.15,       # Smaller model uses 15% of large model energy
    },
    "moderate": {
        "routable_fraction": 0.40,  # 40% of queries can use smaller models
        "energy_ratio": 0.10,       # Smaller model uses 10% of large model energy
    },
    "optimistic": {
        "routable_fraction": 0.50,  # 50% of queries can use smaller models
        "energy_ratio": 0.08,       # Smaller model uses 8% of large model energy
    },
}

# Equivalence factors for CO2 (metric tons)
# Source: EPA Greenhouse Gas Equivalencies Calculator
EQUIVALENCE_FACTORS = {
    "cars_annual_mt": 4.6,           # MT CO2 per car per year (US average)
    "flights_transatlantic_mt": 0.9,  # MT CO2 per passenger (NYC-London round trip)
    "trees_annual_mt": 0.022,         # MT CO2 absorbed per tree per year
    "homes_annual_mt": 7.5,           # MT CO2 per home per year (US average)
    "gallons_gasoline_mt": 0.00887,   # MT CO2 per gallon gasoline
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class EnergyScenario:
    """Represents an energy consumption scenario."""
    name: str
    energy_twh: float
    carbon_intensity_g_kwh: float

    @property
    def total_co2_mt(self) -> float:
        """Total CO2 emissions in metric tons."""
        # TWh -> kWh: multiply by 1e9
        # gCO2 -> MT CO2: divide by 1e6
        # Net: multiply by 1e3
        return self.energy_twh * self.carbon_intensity_g_kwh * 1e3


@dataclass
class SavingsResult:
    """Results from a savings calculation."""
    scenario_name: str
    total_co2_mt: float
    savings_percent: float
    co2_saved_mt: float

    # Equivalences
    cars_removed: int
    flights_avoided: int
    trees_equivalent: int
    homes_powered: int

    # Calculation details
    calculation_formula: str
    assumptions: dict


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""
    parameter: str
    low_value: float
    mid_value: float
    high_value: float
    low_savings_mt: float
    mid_savings_mt: float
    high_savings_mt: float


# =============================================================================
# Calculation Functions
# =============================================================================

def calculate_compression_savings(scenario: EnergyScenario,
                                   compression_percent: float = COMPRESSION_SAVINGS_PERCENT) -> SavingsResult:
    """
    Calculate CO2 savings from prompt compression alone.

    Formula: CO2_saved = Total_CO2 * (compression_percent / 100)

    Args:
        scenario: Energy consumption scenario
        compression_percent: Energy reduction from compression (default: 4.89%)

    Returns:
        SavingsResult with CO2 savings and equivalences
    """
    savings_fraction = compression_percent / 100
    co2_saved = scenario.total_co2_mt * savings_fraction

    return SavingsResult(
        scenario_name=f"Compression Only ({scenario.name})",
        total_co2_mt=scenario.total_co2_mt,
        savings_percent=compression_percent,
        co2_saved_mt=co2_saved,
        cars_removed=int(co2_saved / EQUIVALENCE_FACTORS["cars_annual_mt"]),
        flights_avoided=int(co2_saved / EQUIVALENCE_FACTORS["flights_transatlantic_mt"]),
        trees_equivalent=int(co2_saved / EQUIVALENCE_FACTORS["trees_annual_mt"]),
        homes_powered=int(co2_saved / EQUIVALENCE_FACTORS["homes_annual_mt"]),
        calculation_formula=f"CO2_saved = {scenario.total_co2_mt:.2f} MT * {compression_percent:.2f}% = {co2_saved:.2f} MT",
        assumptions={
            "compression_percent": compression_percent,
            "source": "Phase 2 experimental data (minimal vs full prompts)",
            "energy_twh": scenario.energy_twh,
            "carbon_intensity_g_kwh": scenario.carbon_intensity_g_kwh,
        }
    )


def calculate_routing_savings(scenario: EnergyScenario,
                               routing_config: str = "moderate") -> SavingsResult:
    """
    Calculate CO2 savings from intelligent model routing.

    Formula:
        Energy_saved = routable_fraction * (1 - energy_ratio)
        CO2_saved = Total_CO2 * Energy_saved

    Args:
        scenario: Energy consumption scenario
        routing_config: One of "conservative", "moderate", "optimistic"

    Returns:
        SavingsResult with CO2 savings and equivalences
    """
    config = ROUTING_ASSUMPTIONS[routing_config]
    routable = config["routable_fraction"]
    energy_ratio = config["energy_ratio"]

    # Energy saved = fraction routed * (1 - ratio of energy used by small model)
    savings_fraction = routable * (1 - energy_ratio)
    savings_percent = savings_fraction * 100
    co2_saved = scenario.total_co2_mt * savings_fraction

    formula = (
        f"Energy_saved = {routable:.0%} routable * (1 - {energy_ratio:.0%} energy_ratio) = {savings_fraction:.2%}\n"
        f"CO2_saved = {scenario.total_co2_mt:.2f} MT * {savings_fraction:.2%} = {co2_saved:.2f} MT"
    )

    return SavingsResult(
        scenario_name=f"Routing Only ({routing_config}, {scenario.name})",
        total_co2_mt=scenario.total_co2_mt,
        savings_percent=savings_percent,
        co2_saved_mt=co2_saved,
        cars_removed=int(co2_saved / EQUIVALENCE_FACTORS["cars_annual_mt"]),
        flights_avoided=int(co2_saved / EQUIVALENCE_FACTORS["flights_transatlantic_mt"]),
        trees_equivalent=int(co2_saved / EQUIVALENCE_FACTORS["trees_annual_mt"]),
        homes_powered=int(co2_saved / EQUIVALENCE_FACTORS["homes_annual_mt"]),
        calculation_formula=formula,
        assumptions={
            "routing_config": routing_config,
            "routable_fraction": routable,
            "energy_ratio": energy_ratio,
            "interpretation": f"{routable:.0%} of queries routed to models using {energy_ratio:.0%} of the energy",
        }
    )


def calculate_combined_savings(scenario: EnergyScenario,
                                compression_percent: float = COMPRESSION_SAVINGS_PERCENT,
                                routing_config: str = "moderate") -> SavingsResult:
    """
    Calculate CO2 savings from both compression AND routing.

    Formula (sequential application):
        1. Apply compression to all queries: remaining = (1 - compression%)
        2. Apply routing to remaining: routing_savings = remaining * routable * (1 - energy_ratio)
        3. Total savings = compression% + routing_savings

    Args:
        scenario: Energy consumption scenario
        compression_percent: Energy reduction from compression
        routing_config: Routing assumption set

    Returns:
        SavingsResult with combined CO2 savings
    """
    config = ROUTING_ASSUMPTIONS[routing_config]
    routable = config["routable_fraction"]
    energy_ratio = config["energy_ratio"]

    compression_fraction = compression_percent / 100

    # After compression, remaining energy
    remaining_after_compression = 1 - compression_fraction

    # Routing savings on the remaining energy
    routing_savings_on_remaining = routable * (1 - energy_ratio)

    # Total savings = compression + (remaining * routing_savings)
    # Alternative formulation: 1 - (1 - compression) * (1 - routing_on_all)
    total_savings_fraction = compression_fraction + (remaining_after_compression * routing_savings_on_remaining)

    total_savings_percent = total_savings_fraction * 100
    co2_saved = scenario.total_co2_mt * total_savings_fraction

    formula = (
        f"Step 1 - Compression: {compression_percent:.2f}% savings on all queries\n"
        f"Step 2 - Remaining energy: {remaining_after_compression:.2%}\n"
        f"Step 3 - Routing savings: {remaining_after_compression:.2%} * {routable:.0%} * (1 - {energy_ratio:.0%}) = {remaining_after_compression * routing_savings_on_remaining:.2%}\n"
        f"Total savings: {compression_percent:.2f}% + {remaining_after_compression * routing_savings_on_remaining * 100:.2f}% = {total_savings_percent:.2f}%\n"
        f"CO2_saved = {scenario.total_co2_mt:.2f} MT * {total_savings_fraction:.2%} = {co2_saved:.2f} MT"
    )

    return SavingsResult(
        scenario_name=f"Combined ({routing_config}, {scenario.name})",
        total_co2_mt=scenario.total_co2_mt,
        savings_percent=total_savings_percent,
        co2_saved_mt=co2_saved,
        cars_removed=int(co2_saved / EQUIVALENCE_FACTORS["cars_annual_mt"]),
        flights_avoided=int(co2_saved / EQUIVALENCE_FACTORS["flights_transatlantic_mt"]),
        trees_equivalent=int(co2_saved / EQUIVALENCE_FACTORS["trees_annual_mt"]),
        homes_powered=int(co2_saved / EQUIVALENCE_FACTORS["homes_annual_mt"]),
        calculation_formula=formula,
        assumptions={
            "compression_percent": compression_percent,
            "routing_config": routing_config,
            "routable_fraction": routable,
            "energy_ratio": energy_ratio,
            "method": "Sequential application (compression first, then routing)",
        }
    )


def run_sensitivity_analysis() -> list[SensitivityResult]:
    """
    Run sensitivity analysis across key parameters.

    Varies:
    - AI energy consumption (50-100 TWh)
    - Carbon intensity (300-600 gCO2/kWh)
    - Compression savings (3-7%)
    - Routing effectiveness (conservative to optimistic)

    Returns:
        List of SensitivityResult objects
    """
    results = []

    # Base scenario for sensitivity
    base_energy = AI_ENERGY_ESTIMATES["medium"]
    base_carbon = CARBON_INTENSITY["global_average"]
    base_scenario = EnergyScenario("baseline", base_energy, base_carbon)

    # 1. Sensitivity to AI energy consumption
    low_energy = EnergyScenario("low", AI_ENERGY_ESTIMATES["low"], base_carbon)
    high_energy = EnergyScenario("high", AI_ENERGY_ESTIMATES["high"], base_carbon)

    low_result = calculate_combined_savings(low_energy)
    mid_result = calculate_combined_savings(base_scenario)
    high_result = calculate_combined_savings(high_energy)

    results.append(SensitivityResult(
        parameter="AI Energy Consumption (TWh/year)",
        low_value=AI_ENERGY_ESTIMATES["low"],
        mid_value=base_energy,
        high_value=AI_ENERGY_ESTIMATES["high"],
        low_savings_mt=low_result.co2_saved_mt,
        mid_savings_mt=mid_result.co2_saved_mt,
        high_savings_mt=high_result.co2_saved_mt,
    ))

    # 2. Sensitivity to carbon intensity
    low_carbon = EnergyScenario("low_carbon", base_energy, CARBON_INTENSITY["low"])
    high_carbon = EnergyScenario("high_carbon", base_energy, CARBON_INTENSITY["high"])

    low_c_result = calculate_combined_savings(low_carbon)
    high_c_result = calculate_combined_savings(high_carbon)

    results.append(SensitivityResult(
        parameter="Carbon Intensity (gCO2/kWh)",
        low_value=CARBON_INTENSITY["low"],
        mid_value=CARBON_INTENSITY["global_average"],
        high_value=CARBON_INTENSITY["high"],
        low_savings_mt=low_c_result.co2_saved_mt,
        mid_savings_mt=mid_result.co2_saved_mt,
        high_savings_mt=high_c_result.co2_saved_mt,
    ))

    # 3. Sensitivity to compression effectiveness
    compression_values = [3.0, COMPRESSION_SAVINGS_PERCENT, 7.0]  # Low, measured, optimistic
    compression_results = [
        calculate_combined_savings(base_scenario, compression_percent=cp)
        for cp in compression_values
    ]

    results.append(SensitivityResult(
        parameter="Compression Savings (%)",
        low_value=compression_values[0],
        mid_value=compression_values[1],
        high_value=compression_values[2],
        low_savings_mt=compression_results[0].co2_saved_mt,
        mid_savings_mt=compression_results[1].co2_saved_mt,
        high_savings_mt=compression_results[2].co2_saved_mt,
    ))

    # 4. Sensitivity to routing assumptions
    routing_results = [
        calculate_combined_savings(base_scenario, routing_config=cfg)
        for cfg in ["conservative", "moderate", "optimistic"]
    ]

    results.append(SensitivityResult(
        parameter="Routing Scenario",
        low_value=30,  # Conservative routable %
        mid_value=40,  # Moderate routable %
        high_value=50,  # Optimistic routable %
        low_savings_mt=routing_results[0].co2_saved_mt,
        mid_savings_mt=routing_results[1].co2_saved_mt,
        high_savings_mt=routing_results[2].co2_saved_mt,
    ))

    return results


def generate_scenario_table(results: list[SavingsResult]) -> str:
    """Generate a formatted table of scenario results."""
    lines = [
        "=" * 100,
        f"{'Scenario':<45} {'Savings %':>12} {'CO2 Saved (MT)':>16} {'Cars Removed':>14} {'Trees Equiv.':>14}",
        "=" * 100,
    ]

    for r in results:
        lines.append(
            f"{r.scenario_name:<45} {r.savings_percent:>11.2f}% {r.co2_saved_mt:>15,.0f} {r.cars_removed:>14,} {r.trees_equivalent:>14,}"
        )

    lines.append("=" * 100)
    return "\n".join(lines)


def run_all_calculations() -> dict:
    """
    Run all CO2 savings calculations and return comprehensive results.

    Returns:
        Dictionary with all calculation results
    """
    results = {
        "metadata": {
            "calculation_date": datetime.now().isoformat(),
            "phase2_compression_savings_percent": COMPRESSION_SAVINGS_PERCENT,
            "ai_energy_estimates_twh": AI_ENERGY_ESTIMATES,
            "carbon_intensity_g_kwh": CARBON_INTENSITY,
            "routing_assumptions": ROUTING_ASSUMPTIONS,
            "equivalence_factors": EQUIVALENCE_FACTORS,
        },
        "scenarios": {},
        "sensitivity_analysis": [],
        "summary": {},
    }

    # Create scenarios for different energy/carbon combinations
    scenarios = {
        "conservative": EnergyScenario("conservative",
                                        AI_ENERGY_ESTIMATES["low"],
                                        CARBON_INTENSITY["low"]),
        "moderate": EnergyScenario("moderate",
                                    AI_ENERGY_ESTIMATES["medium"],
                                    CARBON_INTENSITY["global_average"]),
        "aggressive": EnergyScenario("aggressive",
                                      AI_ENERGY_ESTIMATES["high"],
                                      CARBON_INTENSITY["high"]),
    }

    all_savings_results = []

    for scenario_name, scenario in scenarios.items():
        scenario_results = {
            "base_scenario": {
                "energy_twh": scenario.energy_twh,
                "carbon_intensity_g_kwh": scenario.carbon_intensity_g_kwh,
                "total_co2_mt": scenario.total_co2_mt,
            },
            "compression_only": None,
            "routing_only": {},
            "combined": {},
        }

        # Compression only
        comp_result = calculate_compression_savings(scenario)
        scenario_results["compression_only"] = asdict(comp_result)
        all_savings_results.append(comp_result)

        # Routing only (for each config)
        for routing_config in ROUTING_ASSUMPTIONS.keys():
            route_result = calculate_routing_savings(scenario, routing_config)
            scenario_results["routing_only"][routing_config] = asdict(route_result)
            all_savings_results.append(route_result)

        # Combined (for each routing config)
        for routing_config in ROUTING_ASSUMPTIONS.keys():
            combined_result = calculate_combined_savings(scenario, routing_config=routing_config)
            scenario_results["combined"][routing_config] = asdict(combined_result)
            all_savings_results.append(combined_result)

        results["scenarios"][scenario_name] = scenario_results

    # Sensitivity analysis
    sensitivity = run_sensitivity_analysis()
    results["sensitivity_analysis"] = [asdict(s) for s in sensitivity]

    # Summary statistics
    moderate_scenario = scenarios["moderate"]
    moderate_combined = calculate_combined_savings(moderate_scenario, routing_config="moderate")

    results["summary"] = {
        "headline_scenario": {
            "name": "Moderate scenario with moderate routing",
            "total_ai_co2_mt": moderate_scenario.total_co2_mt,
            "total_savings_percent": moderate_combined.savings_percent,
            "co2_saved_mt": moderate_combined.co2_saved_mt,
            "equivalent_cars_removed": moderate_combined.cars_removed,
            "equivalent_transatlantic_flights": moderate_combined.flights_avoided,
            "equivalent_trees_planted": moderate_combined.trees_equivalent,
            "equivalent_homes_powered": moderate_combined.homes_powered,
        },
        "compression_contribution": {
            "savings_percent": COMPRESSION_SAVINGS_PERCENT,
            "description": "Direct energy savings from reducing token count via prompt optimization",
        },
        "routing_contribution": {
            "moderate_savings_percent": 36.0,  # 40% * 90%
            "description": "Energy savings from directing queries to appropriately-sized models",
        },
        "range": {
            "conservative_total_savings_mt": results["scenarios"]["conservative"]["combined"]["conservative"]["co2_saved_mt"],
            "moderate_total_savings_mt": results["scenarios"]["moderate"]["combined"]["moderate"]["co2_saved_mt"],
            "aggressive_total_savings_mt": results["scenarios"]["aggressive"]["combined"]["optimistic"]["co2_saved_mt"],
        },
    }

    return results


def print_report(results: dict) -> None:
    """Print a formatted report of the CO2 savings analysis."""
    print("\n" + "=" * 80)
    print("CO2 SAVINGS ANALYSIS - AI INFERENCE OPTIMIZATION")
    print("=" * 80)

    print("\n--- INPUT PARAMETERS ---")
    print(f"Global AI Inference Energy: {AI_ENERGY_ESTIMATES['low']}-{AI_ENERGY_ESTIMATES['high']} TWh/year")
    print(f"Carbon Intensity Range: {CARBON_INTENSITY['low']}-{CARBON_INTENSITY['high']} gCO2/kWh")
    print(f"Compression Savings (Phase 2): {COMPRESSION_SAVINGS_PERCENT:.2f}%")
    print(f"Routing Scenarios: {list(ROUTING_ASSUMPTIONS.keys())}")

    print("\n--- BASELINE CO2 EMISSIONS ---")
    for name, scenario_data in results["scenarios"].items():
        base = scenario_data["base_scenario"]
        print(f"  {name.capitalize()}: {base['total_co2_mt']:,.0f} MT CO2/year "
              f"({base['energy_twh']} TWh @ {base['carbon_intensity_g_kwh']} gCO2/kWh)")

    print("\n--- SAVINGS SCENARIOS (MODERATE BASELINE) ---")
    moderate = results["scenarios"]["moderate"]

    print("\n1. COMPRESSION ONLY")
    comp = moderate["compression_only"]
    print(f"   Savings: {comp['savings_percent']:.2f}%")
    print(f"   CO2 Saved: {comp['co2_saved_mt']:,.0f} MT/year")
    print(f"   Equivalent to removing {comp['cars_removed']:,} cars from the road")

    print("\n2. ROUTING ONLY (moderate assumptions)")
    route = moderate["routing_only"]["moderate"]
    print(f"   Savings: {route['savings_percent']:.2f}%")
    print(f"   CO2 Saved: {route['co2_saved_mt']:,.0f} MT/year")
    print(f"   Equivalent to removing {route['cars_removed']:,} cars from the road")

    print("\n3. COMBINED (compression + routing)")
    combined = moderate["combined"]["moderate"]
    print(f"   Savings: {combined['savings_percent']:.2f}%")
    print(f"   CO2 Saved: {combined['co2_saved_mt']:,.0f} MT/year")
    print(f"   Equivalent to:")
    print(f"     - Removing {combined['cars_removed']:,} cars from the road")
    print(f"     - Avoiding {combined['flights_avoided']:,} transatlantic flights")
    print(f"     - Planting {combined['trees_equivalent']:,} trees")

    print("\n--- SENSITIVITY ANALYSIS ---")
    for sens in results["sensitivity_analysis"]:
        print(f"\n  {sens['parameter']}:")
        print(f"    Low ({sens['low_value']}): {sens['low_savings_mt']:,.0f} MT saved")
        print(f"    Mid ({sens['mid_value']}): {sens['mid_savings_mt']:,.0f} MT saved")
        print(f"    High ({sens['high_value']}): {sens['high_savings_mt']:,.0f} MT saved")

    print("\n--- HEADLINE SUMMARY ---")
    summary = results["summary"]["headline_scenario"]
    print(f"Under moderate assumptions, optimization could save:")
    print(f"  - {summary['total_savings_percent']:.1f}% of AI inference energy")
    print(f"  - {summary['co2_saved_mt']:,.0f} metric tons CO2 per year")
    print(f"  - Equivalent to {summary['equivalent_cars_removed']:,} cars removed from roads")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate CO2 savings from AI inference optimization"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress printed report"
    )
    args = parser.parse_args()

    # Run all calculations
    results = run_all_calculations()

    # Print report
    if not args.quiet:
        print_report(results)

    # Save to file if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
