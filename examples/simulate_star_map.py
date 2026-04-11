"""
End-to-End SWIR Star Map Simulation Example
=============================================
Demonstrates the complete research pipeline recommended in the problem
statement:

  Phase 1 – MODEL STUDY
    Build all physical models (optics, atmosphere, detector).

  Phase 2 – SIMULATION & ANALYSIS
    Generate synthetic star maps under controlled conditions and evaluate
    their quality using quantitative metrics.

  Phase 3 – EXPERIMENT (not shown here – requires real hardware data)
    Compare simulation predictions against measured data.

Run from the repository root:
  python examples/simulate_star_map.py
"""

import sys
import os

# Allow running from the repository root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from research_framework import StarMapSimulator
from research_framework.star_map_simulator import SimulationConfig
from research_framework.atmospheric_model import AtmosphericParams
from research_framework.energy_transfer import OpticsParams
from analysis import StarMapEvaluator


def run_scenario(name: str, config: SimulationConfig) -> None:
    """Simulate a star map for the given configuration and print results."""
    print(f"\n{'#' * 60}")
    print(f"  Scenario: {name}")
    print(f"{'#' * 60}")

    simulator = StarMapSimulator()
    result = simulator.simulate(config)
    simulator.print_summary(result)

    # Evaluate image quality
    centroids = [
        (sr.x_px, sr.y_px, sr.star.star_id)
        for sr in result.stars
        if sr.in_frame
    ]
    evaluator = StarMapEvaluator(result.image)
    report = evaluator.evaluate(centroids)
    report.print()


def main() -> None:
    # -------------------------------------------------------------------
    # Phase 1 – Model Study: define optical system and detector parameters
    # -------------------------------------------------------------------
    optics = OpticsParams(
        aperture_diameter_mm=50.0,
        focal_length_mm=150.0,
        optical_transmittance=0.80,
        quantum_efficiency=0.65,
        bandpass_nm=1500.0,
        center_wavelength_nm=1550.0,
        integration_time_ms=100.0,
        pixel_pitch_um=15.0,
        read_noise_e=60.0,
        dark_current_e_per_s=800.0,
        full_well_capacity_e=80_000.0,
    )

    # -------------------------------------------------------------------
    # Phase 2 – Simulation & Analysis: vary atmospheric conditions
    # -------------------------------------------------------------------

    # Scenario A: clear sky, small zenith angle (best case)
    config_a = SimulationConfig(
        boresight_ra_deg=114.825,
        boresight_dec_deg=5.227,
        half_fov_deg=10.0,
        optics=optics,
        atmosphere=AtmosphericParams(
            altitude_km=0.0,
            zenith_angle_deg=20.0,
            wavelength_nm=1550.0,
            fried_parameter_r0_cm=15.0,
            aerosol_optical_depth=0.05,
            sky_background_radiance=30.0,
        ),
        image_width=512,
        image_height=512,
        random_seed=42,
    )
    run_scenario("Clear sky, zenith angle=20°", config_a)

    # Scenario B: hazy sky, large zenith angle (degraded performance)
    config_b = SimulationConfig(
        boresight_ra_deg=114.825,
        boresight_dec_deg=5.227,
        half_fov_deg=10.0,
        optics=optics,
        atmosphere=AtmosphericParams(
            altitude_km=0.0,
            zenith_angle_deg=60.0,
            wavelength_nm=1550.0,
            fried_parameter_r0_cm=5.0,
            aerosol_optical_depth=0.30,
            sky_background_radiance=100.0,
        ),
        image_width=512,
        image_height=512,
        random_seed=42,
    )
    run_scenario("Hazy sky, zenith angle=60°", config_b)

    # Scenario C: elevated altitude (e.g. airborne platform at 5 km)
    config_c = SimulationConfig(
        boresight_ra_deg=114.825,
        boresight_dec_deg=5.227,
        half_fov_deg=10.0,
        optics=optics,
        atmosphere=AtmosphericParams(
            altitude_km=5.0,
            zenith_angle_deg=30.0,
            wavelength_nm=1550.0,
            fried_parameter_r0_cm=12.0,
            aerosol_optical_depth=0.08,
            sky_background_radiance=15.0,
        ),
        image_width=512,
        image_height=512,
        random_seed=42,
    )
    run_scenario("Airborne platform, altitude=5 km", config_c)


if __name__ == "__main__":
    main()
