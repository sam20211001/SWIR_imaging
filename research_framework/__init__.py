"""
SWIR Star Sensor Research Framework
=====================================
A simulation framework for near-earth space Short-Wave Infrared (SWIR)
star sensors, based on the methodology described in:

  Wang et al., "Near-earth space star map simulation method of short-wave
  infrared star sensor", Infrared Physics & Technology 127 (2022) 104436.

Research Methodology (correct PhD approach):
  1. Model Study   – Build mathematical models for each physical process
  2. Simulation    – Simulate star maps under controlled conditions
  3. Analysis      – Evaluate and quantify performance metrics
  4. Experiment    – Validate against measured data (final step)

Modules:
  star_catalog          – Navigation star catalog management
  coordinate_conversion – Celestial-to-image coordinate transforms
  atmospheric_model     – Atmospheric background, refraction,
                          turbulence, and aerosol models
  energy_transfer       – Stellar flux through optical system
  image_formation       – Detector response and image degradation
  star_map_simulator    – End-to-end simulation pipeline
"""

from .star_catalog import StarCatalog
from .coordinate_conversion import CoordinateConverter
from .atmospheric_model import AtmosphericModel
from .energy_transfer import EnergyTransfer
from .image_formation import ImageFormation
from .star_map_simulator import StarMapSimulator

__all__ = [
    "StarCatalog",
    "CoordinateConverter",
    "AtmosphericModel",
    "EnergyTransfer",
    "ImageFormation",
    "StarMapSimulator",
]
