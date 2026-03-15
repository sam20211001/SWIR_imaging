# SWIR Imaging — Near-Earth Space Star Sensor Research Framework

> **Reference paper**: Wang et al., *"Near-earth space star map simulation method of
> short-wave infrared star sensor"*, Infrared Physics & Technology **127** (2022) 104436.
> ([`wangbingwen_swir_imaging.pdf`](wangbingwen_swir_imaging.pdf))

---

## Research Methodology

The correct PhD research methodology for star sensor key technology studies is:

```
Phase 1 – MODEL STUDY      →   Phase 2 – SIMULATION & ANALYSIS   →   Phase 3 – EXPERIMENT
  Build physical models          Generate synthetic star maps            Validate against
  for every process:             under controlled conditions             measured data;
  • Optics / detector            and quantify performance:              compare simulation
  • Atmosphere                   • Star-map quality metrics             predictions with
  • Coordinate geometry          • SNR vs. magnitude curves             actual observations
  • Energy transfer              • Atmospheric sensitivity study
```

**Why this order matters:**
- Jumping directly to hardware experiments without first understanding the models leads to
  uninterpretable results, because the atmosphere is complex, changeable, and highly coupled.
- Star maps acquired by an outfield observer are time-consuming and costly, and it is not
  possible to isolate the effect of a single atmospheric disturbance factor from a real image.
- Simulation provides a controlled, reproducible environment in which each factor (background,
  refraction, turbulence, aerosol) can be varied independently.

---

## Repository Structure

```
SWIR_imaging/
├── wangbingwen_swir_imaging.pdf   # Reference paper (Wang et al., 2022)
├── research_framework/            # Phase 1 – physical models
│   ├── __init__.py
│   ├── star_catalog.py            # Navigation star database
│   ├── coordinate_conversion.py  # ICRS → image-plane projection
│   ├── atmospheric_model.py      # Background, refraction, turbulence, aerosol
│   ├── energy_transfer.py        # Stellar flux, SNR, limiting magnitude
│   ├── image_formation.py        # PSF, detector noise, ADC quantisation
│   └── star_map_simulator.py     # End-to-end simulation pipeline
├── analysis/                      # Phase 2 – evaluation tools
│   ├── __init__.py
│   └── evaluation_metrics.py     # Star-map quality metrics
├── examples/
│   └── simulate_star_map.py      # Worked example (all three scenarios)
└── tests/
    └── test_framework.py         # Unit & integration tests
```

---

## Quick Start

### Prerequisites

```bash
pip install numpy pytest
```

### Run the example

```bash
python examples/simulate_star_map.py
```

The example simulates three observing scenarios and prints a quality report for each:

| Scenario | Altitude | Zenith angle | Aerosol OD |
|----------|----------|-------------|------------|
| A – clear sky       | 0 km | 20° | 0.05 |
| B – hazy sky        | 0 km | 60° | 0.30 |
| C – airborne (5 km) | 5 km | 30° | 0.08 |

### Run the tests

```bash
python -m pytest tests/ -v
```

---

## Module Overview

### Phase 1 – Model Study

#### `research_framework/star_catalog.py`
Manages the navigation star database.  Each `StarRecord` stores RA, Dec, visual
magnitude, and SWIR magnitude.  The built-in catalog contains representative bright
stars; for production use it can be replaced with the full BSC5 or Hipparcos catalog.

```python
from research_framework import StarCatalog
cat = StarCatalog(mag_limit_swir=4.0)
fov_stars = cat.stars_in_fov(boresight_vec, half_fov_rad=0.15)
```

#### `research_framework/coordinate_conversion.py`
Transforms ICRS celestial coordinates to detector pixel positions via:

```
ICRS direction vector  →  Attitude matrix A  →  Optical frame  →  Pin-hole projection  →  Pixel (x, y)
```

```python
from research_framework import CoordinateConverter
conv = CoordinateConverter.from_boresight_ra_dec(
    ra_deg=114.8, dec_deg=5.2, roll_deg=0.0,
    focal_length_mm=150, pixel_pitch_um=15,
    image_size=(1024, 1024),
)
xy = conv.icrs_to_image(star.unit_vector())
```

#### `research_framework/atmospheric_model.py`
Models the four atmospheric disturbance factors (Wang et al., 2022, §2):

| Factor | Model |
|--------|-------|
| Sky background | Exponential altitude scaling × air-mass |
| Atmospheric refraction | Bennett (1982) formula + altitude correction |
| Atmospheric turbulence | Fried parameter r₀ → seeing FWHM |
| Aerosol scattering | Beer-Lambert extinction |

```python
from research_framework.atmospheric_model import AtmosphericModel, AtmosphericParams
atm = AtmosphericModel(AtmosphericParams(altitude_km=0, zenith_angle_deg=45))
print(atm.summary())
```

#### `research_framework/energy_transfer.py`
Computes stellar signal electrons, background electrons, and SNR.

```python
from research_framework.energy_transfer import EnergyTransfer, OpticsParams
et = EnergyTransfer(OpticsParams(aperture_diameter_mm=50, focal_length_mm=150), atm)
print(f"SNR for 2nd-mag star: {et.snr(2.0):.1f}")
print(f"Limiting magnitude:   {et.limiting_magnitude():.2f}")
```

#### `research_framework/image_formation.py`
Renders star-point PSFs (Gaussian approximation) onto the detector array and
applies shot noise, dark current, read noise, saturation, and ADC quantisation.

#### `research_framework/star_map_simulator.py`
Orchestrates all modules into a single `simulate()` call that returns a
`SimulationResult` containing the rendered image and per-star metadata.

```python
from research_framework import StarMapSimulator
from research_framework.star_map_simulator import SimulationConfig

sim = StarMapSimulator()
result = sim.simulate(SimulationConfig())
sim.print_summary(result)
```

### Phase 2 – Simulation & Analysis

#### `analysis/evaluation_metrics.py`
Provides the assessment indicators listed in Wang et al. (2022), §4:

- Image gray mean and noise RMS
- 3×3 pixel window mean and peak around each star centroid
- Star-point centroid offset (detected vs. nominal position)
- Per-star SNR

```python
from analysis import StarMapEvaluator
ev = StarMapEvaluator(result.image)
report = ev.evaluate(centroids)
report.print()
```

---

## Key Results from the Reference Paper

| Observation | Prediction |
|-------------|------------|
| Stronger sky background | Larger gray mean, larger noise RMS, larger 3×3 mean and peak |
| Larger view zenith angle | More significant star-point centroid offset (refraction) |

These predictions are reproduced by the simulation framework and can be verified
by running the example script.

---

## Specific Steps for Star Sensor Key Technology Research

Following the methodology of Wang et al. (2022), the recommended research workflow is:

1. **Build the navigation star database** – Select suitable bright stars within the
   sensor's spectral response; assign SWIR magnitudes from spectral-type conversion.

2. **Model the coordinate transformation pipeline** – From ICRS through attitude
   matrix to focal-plane pixel coordinates (pin-hole model).

3. **Model each atmospheric disturbance factor independently** – Quantify background,
   refraction, turbulence (r₀), and aerosol extinction separately.

4. **Compute energy transfer** – Calculate stellar flux through the optical system and
   detector to obtain predicted SNR for each star under each condition.

5. **Generate synthetic star maps** – Combine all models to render realistic images for
   specific observing scenarios (altitude, zenith angle, sky condition).

6. **Evaluate and analyse** – Apply quantitative metrics (gray mean, noise RMS, 3×3
   window statistics, centroid accuracy) to understand how each factor degrades
   star-map quality.

7. **Design and optimise** – Use the sensitivity results to optimise the optical system,
   integration time, and detection threshold.

8. **Validate against measured data** – Only at this final stage acquire real star maps
   and compare the simulation predictions against the measurements.

---

## References

- Wang, H., Wang, B., Gao, Y., & Wu, S. (2022). *Near-earth space star map simulation
  method of short-wave infrared star sensor*. Infrared Physics & Technology, 127, 104436.
  https://doi.org/10.1016/j.infrared.2022.104436
