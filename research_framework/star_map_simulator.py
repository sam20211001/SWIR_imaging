"""
SWIR Star Map Simulator
========================
Orchestrates the end-to-end star map simulation pipeline:

  Step 1 – Star catalog query
           Select navigation stars within the sensor's field of view.

  Step 2 – Coordinate conversion
           Transform ICRS star vectors to detector pixel coordinates,
           applying atmospheric refraction shift.

  Step 3 – Energy transfer
           Compute stellar signal electrons, background level, and SNR.

  Step 4 – Image formation
           Render the star map on the detector, including PSF, noise,
           and quantisation.

This modular design mirrors the correct PhD methodology described in the
project problem statement: Model Study → Simulation/Analysis → Experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .atmospheric_model import AtmosphericModel, AtmosphericParams
from .coordinate_conversion import CoordinateConverter
from .energy_transfer import EnergyTransfer, OpticsParams
from .image_formation import ImageFormation
from .star_catalog import StarCatalog, StarRecord


@dataclass
class SimulationConfig:
    """Complete configuration for one simulation run."""

    # Boresight pointing
    boresight_ra_deg: float = 114.825
    boresight_dec_deg: float = 5.227
    boresight_roll_deg: float = 0.0

    # Field of view (half-angle)
    half_fov_deg: float = 8.0

    # Optics / detector
    optics: OpticsParams = field(default_factory=OpticsParams)

    # Atmosphere
    atmosphere: AtmosphericParams = field(default_factory=AtmosphericParams)

    # Image
    image_width: int = 1024
    image_height: int = 1024
    adc_bits: int = 12

    # Reproducibility
    random_seed: Optional[int] = 42


@dataclass
class StarResult:
    """Simulation result for a single star."""

    star: StarRecord
    x_px: float          # image column (sub-pixel)
    y_px: float          # image row   (sub-pixel)
    signal_e: float      # collected electrons
    snr: float           # SNR
    in_frame: bool       # within image bounds?


@dataclass
class SimulationResult:
    """Full result of one simulation run."""

    image: np.ndarray                  # rendered star-map image (uint16)
    stars: List[StarResult]            # per-star results
    config: SimulationConfig           # configuration used
    atmospheric_summary: Dict          # atmospheric disturbance values
    psf_fwhm_px: float                 # combined PSF FWHM [px]
    background_e_per_px: float         # sky background [e/px/frame]


class StarMapSimulator:
    """
    End-to-end SWIR star map simulator.

    Parameters
    ----------
    catalog : StarCatalog, optional
        Navigation star database.  Uses the built-in catalog if omitted.
    """

    def __init__(self, catalog: Optional[StarCatalog] = None) -> None:
        self.catalog = catalog or StarCatalog()

    # ------------------------------------------------------------------
    # Main simulation entry point
    # ------------------------------------------------------------------

    def simulate(self, config: SimulationConfig) -> SimulationResult:
        """
        Run a full simulation and return the star-map image and metadata.

        Parameters
        ----------
        config : SimulationConfig
            Complete simulation configuration.

        Returns
        -------
        SimulationResult
        """
        op = config.optics
        atm_p = config.atmosphere

        # --- Build sub-models ---
        atm_model = AtmosphericModel(atm_p)
        converter = CoordinateConverter.from_boresight_ra_dec(
            ra_deg=config.boresight_ra_deg,
            dec_deg=config.boresight_dec_deg,
            roll_deg=config.boresight_roll_deg,
            focal_length_mm=op.focal_length_mm,
            pixel_pitch_um=op.pixel_pitch_um,
            image_size=(config.image_width, config.image_height),
        )
        energy = EnergyTransfer(op, atm_model)
        image_size = (config.image_width, config.image_height)
        renderer = ImageFormation(
            image_size=image_size,
            adc_bits=config.adc_bits,
            full_well_capacity_e=op.full_well_capacity_e,
            read_noise_e=op.read_noise_e,
            dark_current_e_per_s=op.dark_current_e_per_s,
            integration_time_ms=op.integration_time_ms,
            seed=config.random_seed,
        )

        # --- Plate scale [arcsec/px] ---
        plate_scale_arcsec = (
            np.degrees(np.arctan(op.pixel_pitch_um * 1e-3 / op.focal_length_mm))
            * 3600.0
        )

        # --- PSF FWHM ---
        diff_fwhm_px = ImageFormation.diffraction_fwhm_pixels(
            wavelength_nm=atm_p.wavelength_nm,
            aperture_diameter_mm=op.aperture_diameter_mm,
            focal_length_mm=op.focal_length_mm,
            pixel_pitch_um=op.pixel_pitch_um,
        )
        turb_fwhm_px = atm_model.turbulence_broadening_pixels(plate_scale_arcsec)
        psf_fwhm_px = ImageFormation.combined_psf_fwhm(diff_fwhm_px, turb_fwhm_px)

        # --- Refraction shift [px] along zenith direction ---
        refraction_px = atm_model.refraction_offset_pixels(plate_scale_arcsec)

        # --- Boresight unit vector (for FoV query) ---
        boresight_vec = CoordinateConverter.from_boresight_ra_dec(
            ra_deg=config.boresight_ra_deg,
            dec_deg=config.boresight_dec_deg,
            roll_deg=0.0,
            focal_length_mm=op.focal_length_mm,
            pixel_pitch_um=op.pixel_pitch_um,
            image_size=image_size,
        ).A[2]   # third row of A = optical-Z axis expressed in ICRS

        half_fov_rad = np.deg2rad(config.half_fov_deg)
        visible_stars = self.catalog.stars_in_fov(boresight_vec, half_fov_rad)

        # --- Project and compute per-star quantities ---
        star_results: List[StarResult] = []
        positions: List[Tuple[float, float]] = []
        signals: List[float] = []

        for star in visible_stars:
            xy = converter.icrs_to_image(star.unit_vector())
            if xy is None:
                continue
            x_px, y_px = xy

            # Shift upward (decrease y) because zenith is at the top of the image
            # and atmospheric refraction displaces stars toward the zenith.
            y_px_refracted = y_px - refraction_px

            sig_e = energy.star_signal_electrons(star.mag_swir)
            snr = energy.snr(star.mag_swir)
            in_frame = converter.is_in_frame(x_px, y_px_refracted)

            star_results.append(StarResult(
                star=star,
                x_px=x_px,
                y_px=y_px_refracted,
                signal_e=sig_e,
                snr=snr,
                in_frame=in_frame,
            ))

            if in_frame:
                positions.append((x_px, y_px_refracted))
                signals.append(sig_e)

        # --- Background ---
        bg_e = energy.background_electrons_per_pixel()

        # --- Render image ---
        image = renderer.render(
            star_positions=positions,
            star_signals_e=signals,
            psf_fwhm_pixels=psf_fwhm_px,
            background_e_per_px=bg_e,
        )

        return SimulationResult(
            image=image,
            stars=star_results,
            config=config,
            atmospheric_summary=atm_model.summary(),
            psf_fwhm_px=psf_fwhm_px,
            background_e_per_px=bg_e,
        )

    def print_summary(self, result: SimulationResult) -> None:
        """Print a human-readable summary of a simulation result."""
        cfg = result.config
        print("=" * 60)
        print("SWIR Star Map Simulation Summary")
        print("=" * 60)
        print(f"  Boresight:     RA={cfg.boresight_ra_deg:.2f}°  "
              f"Dec={cfg.boresight_dec_deg:.2f}°  "
              f"Roll={cfg.boresight_roll_deg:.1f}°")
        print(f"  Altitude:      {cfg.atmosphere.altitude_km:.1f} km")
        print(f"  Zenith angle:  {cfg.atmosphere.zenith_angle_deg:.1f}°")
        print(f"  Image size:    {cfg.image_width}×{cfg.image_height} px")
        print(f"  PSF FWHM:      {result.psf_fwhm_px:.2f} px")
        print(f"  Background:    {result.background_e_per_px:.1f} e/px")
        print()
        atm = result.atmospheric_summary
        print("  Atmospheric disturbances:")
        print(f"    Sky background level : {atm['sky_background_level']:.2f}")
        print(f"    Refraction           : {atm['refraction_arcsec']:.2f} arcsec")
        print(f"    Seeing FWHM          : {atm['turbulence_seeing_fwhm_arcsec']:.2f} arcsec")
        print(f"    Aerosol transmission : {atm['aerosol_transmission']:.4f}")
        print(f"    Air mass             : {atm['air_mass']:.3f}")
        print()
        in_frame = [s for s in result.stars if s.in_frame]
        print(f"  Stars in FoV:   {len(result.stars)}")
        print(f"  Stars in frame: {len(in_frame)}")
        if in_frame:
            print(f"  {'ID':>4}  {'RA':>8}  {'Dec':>8}  "
                  f"{'Mag_SWIR':>8}  {'x_px':>7}  {'y_px':>7}  "
                  f"{'Signal_e':>10}  {'SNR':>6}")
            print("  " + "-" * 74)
            for sr in sorted(in_frame, key=lambda s: s.star.mag_swir):
                print(
                    f"  {sr.star.star_id:>4}  "
                    f"{sr.star.ra_deg:>8.3f}  "
                    f"{sr.star.dec_deg:>8.3f}  "
                    f"{sr.star.mag_swir:>8.2f}  "
                    f"{sr.x_px:>7.1f}  "
                    f"{sr.y_px:>7.1f}  "
                    f"{sr.signal_e:>10.0f}  "
                    f"{sr.snr:>6.1f}"
                )
        print("=" * 60)
