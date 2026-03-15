"""
Energy Transfer
================
Calculates the stellar photon flux arriving at the detector, accounting
for:
  * Stellar intrinsic luminosity (from SWIR magnitude)
  * Optical system throughput (aperture, transmittance, quantum efficiency)
  * Atmospheric aerosol extinction (via AtmosphericModel)
  * Integration time

This module implements the energy-transfer portion of the simulation
pipeline described in Wang et al. (2022), Section 2.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .atmospheric_model import AtmosphericModel


# Zero-magnitude SWIR (1550 nm) flux density reference
# F0 ≈ 3631 Jy (AB system); 1 Jy = 1e-26 W m^-2 Hz^-1
_F0_AB_JY = 3631.0          # Jansky
_JY_TO_SI = 1.0e-26         # W m^-2 Hz^-1 per Jy
_PLANCK_H = 6.626e-34       # J·s
_SPEED_C = 3.0e8            # m/s


@dataclass
class OpticsParams:
    """Optical system and detector parameters."""

    aperture_diameter_mm: float = 50.0
    """Entrance pupil diameter [mm]."""

    focal_length_mm: float = 100.0
    """Effective focal length [mm]."""

    optical_transmittance: float = 0.85
    """Total optical transmittance (lenses + filter losses)."""

    quantum_efficiency: float = 0.70
    """Detector quantum efficiency at the representative wavelength."""

    bandpass_nm: float = 500.0
    """Spectral bandpass of the SWIR filter [nm] (e.g. 1000–2500 nm = 1500 nm)."""

    center_wavelength_nm: float = 1550.0
    """Central wavelength of the bandpass [nm]."""

    integration_time_ms: float = 100.0
    """Detector integration (exposure) time [ms]."""

    pixel_pitch_um: float = 15.0
    """Detector pixel pitch [μm]."""

    read_noise_e: float = 50.0
    """Detector read noise [electrons RMS]."""

    dark_current_e_per_s: float = 1000.0
    """Dark current [electrons per second per pixel]."""

    full_well_capacity_e: float = 100000.0
    """Pixel full-well capacity [electrons]."""


class EnergyTransfer:
    """
    Computes stellar signal electrons on the detector.

    Parameters
    ----------
    optics : OpticsParams
        Optical system and detector specification.
    atm_model : AtmosphericModel, optional
        Atmospheric model used to account for extinction.  If *None*,
        aerosol extinction is ignored (space-like conditions).
    """

    def __init__(
        self,
        optics: OpticsParams,
        atm_model: Optional[AtmosphericModel] = None,
    ) -> None:
        self.optics = optics
        self.atm_model = atm_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def star_signal_electrons(self, mag_swir: float) -> float:
        """
        Compute the total signal electrons integrated over one frame for
        a star of given SWIR magnitude.

        Parameters
        ----------
        mag_swir : float
            SWIR magnitude of the star.

        Returns
        -------
        float
            Electrons collected per pixel (assuming all star-point energy
            falls in one pixel – the per-pixel value after PSF spreading
            is computed in :mod:`image_formation`).
        """
        op = self.optics
        t_int = op.integration_time_ms * 1e-3       # seconds

        # Spectral flux density [W m^-2 Hz^-1]
        flux_jy = _F0_AB_JY * 10.0 ** (-0.4 * mag_swir)
        flux_si = flux_jy * _JY_TO_SI

        # Convert to photon flux [photons m^-2 s^-1] over the bandpass
        freq_hz = _SPEED_C / (op.center_wavelength_nm * 1e-9)
        # Bandwidth in frequency: Δν ≈ c/λ² · Δλ
        bandwidth_m = op.bandpass_nm * 1e-9
        delta_nu = _SPEED_C / (op.center_wavelength_nm * 1e-9) ** 2 * bandwidth_m
        energy_per_photon = _PLANCK_H * freq_hz
        photon_flux = flux_si * delta_nu / energy_per_photon  # ph m^-2 s^-1

        # Effective collecting area [m^2]
        area_m2 = math.pi * (op.aperture_diameter_mm * 0.5e-3) ** 2

        # Atmospheric transmission
        atm_T = self.atm_model.aerosol_transmission() if self.atm_model else 1.0

        # Total electrons before QE (signal photons)
        total_electrons = (
            photon_flux
            * area_m2
            * op.optical_transmittance
            * atm_T
            * op.quantum_efficiency
            * t_int
        )
        return total_electrons

    def background_electrons_per_pixel(self) -> float:
        """
        Electrons per pixel contributed by sky background per frame.

        Uses the background radiance level from the atmospheric model.
        If no atmospheric model is provided, returns 0.
        """
        if self.atm_model is None:
            return 0.0
        op = self.optics
        t_int = op.integration_time_ms * 1e-3
        bg_level = self.atm_model.sky_background_level()
        # bg_level is already in gray-level equivalent units; convert using
        # a representative sky photon flux density (empirical scaling)
        bg_photon_flux_per_px = bg_level * 10.0  # ph px^-1 s^-1 per gray-unit
        return bg_photon_flux_per_px * op.quantum_efficiency * t_int

    def snr(self, mag_swir: float) -> float:
        """
        Signal-to-Noise Ratio for a star of given SWIR magnitude.

        SNR = S / sqrt(S + N_bg + N_dark + N_read^2)

        Parameters
        ----------
        mag_swir : float
            SWIR magnitude of the star.

        Returns
        -------
        float
            SNR value; returns 0 if signal is zero.
        """
        op = self.optics
        t_int = op.integration_time_ms * 1e-3
        S = self.star_signal_electrons(mag_swir)
        N_bg = self.background_electrons_per_pixel()
        N_dark = op.dark_current_e_per_s * t_int
        N_read = op.read_noise_e
        noise_total = math.sqrt(S + N_bg + N_dark + N_read ** 2)
        return S / noise_total if noise_total > 0 else 0.0

    def limiting_magnitude(self, min_snr: float = 7.0) -> float:
        """
        Estimate the faint limiting SWIR magnitude that achieves *min_snr*.

        Uses binary search between mag = -2 and mag = 10.

        Parameters
        ----------
        min_snr : float
            Minimum required SNR for detection (default 7).

        Returns
        -------
        float
            Limiting SWIR magnitude.
        """
        lo, hi = -2.0, 10.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if self.snr(mid) >= min_snr:
                lo = mid
            else:
                hi = mid
        return lo
