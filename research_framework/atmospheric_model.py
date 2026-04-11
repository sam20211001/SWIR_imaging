"""
Atmospheric Disturbance Models
================================
Models the four primary atmospheric disturbances that degrade SWIR star
maps in near-earth space, following Wang et al. (2022):

  1. **Sky background** – upwelling radiance reduces detection SNR.
  2. **Atmospheric refraction** – apparent star position is shifted
     towards the zenith.
  3. **Atmospheric turbulence** – random wavefront aberrations broaden
     the star-point PSF (described by the Fried parameter r0).
  4. **Aerosol scattering** – Beer-Lambert extinction reduces apparent
     stellar flux.

All altitude values are in kilometres; wavelengths in nanometres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class AtmosphericParams:
    """Tuneable parameters for the atmospheric model."""

    altitude_km: float = 0.0
    """Observer altitude above sea level [km]."""

    zenith_angle_deg: float = 30.0
    """View zenith angle [deg] – 0 = straight up."""

    wavelength_nm: float = 1550.0
    """Representative SWIR wavelength for chromatic calculations [nm]."""

    fried_parameter_r0_cm: float = 10.0
    """Atmospheric coherence length (Fried parameter) r0 [cm].
    Typical daytime near-ground value ~5–15 cm."""

    aerosol_optical_depth: float = 0.15
    """Aerosol optical depth (AOD) at the representative wavelength.
    Clear sky ≈ 0.05, hazy ≈ 0.3."""

    sky_background_radiance: float = 50.0
    """Sky background radiance in sensor-equivalent gray-level units."""


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class AtmosphericModel:
    """
    Atmospheric disturbance model for near-earth SWIR star sensors.

    Parameters
    ----------
    params : AtmosphericParams
        Model configuration.  Can be updated via :attr:`params`.
    """

    def __init__(self, params: AtmosphericParams | None = None) -> None:
        self.params = params or AtmosphericParams()

    # ------------------------------------------------------------------
    # 1. Sky background
    # ------------------------------------------------------------------

    def sky_background_level(self) -> float:
        """
        Return the sky background gray-level contribution.

        The background scales with the air mass (a proxy for scattered
        sunlight path length) and decreases exponentially with altitude.

        Returns
        -------
        float
            Background gray-level value (same units as image pixel values).
        """
        p = self.params
        air_mass = self._air_mass(p.zenith_angle_deg)
        # Exponential attenuation with altitude (scale height ~8 km)
        altitude_factor = math.exp(-p.altitude_km / 8.0)
        return p.sky_background_radiance * air_mass * altitude_factor

    # ------------------------------------------------------------------
    # 2. Atmospheric refraction
    # ------------------------------------------------------------------

    def refraction_offset_arcsec(self) -> float:
        """
        Compute the atmospheric refraction displacement [arcsec].

        Uses the standard Bennett formula valid from 5° to 90° elevation.

        Returns
        -------
        float
            Angular shift toward the zenith [arcsec].  Zero at zenith
            (zenith_angle = 0°), increases toward the horizon.
        """
        p = self.params
        elev_deg = 90.0 - p.zenith_angle_deg
        if elev_deg < 1.0:
            elev_deg = 1.0   # formula breaks down at/below horizon

        # Bennett (1982) formula; altitude correction applied
        h = elev_deg
        R_arcmin = 1.0 / math.tan(math.radians(h + 7.31 / (h + 4.4)))
        # Altitude correction: refraction decreases as air density drops
        altitude_correction = math.exp(-p.altitude_km / 8.0)
        return R_arcmin * 60.0 * altitude_correction  # arcsec

    def refraction_offset_pixels(self, plate_scale_arcsec_per_px: float) -> float:
        """
        Convert the refraction shift to pixel units.

        Parameters
        ----------
        plate_scale_arcsec_per_px : float
            Angular size of one pixel [arcsec/px].
        """
        return self.refraction_offset_arcsec() / plate_scale_arcsec_per_px

    # ------------------------------------------------------------------
    # 3. Atmospheric turbulence
    # ------------------------------------------------------------------

    def turbulence_fwhm_arcsec(self) -> float:
        """
        Return the long-exposure seeing disk FWHM [arcsec] due to
        atmospheric turbulence, computed from the Fried parameter r0.

        FWHM_seeing ≈ 0.98 λ / r0

        Returns
        -------
        float
            Seeing FWHM [arcsec].
        """
        p = self.params
        lam_m = p.wavelength_nm * 1e-9
        r0_m = p.fried_parameter_r0_cm * 1e-2
        fwhm_rad = 0.98 * lam_m / r0_m
        return math.degrees(fwhm_rad) * 3600.0  # convert to arcsec

    def turbulence_broadening_pixels(
        self, plate_scale_arcsec_per_px: float
    ) -> float:
        """
        Turbulence broadening expressed in pixels.

        Parameters
        ----------
        plate_scale_arcsec_per_px : float
            Angular size of one pixel [arcsec/px].
        """
        return self.turbulence_fwhm_arcsec() / plate_scale_arcsec_per_px

    # ------------------------------------------------------------------
    # 4. Aerosol extinction
    # ------------------------------------------------------------------

    def aerosol_transmission(self) -> float:
        """
        Return the aerosol transmittance along the line of sight.

        T = exp(-AOD / cos(zenith_angle))

        Returns
        -------
        float
            Fractional transmission in [0, 1].
        """
        p = self.params
        cos_z = math.cos(math.radians(p.zenith_angle_deg))
        cos_z = max(cos_z, 0.01)  # avoid divide-by-zero near horizon
        return math.exp(-p.aerosol_optical_depth / cos_z)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _air_mass(zenith_angle_deg: float) -> float:
        """Plane-parallel air mass approximation."""
        cos_z = math.cos(math.radians(zenith_angle_deg))
        cos_z = max(cos_z, 0.01)
        return 1.0 / cos_z

    def summary(self) -> dict:
        """Return a dictionary summarising all disturbance contributions."""
        p = self.params
        plate_scale = 206265.0 * (p.wavelength_nm * 1e-9) / (
            10e-3  # assume 10-mm aperture for a generic scale estimate
        )
        return {
            "sky_background_level": self.sky_background_level(),
            "refraction_arcsec": self.refraction_offset_arcsec(),
            "turbulence_seeing_fwhm_arcsec": self.turbulence_fwhm_arcsec(),
            "aerosol_transmission": self.aerosol_transmission(),
            "air_mass": self._air_mass(p.zenith_angle_deg),
        }
