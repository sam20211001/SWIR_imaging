"""
Image Formation and Degradation
=================================
Renders star-point images onto the detector array, modelling:

  1. **Optical diffraction** – Airy-disk limited PSF (FWHM from aperture).
  2. **Atmospheric turbulence broadening** – adds seeing contribution.
  3. **Detector response** – shot noise, read noise, dark current,
     quantisation, and saturation.
  4. **Background injection** – uniform sky background level per frame.

The module follows the image-quality degradation section (Section 2.3)
of Wang et al. (2022).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from numpy.random import default_rng


class ImageFormation:
    """
    Renders stars onto a simulated SWIR detector image.

    Parameters
    ----------
    image_size : tuple (width, height)
        Image dimensions in pixels.
    adc_bits : int
        Analogue-to-digital converter bit depth (e.g. 12 or 14).
    full_well_capacity_e : float
        Pixel full-well capacity in electrons.
    read_noise_e : float
        Read noise standard deviation [electrons].
    dark_current_e_per_s : float
        Dark current per pixel per second [e/s].
    integration_time_ms : float
        Detector integration time [ms].
    seed : int, optional
        Random seed for reproducible simulations.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        adc_bits: int = 12,
        full_well_capacity_e: float = 100_000.0,
        read_noise_e: float = 50.0,
        dark_current_e_per_s: float = 1000.0,
        integration_time_ms: float = 100.0,
        seed: Optional[int] = None,
    ) -> None:
        self.image_size = image_size
        self.adc_bits = adc_bits
        self.full_well_capacity_e = full_well_capacity_e
        self.read_noise_e = read_noise_e
        self.dark_current_e_per_s = dark_current_e_per_s
        self.integration_time_ms = integration_time_ms
        self.rng = default_rng(seed)

    # ------------------------------------------------------------------
    # Main rendering entry point
    # ------------------------------------------------------------------

    def render(
        self,
        star_positions: List[Tuple[float, float]],
        star_signals_e: List[float],
        psf_fwhm_pixels: float,
        background_e_per_px: float = 0.0,
    ) -> np.ndarray:
        """
        Render a simulated star map image.

        Parameters
        ----------
        star_positions : list of (x_px, y_px)
            Sub-pixel positions of each star (column, row).
        star_signals_e : list of float
            Total signal electrons for each star (before PSF spreading).
        psf_fwhm_pixels : float
            Combined PSF FWHM (optical + turbulence) in pixels.
        background_e_per_px : float
            Mean sky background electrons per pixel per frame.

        Returns
        -------
        ndarray, shape (height, width), dtype uint16
            Quantised 16-bit (but ADC-limited) star-map image.
        """
        w, h = self.image_size
        t_int_s = self.integration_time_ms * 1e-3

        # --- Electron accumulation image (floating point) ---
        electron_image = np.zeros((h, w), dtype=np.float64)

        # Add sky background (Poisson)
        if background_e_per_px > 0:
            bg = self.rng.poisson(background_e_per_px, size=(h, w)).astype(float)
            electron_image += bg

        # Add dark current (Poisson)
        dark_e = self.dark_current_e_per_s * t_int_s
        if dark_e > 0:
            dark = self.rng.poisson(dark_e, size=(h, w)).astype(float)
            electron_image += dark

        # Stamp each star's PSF
        sigma = psf_fwhm_pixels / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        for (x_px, y_px), signal_e in zip(star_positions, star_signals_e):
            self._stamp_gaussian(electron_image, x_px, y_px, signal_e, sigma)

        # Add shot noise
        # Approximation: Gaussian noise with σ = √signal, valid when signal >> 1 e⁻.
        # For low signals (< ~10 electrons), true Poisson statistics will differ.
        shot_noise = self.rng.normal(0, np.sqrt(np.maximum(electron_image, 0)))
        electron_image += shot_noise

        # Add read noise
        read_noise = self.rng.normal(0, self.read_noise_e, size=(h, w))
        electron_image += read_noise

        # Saturate at full-well capacity
        electron_image = np.clip(electron_image, 0, self.full_well_capacity_e)

        # ADC quantisation
        dn_image = self._quantise(electron_image)
        return dn_image

    # ------------------------------------------------------------------
    # PSF helpers
    # ------------------------------------------------------------------

    @staticmethod
    def diffraction_fwhm_pixels(
        wavelength_nm: float,
        aperture_diameter_mm: float,
        focal_length_mm: float,
        pixel_pitch_um: float,
    ) -> float:
        """
        Diffraction-limited Airy disk FWHM in pixels.

        FWHM_diff ≈ 1.028 λ f / D  (in focal-plane length)

        Parameters
        ----------
        wavelength_nm : float
            Wavelength [nm].
        aperture_diameter_mm : float
            Aperture diameter [mm].
        focal_length_mm : float
            Focal length [mm].
        pixel_pitch_um : float
            Pixel pitch [μm].
        """
        lam_mm = wavelength_nm * 1e-6
        fwhm_mm = 1.028 * lam_mm * focal_length_mm / aperture_diameter_mm
        return fwhm_mm * 1e3 / pixel_pitch_um   # mm → μm → pixels

    @staticmethod
    def combined_psf_fwhm(
        diffraction_fwhm_px: float,
        turbulence_fwhm_px: float,
    ) -> float:
        """
        Total PSF FWHM from quadratic addition of independent broadening
        components.

        FWHM_total = sqrt(FWHM_diff^2 + FWHM_turb^2)
        """
        return math.sqrt(diffraction_fwhm_px ** 2 + turbulence_fwhm_px ** 2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stamp_gaussian(
        self,
        image: np.ndarray,
        x_c: float,
        y_c: float,
        total_signal_e: float,
        sigma_px: float,
    ) -> None:
        """Stamp a 2-D Gaussian PSF centred at (x_c, y_c) onto *image*."""
        if total_signal_e <= 0:
            return
        h, w = image.shape
        # Bounding box: ±4σ around the centre
        r = int(math.ceil(4.0 * sigma_px)) + 1
        xi_min = max(0, int(x_c) - r)
        xi_max = min(w - 1, int(x_c) + r)
        yi_min = max(0, int(y_c) - r)
        yi_max = min(h - 1, int(y_c) + r)

        xi = np.arange(xi_min, xi_max + 1, dtype=float)
        yi = np.arange(yi_min, yi_max + 1, dtype=float)
        xx, yy = np.meshgrid(xi, yi)

        gauss = np.exp(
            -((xx - x_c) ** 2 + (yy - y_c) ** 2) / (2.0 * sigma_px ** 2)
        )
        gauss_sum = gauss.sum()
        if gauss_sum > 0:
            gauss *= total_signal_e / gauss_sum

        image[yi_min:yi_max + 1, xi_min:xi_max + 1] += gauss

    def _quantise(self, electron_image: np.ndarray) -> np.ndarray:
        """
        Convert electron image to digital numbers (DN) using the ADC model.

        The full-well charge maps to 2^adc_bits - 1.
        """
        max_dn = (1 << self.adc_bits) - 1
        dn = electron_image * (max_dn / self.full_well_capacity_e)
        dn = np.clip(np.round(dn), 0, max_dn).astype(np.uint16)
        return dn
