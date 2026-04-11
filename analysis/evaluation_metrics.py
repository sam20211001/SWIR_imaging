"""
Star Map Quality Evaluation
=============================
Quantitative assessment indicators for SWIR star maps, following the
evaluation framework in Wang et al. (2022), Section 4.

Metrics computed:
  * Image-level statistics: gray mean, noise RMS
  * Star-point statistics: 3×3 pixel mean and peak value around each
    detected star centroid
  * Centroid accuracy: offset between nominal and detected positions
  * Signal-to-Noise Ratio per star
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StarPointStats:
    """Statistics for the 3×3 pixel region around a star centroid."""

    star_id: int
    x_nominal: float      # projected/simulated centroid (column)
    y_nominal: float      # projected/simulated centroid (row)
    mean_3x3: float       # mean gray value in 3×3 window
    peak_3x3: float       # peak gray value in 3×3 window
    snr: float            # signal-to-noise ratio


@dataclass
class EvaluationReport:
    """Full quality evaluation report for one star-map image."""

    image_gray_mean: float
    image_noise_rms: float
    star_stats: List[StarPointStats]
    mean_star_peak: float
    mean_star_mean_3x3: float
    centroid_offsets_px: List[float]   # distance from nominal centroid
    mean_centroid_offset_px: float

    def print(self) -> None:
        print("=" * 60)
        print("Star Map Quality Evaluation Report")
        print("=" * 60)
        print(f"  Image gray mean       : {self.image_gray_mean:.2f}")
        print(f"  Image noise RMS       : {self.image_noise_rms:.2f}")
        print(f"  Mean star peak (3×3)  : {self.mean_star_peak:.2f}")
        print(f"  Mean star mean (3×3)  : {self.mean_star_mean_3x3:.2f}")
        print(f"  Mean centroid offset  : "
              f"{self.mean_centroid_offset_px:.4f} px")
        print()
        if self.star_stats:
            print(f"  {'ID':>4}  {'x_nom':>7}  {'y_nom':>7}  "
                  f"{'3x3 mean':>8}  {'3x3 peak':>8}  "
                  f"{'SNR':>6}  {'offset_px':>9}")
            print("  " + "-" * 64)
            for ss in self.star_stats:
                offset = (
                    self.centroid_offsets_px[self.star_stats.index(ss)]
                    if self.centroid_offsets_px else float("nan")
                )
                print(
                    f"  {ss.star_id:>4}  "
                    f"{ss.x_nominal:>7.1f}  "
                    f"{ss.y_nominal:>7.1f}  "
                    f"{ss.mean_3x3:>8.2f}  "
                    f"{ss.peak_3x3:>8.2f}  "
                    f"{ss.snr:>6.1f}  "
                    f"{offset:>9.4f}"
                )
        print("=" * 60)


class StarMapEvaluator:
    """
    Evaluates the quality of a simulated (or real) SWIR star-map image.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Star-map image in digital number (DN) units.
    background_dn : float, optional
        Expected background DN level; subtracted before SNR calculation.
        If *None*, the image median is used as the background estimate.
    """

    def __init__(
        self,
        image: np.ndarray,
        background_dn: Optional[float] = None,
    ) -> None:
        self.image = image.astype(float)
        if background_dn is None:
            self.background_dn = float(np.median(self.image))
        else:
            self.background_dn = float(background_dn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        star_centroids: List[Tuple[float, float, int]],
    ) -> EvaluationReport:
        """
        Compute full quality metrics for the given star centroids.

        Parameters
        ----------
        star_centroids : list of (x_px, y_px, star_id)
            Nominal (simulated) centroid positions and star IDs.

        Returns
        -------
        EvaluationReport
        """
        img = self.image

        # Image-level metrics
        gray_mean = float(img.mean())
        # Noise RMS estimated from the background region
        bg_region = img[img < (self.background_dn * 3)]
        noise_rms = float(bg_region.std()) if bg_region.size > 0 else float(img.std())

        # Per-star metrics
        star_stats: List[StarPointStats] = []
        offsets: List[float] = []

        for x_nom, y_nom, sid in star_centroids:
            stats, detected_xy = self._star_point_stats(x_nom, y_nom, sid)
            star_stats.append(stats)
            if detected_xy is not None:
                dx = detected_xy[0] - x_nom
                dy = detected_xy[1] - y_nom
                offsets.append(float(np.hypot(dx, dy)))
            else:
                offsets.append(float("nan"))

        valid_offsets = [o for o in offsets if not np.isnan(o)]

        return EvaluationReport(
            image_gray_mean=gray_mean,
            image_noise_rms=noise_rms,
            star_stats=star_stats,
            mean_star_peak=(
                float(np.mean([s.peak_3x3 for s in star_stats]))
                if star_stats else 0.0
            ),
            mean_star_mean_3x3=(
                float(np.mean([s.mean_3x3 for s in star_stats]))
                if star_stats else 0.0
            ),
            centroid_offsets_px=offsets,
            mean_centroid_offset_px=(
                float(np.mean(valid_offsets)) if valid_offsets else 0.0
            ),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _star_point_stats(
        self,
        x_nom: float,
        y_nom: float,
        star_id: int,
    ) -> Tuple[StarPointStats, Optional[Tuple[float, float]]]:
        """Compute 3×3 window stats and detected centroid around (x, y)."""
        img = self.image
        h, w = img.shape
        ix = int(round(x_nom))
        iy = int(round(y_nom))

        # Extract 3×3 window (clamped to image)
        x0 = max(0, ix - 1)
        x1 = min(w - 1, ix + 1)
        y0 = max(0, iy - 1)
        y1 = min(h - 1, iy + 1)
        window = img[y0:y1 + 1, x0:x1 + 1]

        mean_val = float(window.mean())
        peak_val = float(window.max())

        # Signal above background
        signal = peak_val - self.background_dn
        noise = float(np.sqrt(np.maximum(signal, 0) + self.background_dn))
        snr = signal / noise if noise > 0 else 0.0

        # Centroid (intensity-weighted centre-of-mass) within the window
        detected_xy: Optional[Tuple[float, float]] = None
        net_window = window - self.background_dn
        net_window = np.maximum(net_window, 0)
        total = net_window.sum()
        if total > 0:
            ys_local, xs_local = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            cx = float((xs_local * net_window).sum() / total)
            cy = float((ys_local * net_window).sum() / total)
            detected_xy = (cx, cy)

        return StarPointStats(
            star_id=star_id,
            x_nominal=x_nom,
            y_nominal=y_nom,
            mean_3x3=mean_val,
            peak_3x3=peak_val,
            snr=snr,
        ), detected_xy
