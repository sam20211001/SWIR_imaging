"""
Navigation Star Catalog
========================
Manages the navigation star database used by the SWIR star sensor.

Each record stores:
  - Right Ascension (RA) in degrees [J2000]
  - Declination (Dec) in degrees [J2000]
  - Visual magnitude (Mv)
  - SWIR magnitude (M_swir) derived from spectral type

The built-in catalog provides a small representative set of bright stars
suitable for algorithm development and unit tests.  For production use,
replace or extend it with the full Yale Bright Star Catalogue (BSC5) or
the Hipparcos catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class StarRecord:
    """A single entry in the navigation star catalog."""

    star_id: int
    ra_deg: float        # Right Ascension [deg, J2000]
    dec_deg: float       # Declination [deg, J2000]
    mag_visual: float    # Visual magnitude
    mag_swir: float      # Effective SWIR magnitude (1000–2500 nm band)
    spectral_type: str = ""

    @property
    def ra_rad(self) -> float:
        return np.deg2rad(self.ra_deg)

    @property
    def dec_rad(self) -> float:
        return np.deg2rad(self.dec_deg)

    def unit_vector(self) -> np.ndarray:
        """Return the unit direction vector in ICRS (ECI-like) frame."""
        cos_dec = np.cos(self.dec_rad)
        return np.array([
            cos_dec * np.cos(self.ra_rad),
            cos_dec * np.sin(self.ra_rad),
            np.sin(self.dec_rad),
        ])


# ---------------------------------------------------------------------------
# Built-in representative star catalog (subset of bright stars)
# ---------------------------------------------------------------------------
_DEFAULT_STARS: List[StarRecord] = [
    StarRecord(1,  79.172,  45.998,  0.08,  -1.0, "K5III"),    # Capella
    StarRecord(2,  88.793,   7.407,  0.18,  -0.8, "B8Ia"),     # Rigel
    StarRecord(3, 114.825,   5.227,  0.40,   0.6, "K0III"),    # Procyon
    StarRecord(4, 213.915,  19.182,  0.98,   0.5, "B1V"),      # Spica
    StarRecord(5, 116.329,  28.026,  1.14,   1.2, "A1V"),      # Pollux
    StarRecord(6, 101.287,  16.716, -1.46,  -1.4, "A1V"),      # Sirius
    StarRecord(7, 297.695,   8.868,  0.77,   0.4, "A7IV"),     # Altair
    StarRecord(8,  68.980,  16.510,  0.87,   0.9, "K5III"),    # Aldebaran
    StarRecord(9, 247.352, -26.432,  1.06,   0.7, "M1Iab"),    # Antares
    StarRecord(10, 344.413,  15.345,  2.04,   1.8, "G8III"),   # Sadalsuud
]


class StarCatalog:
    """
    Navigation star catalog used by the SWIR star sensor simulator.

    Parameters
    ----------
    stars : list of StarRecord, optional
        Provide a custom star list.  If omitted the built-in representative
        catalog is used.
    mag_limit_swir : float
        Faint-end SWIR magnitude limit; stars fainter than this value are
        excluded from detection.
    """

    def __init__(
        self,
        stars: Optional[List[StarRecord]] = None,
        mag_limit_swir: float = 6.0,
    ) -> None:
        self._stars: List[StarRecord] = list(stars or _DEFAULT_STARS)
        self.mag_limit_swir = mag_limit_swir

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def stars(self) -> List[StarRecord]:
        return list(self._stars)

    def detectable_stars(self) -> List[StarRecord]:
        """Return stars brighter than *mag_limit_swir*."""
        return [s for s in self._stars if s.mag_swir <= self.mag_limit_swir]

    def stars_in_fov(
        self,
        boresight_vec: np.ndarray,
        half_fov_rad: float,
    ) -> List[StarRecord]:
        """
        Return stars whose angular separation from *boresight_vec* is within
        *half_fov_rad* (radians).

        Parameters
        ----------
        boresight_vec : ndarray, shape (3,)
            Unit vector pointing to the field-of-view centre (ICRS frame).
        half_fov_rad : float
            Half-angle of the (square or circular) field of view [rad].
        """
        boresight_vec = boresight_vec / np.linalg.norm(boresight_vec)
        visible = []
        for star in self.detectable_stars():
            uv = star.unit_vector()
            cos_angle = np.clip(np.dot(boresight_vec, uv), -1.0, 1.0)
            if np.arccos(cos_angle) <= half_fov_rad:
                visible.append(star)
        return visible

    def get_by_id(self, star_id: int) -> Optional[StarRecord]:
        for s in self._stars:
            if s.star_id == star_id:
                return s
        return None

    def add_star(self, star: StarRecord) -> None:
        self._stars.append(star)

    def __len__(self) -> int:
        return len(self._stars)

    def __repr__(self) -> str:
        return (
            f"StarCatalog({len(self._stars)} stars, "
            f"SWIR mag limit={self.mag_limit_swir})"
        )
