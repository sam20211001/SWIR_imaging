"""
Coordinate Conversion
======================
Transforms star direction vectors from the celestial (ICRS) frame to
the optical sensor image plane, following the pipeline:

  ICRS  →  Body frame  →  Optical frame  →  Image plane (pixels)

Key equations mirror those in Wang et al. (2022), Section 2.1:

  * Attitude matrix  A  rotates ICRS unit vectors into sensor body frame.
  * The optical system projects the body-frame vector onto the focal plane
    using the pin-hole (perspective) model with focal length *f*.
  * The image-plane position is offset by the principal point (cx, cy)
    and scaled by the pixel pitch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class CoordinateConverter:
    """
    Coordinate converter: ICRS → image plane.

    Parameters
    ----------
    attitude_matrix : ndarray, shape (3, 3)
        Rotation matrix from ICRS to sensor optical frame.
        Columns are the sensor X, Y, Z (optical axis) unit vectors
        expressed in ICRS coordinates.
    focal_length_mm : float
        Effective focal length of the optical system [mm].
    pixel_pitch_um : float
        Detector pixel pitch [μm].
    image_size : tuple (width, height)
        Number of pixels in the horizontal and vertical directions.
    principal_point : tuple (cx, cy), optional
        Principal point offset in pixels from the image centre.
        Defaults to (0, 0) – i.e. the principal point is at the centre.
    """

    def __init__(
        self,
        attitude_matrix: np.ndarray,
        focal_length_mm: float,
        pixel_pitch_um: float,
        image_size: Tuple[int, int],
        principal_point: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.A = np.asarray(attitude_matrix, dtype=float)
        if self.A.shape != (3, 3):
            raise ValueError("attitude_matrix must be 3×3")
        self.focal_length_px = focal_length_mm * 1e3 / pixel_pitch_um
        self.pixel_pitch_um = pixel_pitch_um
        self.image_size = image_size
        self.cx = image_size[0] / 2.0 + principal_point[0]
        self.cy = image_size[1] / 2.0 + principal_point[1]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def icrs_to_image(
        self,
        unit_vec_icrs: np.ndarray,
    ) -> Optional[Tuple[float, float]]:
        """
        Project a star direction vector (ICRS) onto the image plane.

        Parameters
        ----------
        unit_vec_icrs : ndarray, shape (3,)
            Unit direction vector of the star in ICRS frame.

        Returns
        -------
        (x_px, y_px) : tuple of float
            Sub-pixel image coordinates (column, row), measured from the
            top-left corner.  Returns *None* if the star is behind the
            sensor (z_opt ≤ 0).
        """
        vec_opt = self.A @ unit_vec_icrs
        z = vec_opt[2]
        if z <= 0:
            return None

        # Pin-hole projection
        x_fp = vec_opt[0] / z * self.focal_length_px
        y_fp = vec_opt[1] / z * self.focal_length_px

        # Convert to pixel coordinates (image origin at top-left)
        x_px = self.cx + x_fp
        # y-axis points down in image coordinates (row increases downward),
        # so we subtract y_fp to convert from the optical frame where y points up.
        y_px = self.cy - y_fp

        return (x_px, y_px)

    def is_in_frame(self, x_px: float, y_px: float) -> bool:
        """Return True when the pixel position falls within the image."""
        w, h = self.image_size
        return 0 <= x_px < w and 0 <= y_px < h

    @classmethod
    def from_boresight_ra_dec(
        cls,
        ra_deg: float,
        dec_deg: float,
        roll_deg: float,
        focal_length_mm: float,
        pixel_pitch_um: float,
        image_size: Tuple[int, int],
        principal_point: Tuple[float, float] = (0.0, 0.0),
    ) -> "CoordinateConverter":
        """
        Convenience constructor: build the attitude matrix from boresight
        pointing (RA, Dec) and roll angle.

        Parameters
        ----------
        ra_deg, dec_deg : float
            Boresight pointing in ICRS [degrees].
        roll_deg : float
            Roll angle around the boresight [degrees].
        """
        ra = np.deg2rad(ra_deg)
        dec = np.deg2rad(dec_deg)
        roll = np.deg2rad(roll_deg)

        # Boresight unit vector (optical Z axis in ICRS)
        z_opt = np.array([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec),
        ])

        # Choose a reference "north" vector for zero roll
        north = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(north, z_opt)) > 0.99:
            north = np.array([1.0, 0.0, 0.0])

        y_opt_0 = np.cross(z_opt, north)
        y_opt_0 /= np.linalg.norm(y_opt_0)
        x_opt_0 = np.cross(y_opt_0, z_opt)

        # Apply roll rotation around Z axis
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        x_opt = cos_r * x_opt_0 + sin_r * y_opt_0
        y_opt = -sin_r * x_opt_0 + cos_r * y_opt_0

        # Build rotation matrix: columns are x_opt, y_opt, z_opt in ICRS
        R = np.column_stack([x_opt, y_opt, z_opt])
        # A maps ICRS → optical: rows are [x_opt, y_opt, z_opt] in ICRS
        A = R.T

        return cls(A, focal_length_mm, pixel_pitch_um, image_size,
                   principal_point)
