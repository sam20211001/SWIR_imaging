"""
Unit tests for the SWIR Star Sensor Research Framework.

Run with:
  python -m pytest tests/ -v
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from research_framework.star_catalog import StarCatalog, StarRecord
from research_framework.coordinate_conversion import CoordinateConverter
from research_framework.atmospheric_model import AtmosphericModel, AtmosphericParams
from research_framework.energy_transfer import EnergyTransfer, OpticsParams
from research_framework.image_formation import ImageFormation
from research_framework.star_map_simulator import (
    StarMapSimulator, SimulationConfig,
)
from analysis.evaluation_metrics import StarMapEvaluator


# ---------------------------------------------------------------------------
# StarCatalog tests
# ---------------------------------------------------------------------------

class TestStarCatalog:
    def test_default_catalog_not_empty(self):
        cat = StarCatalog()
        assert len(cat) > 0

    def test_detectable_stars_within_mag_limit(self):
        cat = StarCatalog(mag_limit_swir=1.0)
        for s in cat.detectable_stars():
            assert s.mag_swir <= 1.0

    def test_stars_in_fov_returns_subset(self):
        cat = StarCatalog()
        # Boresight towards a known star (Procyon)
        boresight = cat.get_by_id(3).unit_vector()
        fov_stars = cat.stars_in_fov(boresight, half_fov_rad=math.radians(15))
        assert len(fov_stars) >= 1
        # All returned stars should be within the FoV
        for s in fov_stars:
            cos_a = np.dot(boresight / np.linalg.norm(boresight),
                           s.unit_vector())
            angle = math.acos(max(-1, min(1, cos_a)))
            assert angle <= math.radians(15) + 1e-9

    def test_get_by_id(self):
        cat = StarCatalog()
        s = cat.get_by_id(1)
        assert s is not None
        assert s.star_id == 1

    def test_add_star(self):
        cat = StarCatalog()
        before = len(cat)
        cat.add_star(StarRecord(999, 0.0, 0.0, 5.0, 5.0))
        assert len(cat) == before + 1

    def test_unit_vector_is_unit(self):
        cat = StarCatalog()
        for s in cat.stars:
            uv = s.unit_vector()
            assert abs(np.linalg.norm(uv) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# CoordinateConverter tests
# ---------------------------------------------------------------------------

class TestCoordinateConverter:
    def _make_converter(self):
        return CoordinateConverter.from_boresight_ra_dec(
            ra_deg=114.825,
            dec_deg=5.227,
            roll_deg=0.0,
            focal_length_mm=100.0,
            pixel_pitch_um=15.0,
            image_size=(1024, 1024),
        )

    def test_boresight_projects_to_centre(self):
        """The boresight unit vector should project near the image centre."""
        conv = self._make_converter()
        # boresight vector = third row of A^T = third column of A (row of A^T)
        boresight_icrs = conv.A[2]
        xy = conv.icrs_to_image(boresight_icrs)
        assert xy is not None
        x, y = xy
        assert abs(x - 512) < 1.0
        assert abs(y - 512) < 1.0

    def test_behind_sensor_returns_none(self):
        conv = self._make_converter()
        # Anti-boresight vector
        boresight_icrs = conv.A[2]
        xy = conv.icrs_to_image(-boresight_icrs)
        assert xy is None

    def test_is_in_frame(self):
        conv = self._make_converter()
        assert conv.is_in_frame(100.0, 200.0) is True
        assert conv.is_in_frame(-1.0, 512.0) is False
        assert conv.is_in_frame(512.0, 1024.0) is False


# ---------------------------------------------------------------------------
# AtmosphericModel tests
# ---------------------------------------------------------------------------

class TestAtmosphericModel:
    def test_sky_background_positive(self):
        model = AtmosphericModel(AtmosphericParams(zenith_angle_deg=30.0))
        assert model.sky_background_level() > 0

    def test_sky_background_decreases_with_altitude(self):
        p_low = AtmosphericParams(altitude_km=0.0, zenith_angle_deg=30.0)
        p_high = AtmosphericParams(altitude_km=10.0, zenith_angle_deg=30.0)
        assert (AtmosphericModel(p_low).sky_background_level()
                > AtmosphericModel(p_high).sky_background_level())

    def test_refraction_zero_at_zenith(self):
        model = AtmosphericModel(AtmosphericParams(zenith_angle_deg=0.0))
        # At zenith (elevation=90°) refraction should be very small
        assert model.refraction_offset_arcsec() < 0.1

    def test_refraction_increases_toward_horizon(self):
        m20 = AtmosphericModel(AtmosphericParams(zenith_angle_deg=20.0))
        m60 = AtmosphericModel(AtmosphericParams(zenith_angle_deg=60.0))
        assert m60.refraction_offset_arcsec() > m20.refraction_offset_arcsec()

    def test_turbulence_fwhm_positive(self):
        model = AtmosphericModel()
        assert model.turbulence_fwhm_arcsec() > 0

    def test_aerosol_transmission_in_range(self):
        for aod in [0.0, 0.1, 0.5, 1.0]:
            p = AtmosphericParams(aerosol_optical_depth=aod)
            T = AtmosphericModel(p).aerosol_transmission()
            assert 0.0 < T <= 1.0

    def test_aerosol_transmission_decreases_with_aod(self):
        T1 = AtmosphericModel(AtmosphericParams(aerosol_optical_depth=0.1)).aerosol_transmission()
        T2 = AtmosphericModel(AtmosphericParams(aerosol_optical_depth=0.5)).aerosol_transmission()
        assert T1 > T2

    def test_summary_has_required_keys(self):
        model = AtmosphericModel()
        s = model.summary()
        for key in ["sky_background_level", "refraction_arcsec",
                    "turbulence_seeing_fwhm_arcsec", "aerosol_transmission",
                    "air_mass"]:
            assert key in s


# ---------------------------------------------------------------------------
# EnergyTransfer tests
# ---------------------------------------------------------------------------

class TestEnergyTransfer:
    def _make_energy(self, with_atm=True):
        op = OpticsParams(
            aperture_diameter_mm=50.0,
            focal_length_mm=100.0,
            integration_time_ms=100.0,
        )
        atm = AtmosphericModel() if with_atm else None
        return EnergyTransfer(op, atm)

    def test_bright_star_more_signal_than_faint(self):
        et = self._make_energy()
        assert et.star_signal_electrons(0.0) > et.star_signal_electrons(3.0)

    def test_signal_electrons_positive(self):
        et = self._make_energy()
        assert et.star_signal_electrons(1.0) > 0

    def test_snr_positive(self):
        et = self._make_energy()
        assert et.snr(1.0) > 0

    def test_bright_star_higher_snr(self):
        et = self._make_energy()
        assert et.snr(0.0) > et.snr(4.0)

    def test_limiting_magnitude_reasonable(self):
        et = self._make_energy()
        lim = et.limiting_magnitude(min_snr=5.0)
        assert 0.0 < lim < 10.0

    def test_background_electrons_no_atm(self):
        et = self._make_energy(with_atm=False)
        assert et.background_electrons_per_pixel() == 0.0


# ---------------------------------------------------------------------------
# ImageFormation tests
# ---------------------------------------------------------------------------

class TestImageFormation:
    def _make_renderer(self):
        return ImageFormation(
            image_size=(256, 256),
            adc_bits=12,
            full_well_capacity_e=50_000.0,
            read_noise_e=30.0,
            dark_current_e_per_s=500.0,
            integration_time_ms=100.0,
            seed=0,
        )

    def test_render_output_shape(self):
        renderer = self._make_renderer()
        img = renderer.render([], [], psf_fwhm_pixels=2.0)
        assert img.shape == (256, 256)

    def test_render_star_visible(self):
        renderer = self._make_renderer()
        img = renderer.render(
            star_positions=[(128.0, 128.0)],
            star_signals_e=[20_000.0],
            psf_fwhm_pixels=2.5,
        )
        centre_val = int(img[128, 128])
        corner_val = int(img[0, 0])
        assert centre_val > corner_val

    def test_render_no_saturation_within_well(self):
        renderer = self._make_renderer()
        img = renderer.render(
            star_positions=[(64.0, 64.0)],
            star_signals_e=[1000.0],
            psf_fwhm_pixels=2.0,
        )
        max_dn = (1 << 12) - 1
        # No overflow should occur with a modest signal
        assert int(img.max()) <= max_dn

    def test_diffraction_fwhm_positive(self):
        fwhm = ImageFormation.diffraction_fwhm_pixels(
            wavelength_nm=1550, aperture_diameter_mm=50,
            focal_length_mm=100, pixel_pitch_um=15,
        )
        assert fwhm > 0

    def test_combined_psf_fwhm(self):
        fwhm = ImageFormation.combined_psf_fwhm(2.0, 1.5)
        assert abs(fwhm - math.sqrt(4.0 + 2.25)) < 1e-9


# ---------------------------------------------------------------------------
# StarMapSimulator integration test
# ---------------------------------------------------------------------------

class TestStarMapSimulator:
    def _make_config(self, zenith_deg=30.0) -> SimulationConfig:
        return SimulationConfig(
            boresight_ra_deg=114.825,
            boresight_dec_deg=5.227,
            half_fov_deg=15.0,
            optics=OpticsParams(
                aperture_diameter_mm=50.0,
                focal_length_mm=100.0,
                integration_time_ms=100.0,
                pixel_pitch_um=15.0,
            ),
            atmosphere=AtmosphericParams(
                altitude_km=0.0,
                zenith_angle_deg=zenith_deg,
            ),
            image_width=256,
            image_height=256,
            random_seed=42,
        )

    def test_simulate_returns_result(self):
        sim = StarMapSimulator()
        result = sim.simulate(self._make_config())
        assert result.image is not None
        assert result.image.shape == (256, 256)

    def test_larger_zenith_angle_more_refraction(self):
        sim = StarMapSimulator()
        r20 = sim.simulate(self._make_config(zenith_deg=20.0))
        r60 = sim.simulate(self._make_config(zenith_deg=60.0))
        atm20 = r20.atmospheric_summary["refraction_arcsec"]
        atm60 = r60.atmospheric_summary["refraction_arcsec"]
        assert atm60 > atm20

    def test_hazy_sky_reduces_transmission(self):
        cfg_clear = self._make_config()
        cfg_clear.atmosphere.aerosol_optical_depth = 0.05
        cfg_hazy = self._make_config()
        cfg_hazy.atmosphere.aerosol_optical_depth = 0.40
        sim = StarMapSimulator()
        r_clear = sim.simulate(cfg_clear)
        r_hazy = sim.simulate(cfg_hazy)
        t_clear = r_clear.atmospheric_summary["aerosol_transmission"]
        t_hazy = r_hazy.atmospheric_summary["aerosol_transmission"]
        assert t_clear > t_hazy


# ---------------------------------------------------------------------------
# StarMapEvaluator tests
# ---------------------------------------------------------------------------

class TestStarMapEvaluator:
    def _make_image(self):
        img = np.zeros((64, 64), dtype=np.uint16)
        img[32, 32] = 3000
        img[31, 32] = 2000
        img[33, 32] = 2000
        img[32, 31] = 2000
        img[32, 33] = 2000
        img += 50   # background
        return img

    def test_evaluate_returns_report(self):
        ev = StarMapEvaluator(self._make_image(), background_dn=50.0)
        report = ev.evaluate([(32.0, 32.0, 1)])
        assert report is not None

    def test_gray_mean_positive(self):
        ev = StarMapEvaluator(self._make_image())
        report = ev.evaluate([(32.0, 32.0, 1)])
        assert report.image_gray_mean > 0

    def test_star_peak_above_background(self):
        ev = StarMapEvaluator(self._make_image(), background_dn=50.0)
        report = ev.evaluate([(32.0, 32.0, 1)])
        assert report.mean_star_peak > 50.0

    def test_centroid_offset_small_for_exact_position(self):
        ev = StarMapEvaluator(self._make_image(), background_dn=50.0)
        report = ev.evaluate([(32.0, 32.0, 1)])
        assert report.mean_centroid_offset_px < 1.0
