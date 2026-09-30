"""Offline tests for fiberseg.sem_eds (phase 2). No DLL or instrument needed."""

import ctypes
import io
import json

import numpy as np
import pytest
from PIL import Image

from fiberseg.sem_eds import bruker_api as ba
from fiberseg.sem_eds.microscope import MockMicroscope, QuantaxMicroscope, StageLimitError
from fiberseg.sem_eds.run_tiles import run
from fiberseg.sem_eds.segmenter import ThresholdSegmenter
from fiberseg.sem_eds.settings import SemEdsConfig, load_settings


def test_struct_sizes_match_packed_c_headers():
    assert ctypes.sizeof(ba.TSegment) == 12
    assert ctypes.sizeof(ba.TRTAPISpectrumHeaderRec) == 90
    assert ctypes.sizeof(ba.TRTImageInfoEx) == 36
    assert ctypes.sizeof(ba.TOpenClientOptions) == 76


def test_points_to_segments_uses_x_as_column_and_single_pixel_runs():
    segs = ba.points_to_segments([(10, 3), (7, 99)])
    assert [(s.XStart, s.Y, s.XCount) for s in segs] == [(10, 3, 1), (7, 99, 1)]


def test_parse_spectrum_reads_header_and_counts():
    hdr = ba.TRTAPISpectrumHeaderRec(ChannelCount=4, CalibrationAbs=-0.1, CalibrationLin=0.01)
    buf = bytes(hdr) + np.array([5, 6, 7, 8], dtype="<i4").tobytes()
    h, counts, calib = ba.parse_spectrum(buf)
    assert h.ChannelCount == 4
    assert counts.tolist() == [5, 6, 7, 8]
    assert calib == (-0.1, 0.01)


def test_decode_image_bmp_and_raw():
    px = (np.arange(12, dtype=np.uint8) * 20).reshape(3, 4)
    bmp = io.BytesIO()
    Image.fromarray(px).save(bmp, format="BMP")
    np.testing.assert_allclose(ba.decode_image(bmp.getvalue(), 4, 3), px / 255.0)
    raw = b"\x00" * 50 + px.tobytes()
    np.testing.assert_allclose(ba.decode_image(raw, 4, 3), px / 255.0)


def test_ok_codes_include_warning_and_already_open():
    assert {0, 1, -201} <= ba.OK_CODES
    assert -110 not in ba.OK_CODES


def test_load_settings_rejects_unknown_keys(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("stage_x_limit_um: [0, 10]\n")  # typo -> must fail, not be ignored
    with pytest.raises(ValueError, match="Unknown"):
        load_settings(p)


def test_public_dict_omits_credentials():
    d = SemEdsConfig(user="u", password="p").public_dict()
    assert "user" not in d and "password" not in d


def test_stage_moves_refused_without_limits_or_outside_them():
    scope = QuantaxMicroscope(SemEdsConfig())
    scope._stage_range = {"xmin": -50, "xmax": 50, "ymin": -50, "ymax": 50}
    with pytest.raises(StageLimitError, match="not set"):
        scope.check_stage_target(0, 0)
    scope.cfg = SemEdsConfig(stage_x_limits_um=(0, 100), stage_y_limits_um=(0, 100))
    scope.check_stage_target(50, 50)
    with pytest.raises(StageLimitError):
        scope.check_stage_target(150, 50)


def test_dry_run_end_to_end(tmp_path):
    cfg = SemEdsConfig(
        grid_cols=2, grid_rows=1, image_px=(256, 192), stage_settle_s=0.0, drift_check=False
    )
    run(cfg, MockMicroscope(0), ThresholdSegmenter(cfg), tmp_path)
    assert (tmp_path / "fibers.csv").exists()
    assert (tmp_path / "tile_000" / "image.tif").exists()
    assert "password" not in json.loads((tmp_path / "config.json").read_text())
