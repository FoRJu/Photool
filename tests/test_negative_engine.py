"""
A negative is inverted where it is decoded (export.engine), so every road that takes
a decoded frame gets the scene.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from adjust import negative
from export import engine
from test_negative import film, scene
from test_scans import write_tiff


@pytest.fixture
def neg(tmp_path):
    """A 16-bit TIFF of the standard negative, its description, and a lookup that the
    test sets and changes by hand."""
    v = film(scene())
    path = write_tiff(tmp_path / "neg.tif", np.clip(np.rint(v * 65535), 0, 65535).astype(np.uint16))
    as_decoded = engine.scan_as_decoded(path, 600)
    found = {"d": negative.measure(as_decoded, kind="colour")}
    engine.set_negative_lookup(lambda source: found["d"] if source == path else None)
    engine.clear_preview_cache()
    yield path, found
    engine.set_negative_lookup(None)
    engine.clear_preview_cache()


def _ramp_rises(img) -> bool:
    a = np.asarray(img, dtype=np.float32)
    row = a[a.shape[0] // 4]
    left, right = row[a.shape[1] // 6].mean(), row[a.shape[1] * 5 // 6].mean()
    return right > left


def test_linear_base_gives_the_scene(neg):
    path, found = neg
    frame = engine.linear_base(path, 600)
    assert frame.look is None
    assert _ramp_rises(frame.rgb)
    engine.set_negative_lookup(None)
    engine.clear_preview_cache()
    plain = engine.linear_base(path, 600)
    assert not _ramp_rises(plain.rgb)


def test_changing_the_description_gives_another_frame(neg):
    path, found = neg
    first = engine.linear_base(path, 600)
    key = first.source_key
    found["d"] = {**found["d"], "contrast": 50}
    engine.negative_changed(path)
    second = engine.linear_base(path, 600)
    assert second is not first and second.source_key != key
    assert not np.allclose(first.rgb, second.rgb)
    found["d"] = None
    engine.negative_changed(path)
    third = engine.linear_base(path, 600)
    assert not _ramp_rises(third.rgb)


def test_the_8bit_roads_give_the_scene(neg):
    path, _ = neg
    assert _ramp_rises(engine.camera_image(path, 600))
    assert _ramp_rises(engine.preview_base(path, 600))


def test_the_thumbnail_variant(neg):
    from thumbnails.edited import edit_variant, render_edited
    from thumbnails.store import ThumbnailSize

    path, _ = neg
    stamp = engine.negative_stamp(path)
    assert stamp and edit_variant({}, stamp) and not edit_variant({}, "")
    import io

    data = render_edited(path, {}, ThumbnailSize.GRID)
    assert _ramp_rises(Image.open(io.BytesIO(data)))


def test_the_loupe(neg):
    path, _ = neg
    patch = engine.loupe_image(path, {}, 0.83, 0.25, side=40, preview_size=600)
    dark = engine.loupe_image(path, {}, 0.17, 0.25, side=40, preview_size=600)
    assert np.asarray(patch).mean() > np.asarray(dark).mean()


def test_no_camera_white_balance(neg):
    from adjust import whitebalance

    path, _ = neg
    assert whitebalance.of_source(path) is None


def test_an_export_is_the_scene(neg, tmp_path):
    from export.engine import ExportTemplate, export_photo

    path, _ = neg
    out = tmp_path / "out"
    out.mkdir()
    written = export_photo(path, out, ExportTemplate(max_resolution=600))
    assert _ramp_rises(Image.open(written))
