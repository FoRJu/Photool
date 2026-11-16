"""
Film scans read at their full depth: TIFF through OpenCV, PSD and PSB through the
app's own reader (export.scans).

The picture is a ramp, different in each channel, with values that differ only in
their low byte, so a read that went through 8 bits is caught.
"""

import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from adjust.render import decode
from export import scans


# ── The pictures ─────────────────────────────────────────────────────

def ramp16(w=300, h=200) -> np.ndarray:
    """(h, w, 3) uint16 RGB: a ramp across, another down, their mix; with a block
    whose values differ only in the low byte."""
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    r = 2000 + 60000 * x + 0 * y
    g = 3000 + 50000 * y + 0 * x
    b = 1000 + 30000 * x + 30000 * y
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint16)
    rgb[10:30, 10:60, :] = (40000 + np.arange(50) * 3)[None, :, None].astype(np.uint16)
    return rgb


def write_tiff(path: Path, arr: np.ndarray) -> Path:
    """``arr`` in RGB(A…) order, written by OpenCV (which wants BGR)."""
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = np.concatenate([arr[..., 2::-1], arr[..., 3:]], axis=-1)
    ok, buf = cv2.imencode(".tif", np.ascontiguousarray(arr))
    assert ok
    path.write_bytes(buf.tobytes())
    return path


def tiff_by_hand(path: Path, arr: np.ndarray, orientation=None, icc=None, extra=0) -> Path:
    """An uncompressed little-endian TIFF of ``arr`` (h, w[, samples]), RGB(A…) or grey,
    uint8, uint16 or float32, with an Orientation tag or a profile when given."""
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    spp = 1 if arr.ndim == 2 else arr.shape[2]
    bits = arr.dtype.itemsize * 8
    fmt = 3 if arr.dtype.kind == "f" else 1
    photometric = 1 if spp <= 2 else 2
    data = arr.astype(arr.dtype.newbyteorder("<")).tobytes()
    entries = []   # (tag, type, values or bytes)
    entries.append((256, 4, [w]))
    entries.append((257, 4, [h]))
    entries.append((258, 3, [bits] * spp))
    entries.append((259, 3, [1]))
    entries.append((262, 3, [photometric]))
    entries.append((273, 4, [0]))          # filled in below
    if orientation is not None:
        entries.append((274, 3, [orientation]))
    entries.append((277, 3, [spp]))
    entries.append((278, 4, [h]))
    entries.append((279, 4, [len(data)]))
    entries.append((284, 3, [1]))
    if spp in (2, 4):
        entries.append((338, 3, [extra]))
    entries.append((339, 3, [fmt] * spp))
    if icc is not None:
        entries.append((34675, 7, icc))
    entries.sort(key=lambda e: e[0])
    ifd_size = 2 + 12 * len(entries) + 4
    blobs, blob_at = b"", 8 + ifd_size
    packed = []
    for tag, typ, values in entries:
        if isinstance(values, bytes):
            body, count = values, len(values)
        else:
            body = struct.pack("<" + ("H" if typ == 3 else "I") * len(values), *values)
            count = len(values)
        packed.append([tag, typ, count, body])
    for item in packed:
        if len(item[3]) > 4:
            if len(blobs) % 2:
                blobs += b"\0"
            item.append(blob_at + len(blobs))
            blobs += item[3]
    data_at = blob_at + len(blobs)
    out = bytearray(b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", len(entries)))
    for item in packed:
        tag, typ, count, body = item[:4]
        if tag == 273:
            body = struct.pack("<I", data_at)
        if len(body) > 4:
            out += struct.pack("<HHII", tag, typ, count, item[4])
        else:
            out += struct.pack("<HHI", tag, typ, count) + body.ljust(4, b"\0")
    out += struct.pack("<I", 0) + blobs + data
    path.write_bytes(bytes(out))
    return path


def matrix_profile(gamma: float) -> bytes:
    """A minimal ICC v2 RGB display profile: sRGB's primaries, one gamma for all three."""
    def s15(v):
        return struct.pack(">i", int(round(v * 65536)))

    def xyz(x, y, z):
        return b"XYZ " + b"\0" * 4 + s15(x) + s15(y) + s15(z)

    curve = b"curv" + b"\0" * 4 + struct.pack(">IH", 1, int(round(gamma * 256))) + b"\0\0"
    desc_text = b"test\0"
    desc = (b"desc" + b"\0" * 4 + struct.pack(">I", len(desc_text)) + desc_text
            + struct.pack(">II", 0, 0) + struct.pack(">HB", 0, 0) + b"\0" * 67)
    tags = [
        (b"desc", desc),
        (b"wtpt", xyz(0.9642, 1.0, 0.8249)),
        (b"rXYZ", xyz(0.4361, 0.2225, 0.0139)),
        (b"gXYZ", xyz(0.3851, 0.7169, 0.0971)),
        (b"bXYZ", xyz(0.1431, 0.0606, 0.7141)),
        (b"rTRC", curve), (b"gTRC", curve), (b"bTRC", curve),
    ]
    offset = 128 + 4 + 12 * len(tags)
    table, data = b"", b""
    for sig, body in tags:
        while len(body) % 4:
            body += b"\0"
        table += sig + struct.pack(">II", offset + len(data), len(body))
        data += body
    size = offset + len(data)
    header = (struct.pack(">I", size) + b"\0" * 4 + struct.pack(">I", 0x02100000)
              + b"mntrRGB XYZ " + b"\0" * 12 + b"acsp" + b"\0" * 24
              + struct.pack(">I", 0) + s15(0.9642) + s15(1.0) + s15(0.8249) + b"\0" * 48)
    assert len(header) == 128
    return header + struct.pack(">I", len(tags)) + table + data


def packbits(row: bytes) -> bytes:
    """PackBits, with repeats for runs of three or more."""
    out, i, n = bytearray(), 0, len(row)
    while i < n:
        j = i
        while j < n and j - i < 128 and row[j] == row[i]:
            j += 1
        if j - i >= 3:
            out += bytes([257 - (j - i), row[i]])
            i = j
            continue
        j = i
        while j < n and j - i < 128:
            if j + 2 < n and row[j] == row[j + 1] == row[j + 2]:
                break
            j += 1
        out += bytes([j - i - 1]) + row[i:j]
        i = j
    return bytes(out)


def literal_packbits(row: bytes) -> bytes:
    """PackBits as literal runs only: quick to write, and the slowest to read."""
    return b"".join(bytes([len(row[i:i + 128]) - 1]) + row[i:i + 128] for i in range(0, len(row), 128))


def write_psd(path: Path, planes: np.ndarray, depth: int, mode: int = 3, compression: int = 0,
              version: int = 1, icc: bytes = None, encode=packbits) -> Path:
    """``planes`` (channels, h, w) as a PSD (version 1) or PSB (2): the reader's inverse."""
    channels, h, w = planes.shape
    dtype = {8: ">u1", 16: ">u2", 32: ">f4"}[depth]
    out = bytearray(b"8BPS" + struct.pack(">H", version) + b"\0" * 6
                    + struct.pack(">HIIHH", channels, h, w, depth, mode))
    out += struct.pack(">I", 0)
    res = b""
    if icc is not None:
        res = b"8BIM" + struct.pack(">H", 1039) + b"\0\0" + struct.pack(">I", len(icc)) + icc
        if len(icc) % 2:
            res += b"\0"
    out += struct.pack(">I", len(res)) + res
    out += struct.pack(">I" if version == 1 else ">Q", 0)
    out += struct.pack(">H", compression)
    data = planes.astype(dtype)
    if compression == 0:
        out += data.tobytes()
    else:
        rows = [encode(data[c, r].tobytes()) for c in range(channels) for r in range(h)]
        fmt = ">H" if version == 1 else ">I"
        out += b"".join(struct.pack(fmt, len(r)) for r in rows)
        out += b"".join(rows)
    path.write_bytes(bytes(out))
    return path


def distinct(channel: np.ndarray) -> int:
    return len(np.unique(channel))


# ── TIFF ─────────────────────────────────────────────────────────────

class TestTiff:
    def test_16bit_rgb_keeps_every_bit(self, tmp_path):
        values = ramp16()
        path = write_tiff(tmp_path / "scan.tif", values)
        rgb = scans.read_linear(path)
        assert rgb.shape == (200, 300, 3) and rgb.dtype == np.float32
        assert distinct(rgb[..., 0]) > 256
        np.testing.assert_allclose(rgb, decode(values / 65535.0), atol=1e-4)
        assert scans.tone_of(path) == "display"

    def test_a_linear_profile_is_read_as_linear(self, tmp_path):
        values = ramp16()
        path = tiff_by_hand(tmp_path / "linear.tif", values, icc=matrix_profile(1.0))
        assert scans.tone_of(path) == "linear"
        np.testing.assert_allclose(scans.read_linear(path), values / 65535.0, atol=1e-6)
        # A gamma 2.2 profile is display-encoded.
        gamma = tiff_by_hand(tmp_path / "gamma.tif", values, icc=matrix_profile(2.2))
        assert scans.tone_of(gamma) == "display"

    def test_8bit_rgb(self, tmp_path):
        values = (ramp16() >> 8).astype(np.uint8)
        rgb = scans.read_linear(write_tiff(tmp_path / "s.tif", values))
        np.testing.assert_allclose(rgb, decode(values / 255.0), atol=1e-6)

    def test_16bit_grey(self, tmp_path):
        grey = ramp16()[..., 0]
        rgb = scans.read_linear(write_tiff(tmp_path / "g.tif", grey))
        assert rgb.shape == (200, 300, 3)
        assert distinct(rgb[..., 0]) > 256
        np.testing.assert_allclose(rgb[..., 1], decode(grey / 65535.0), atol=1e-4)
        assert np.array_equal(rgb[..., 0], rgb[..., 2])

    def test_alpha_is_dropped_not_multiplied(self, tmp_path):
        values = ramp16()
        alpha = np.full(values.shape[:2] + (1,), 1000, np.uint16)
        both = np.concatenate([values, alpha], -1)
        for path in (write_tiff(tmp_path / "a.tif", both),
                     tiff_by_hand(tmp_path / "b.tif", both, extra=2)):
            rgb = scans.read_linear(path)
            np.testing.assert_allclose(rgb, decode(values / 65535.0), atol=1e-4)
        # 16-bit grey with alpha: OpenCV reads it only at 8 bits, so the linear road
        # declines it and the 8-bit road has it.
        grey_alpha = np.stack([values[..., 0], alpha[..., 0]], -1)
        path = tiff_by_hand(tmp_path / "ga.tif", grey_alpha, extra=2)
        assert scans.read_linear(path) is None
        img = np.asarray(scans.read_8bit(path))
        assert abs(int(img[100, 150, 1]) - int(values[100, 150, 0]) // 257) <= 1
        grey_alpha8 = (grey_alpha >> 8).astype(np.uint8)
        rgb = scans.read_linear(tiff_by_hand(tmp_path / "ga8.tif", grey_alpha8, extra=2))
        np.testing.assert_allclose(rgb[..., 1], decode(grey_alpha8[..., 0] / 255.0), atol=1e-6)

    def test_a_fourth_sample_is_dropped(self, tmp_path):
        # An infrared dust channel is written as a fourth sample, like alpha.
        values = ramp16()
        infrared = values[..., :1] // 3
        rgb = scans.read_linear(tiff_by_hand(tmp_path / "ir.tif", np.concatenate([values, infrared], -1)))
        assert rgb.shape == (200, 300, 3)
        np.testing.assert_allclose(rgb, decode(values / 65535.0), atol=1e-4)

    def test_float32_is_linear_with_its_headroom(self, tmp_path):
        values = (ramp16() / 40000.0).astype(np.float32)
        values[0, 0] = -0.5
        path = tiff_by_hand(tmp_path / "f.tif", values)
        assert scans.tone_of(path) == "linear"
        rgb = scans.read_linear(path)
        assert rgb[0, 0].min() == 0.0
        np.testing.assert_allclose(rgb[1:], values[1:], atol=1e-6)
        assert rgb.max() > 1.0

    @pytest.mark.parametrize("orientation, turn", [(6, -1), (8, 1)])
    def test_orientation_is_honoured(self, tmp_path, orientation, turn):
        grey = ramp16()[..., 0]
        path = tiff_by_hand(tmp_path / f"o{orientation}.tif", grey, orientation=orientation)
        assert scans.size_of(path) == (200, 300)
        rgb = scans.read_linear(path)
        assert rgb.shape == (300, 200, 3)
        np.testing.assert_allclose(rgb[..., 0], np.rot90(decode(grey / 65535.0), turn), atol=1e-4)
        # and a reduced decode is turned the same way
        big = tiff_by_hand(tmp_path / f"big{orientation}.tif", np.tile(grey, (6, 6)),
                           orientation=orientation)
        assert scans.size_of(big) == (1200, 1800)
        small = scans.read_linear(big, 100)
        assert small.shape[0] > small.shape[1]
        np.testing.assert_allclose(small[0, 0, 0], small[0, 0, 1])

    def test_a_large_scan_is_reduced_and_still_16_bit(self, tmp_path):
        x = np.linspace(0, 1, 1600)[None, :]
        y = np.linspace(0, 1, 1200)[:, None]
        values = np.stack([2000 + 60000 * x * (y * 0 + 1), 3000 + 50000 * y * (x * 0 + 1),
                           1000 + 30000 * (x + y)], -1).astype(np.uint16)
        path = write_tiff(tmp_path / "big.tif", values)
        reduced = cv2.imdecode(np.fromfile(str(path), np.uint8),
                               cv2.IMREAD_REDUCED_COLOR_2 | cv2.IMREAD_ANYDEPTH)
        assert reduced.dtype == np.uint16 and reduced.shape[1] == 800
        rgb = scans.read_linear(path, 100)
        assert 200 <= max(rgb.shape[:2]) <= 400
        assert distinct(rgb[..., 2]) > 256

    def test_a_non_ascii_name(self, tmp_path):
        path = write_tiff(tmp_path / "胶片 Négatif.tif", ramp16())
        assert scans.read_linear(path).shape == (200, 300, 3)

    def test_read_8bit_of_16bit_grey_saves_as_jpeg(self, tmp_path):
        path = write_tiff(tmp_path / "g.tif", ramp16()[..., 0])
        img = scans.read_8bit(path)
        assert img.mode == "RGB" and img.size == (300, 200)
        img.save(tmp_path / "out.jpg", format="JPEG")
        # And it is the file as it is, encoded back.
        assert abs(int(np.asarray(img)[100, 150, 0]) - int(ramp16()[100, 150, 0]) // 257) <= 1

    def test_what_is_not_read_here_is_none(self, tmp_path):
        path = tmp_path / "cmyk.tif"
        Image.new("CMYK", (40, 30), (10, 20, 30, 40)).save(path)
        assert scans.read_linear(path) is None
        # and the 8-bit road still gives Pillow's reading
        assert scans.read_8bit(path).mode == "RGB"
        junk = tmp_path / "junk.tif"
        junk.write_bytes(b"II*\0garbage")
        assert scans.read_linear(junk) is None and scans.read_8bit(junk) is None


# ── PSD ──────────────────────────────────────────────────────────────

class TestPsd:
    @pytest.mark.parametrize("compression", [0, 1])
    def test_16bit_rgb(self, tmp_path, compression):
        values = ramp16()
        path = write_psd(tmp_path / "s.psd", values.transpose(2, 0, 1), 16, compression=compression)
        assert scans.size_of(path) == (300, 200)
        rgb = scans.read_linear(path)
        np.testing.assert_allclose(rgb, decode(values / 65535.0), atol=1e-6)
        assert distinct(rgb[..., 0]) > 256

    @pytest.mark.parametrize("compression", [0, 1])
    def test_8bit_rgb(self, tmp_path, compression):
        values = (ramp16() >> 8).astype(np.uint8)
        path = write_psd(tmp_path / "s.psd", values.transpose(2, 0, 1), 8, compression=compression)
        np.testing.assert_allclose(scans.read_linear(path), decode(values / 255.0), atol=1e-6)

    def test_16bit_grey(self, tmp_path):
        grey = ramp16()[..., 1]
        path = write_psd(tmp_path / "g.psd", grey[None], 16, mode=1, compression=1)
        rgb = scans.read_linear(path)
        np.testing.assert_allclose(rgb[..., 2], decode(grey / 65535.0), atol=1e-6)
        assert np.array_equal(rgb[..., 0], rgb[..., 1])

    def test_extra_channels_are_dropped(self, tmp_path):
        values = ramp16().transpose(2, 0, 1)
        planes = np.concatenate([values, values[:1] // 2], 0)
        rgb = scans.read_linear(write_psd(tmp_path / "a.psd", planes, 16, compression=1))
        np.testing.assert_allclose(rgb, decode(values.transpose(1, 2, 0) / 65535.0), atol=1e-6)

    def test_32bit_is_linear(self, tmp_path):
        values = (ramp16() / 50000.0).astype(np.float32)
        path = write_psd(tmp_path / "f.psd", values.transpose(2, 0, 1), 32)
        assert scans.tone_of(path) == "linear"
        np.testing.assert_allclose(scans.read_linear(path), values, atol=1e-6)

    @pytest.mark.parametrize("compression", [0, 1])
    def test_psb(self, tmp_path, compression):
        values = ramp16()
        path = write_psd(tmp_path / "s.psb", values.transpose(2, 0, 1), 16,
                         compression=compression, version=2)
        np.testing.assert_allclose(scans.read_linear(path), decode(values / 65535.0), atol=1e-6)

    def test_the_profile_is_found(self, tmp_path):
        values = ramp16()
        path = write_psd(tmp_path / "l.psd", values.transpose(2, 0, 1), 16,
                         icc=matrix_profile(1.0))
        assert scans.tone_of(path) == "linear"
        np.testing.assert_allclose(scans.read_linear(path), values / 65535.0, atol=1e-6)

    def test_reduced(self, tmp_path):
        values = np.tile(ramp16(), (6, 6, 1))   # 1800 x 1200
        for compression in (0, 1):
            path = write_psd(tmp_path / f"r{compression}.psd", values.transpose(2, 0, 1), 16,
                             compression=compression)
            rgb = scans.read_linear(path, 100)
            assert max(rgb.shape[:2]) == 200
            assert distinct(rgb[..., 0]) > 256

    def test_cmyk_and_truncated_are_none(self, tmp_path):
        planes = np.zeros((4, 20, 30), np.uint16)
        assert scans.read_linear(write_psd(tmp_path / "c.psd", planes, 16, mode=4)) is None
        good = write_psd(tmp_path / "t.psd", ramp16().transpose(2, 0, 1), 16, compression=1)
        for cut in (10, 40, len(good.read_bytes()) // 2):
            short = tmp_path / f"t{cut}.psd"
            short.write_bytes(good.read_bytes()[:cut])
            assert scans.read_linear(short) is None
            scans.read_8bit(short)  # does not raise
        raw = write_psd(tmp_path / "r.psd", ramp16().transpose(2, 0, 1), 16)
        short = tmp_path / "r_short.psd"
        short.write_bytes(raw.read_bytes()[:-1000])
        assert scans.read_linear(short) is None

    def test_read_8bit_of_16bit_psd(self, tmp_path):
        path = write_psd(tmp_path / "s.psd", ramp16().transpose(2, 0, 1), 16, compression=1)
        img = scans.read_8bit(path, 150)
        assert img.mode == "RGB" and max(img.size) == 150


def test_is_scan():
    assert scans.is_scan("a/B.TIF") and scans.is_scan("x.psb")
    assert not scans.is_scan("x.jpg")


# ── The app's own roads ──────────────────────────────────────────────

class TestEngine:
    def test_load_linear_keeps_16_bits_and_develops(self, tmp_path):
        from adjust.render import develop_frame
        from export import engine

        path = write_tiff(tmp_path / "scan.tif", ramp16())
        rgb = engine._load_linear(path, 0)
        assert distinct(rgb[..., 0]) > 256
        frame = engine.linear_base(path, 256)
        assert frame is not None
        img = develop_frame(frame, {"exposure": 1})
        assert img.mode == "RGB" and max(img.size) == 256
        assert engine._source_long_side(path) == 300

    def test_an_export_writes_a_jpeg(self, tmp_path):
        from export.engine import ExportTemplate, export_photo

        out = tmp_path / "out"
        out.mkdir()
        for name, path in (("t", write_tiff(tmp_path / "scan.tif", ramp16())),
                           ("p", write_psd(tmp_path / "scan.psd", ramp16().transpose(2, 0, 1), 16))):
            for develop in (None, {"exposure": 0.5}):
                written = export_photo(path, out, ExportTemplate(max_resolution=200),
                                       develop=develop, suffix=f"-{name}{bool(develop)}")
                img = Image.open(written)
                assert img.format == "JPEG" and max(img.size) == 200

    def test_load_image_of_16bit_psd(self, tmp_path):
        from export import engine

        path = write_psd(tmp_path / "s.psd", ramp16().transpose(2, 0, 1), 16, compression=1)
        img = engine._load_image(path, 0)
        assert img.mode == "RGB" and img.size == (300, 200)


class TestThumbnails:
    def test_source_jpeg_of_16bit_grey_tiff_and_psd(self, tmp_path):
        from thumbnails.extract import _source_jpeg

        grey = write_tiff(tmp_path / "g.tif", ramp16()[..., 0])
        psd = write_psd(tmp_path / "s.psd", ramp16().transpose(2, 0, 1), 16, compression=1)
        for path in (grey, psd):
            data = _source_jpeg(path)
            assert data and data[:2] == b"\xff\xd8"


class TestCatalogue:
    def test_a_folder_of_scans_is_indexed_and_its_edits_round_trip(self, tmp_path):
        import time

        from catalog.database import CatalogDB
        from catalog.service import CatalogService
        from catalog.xmp import read_xmp
        from thumbnails.extract import get_exif_summary

        lib = tmp_path / "library"
        folder = lib / "scans"
        folder.mkdir(parents=True)
        write_tiff(folder / "roll-01.tif", ramp16())
        write_tiff(folder / "roll-02.tiff", ramp16()[..., 0])
        write_psd(folder / "roll-03.psd", ramp16().transpose(2, 0, 1), 16, compression=1)
        Image.new("RGB", (30, 20)).save(folder / "dateless.jpg")
        service = CatalogService(CatalogDB(str(tmp_path / "c.db")), str(lib))
        t0 = time.monotonic()
        result = service.index_folder("scans", [".tif", ".tiff", ".psd", ".jpg"],
                                      exif_extractor=get_exif_summary)
        assert time.monotonic() - t0 < 30
        assert result["indexed"] == 4
        photos = {p.filename: p for p in service.get_photos_in_folder("scans")}
        assert len(photos) == 4
        # No date in the file: listed as any other dateless file is.
        dateless = photos.pop("dateless.jpg").capture_date
        assert all(p.capture_date == dateless for p in photos.values())

        service.set_photo_edit("scans/roll-01.tif", {"exposure": 0.5})
        assert service.photo_edit("scans/roll-01.tif") == {"exposure": 0.5}
        assert read_xmp(folder / "roll-01.tif").develop == {"crs:Exposure2012": "+0.50"}
        fresh = CatalogService(CatalogDB(str(tmp_path / "rebuilt.db")), str(lib))
        fresh.index_folder("scans", [".tif", ".tiff", ".psd"])
        assert fresh.photo_edit("scans/roll-01.tif") == {"exposure": 0.5}
