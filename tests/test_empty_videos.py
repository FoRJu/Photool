"""
A file with no picture to give: an empty recording (a camera that stopped a
clip as it started leaves about a kilobyte, with no video in it) or a damaged
file. It is read once, not on every view and every background pass, and the
grid shows a tile saying so rather than a broken picture.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from thumbnails import extract  # noqa: E402
from thumbnails.store import ThumbnailSize  # noqa: E402
from thumbnails.video import has_video  # noqa: E402
from test_develop_routes import client  # noqa: E402,F401  (a fixture)


class _Cache:
    def get(self, *a, **k):
        return None

    def put(self, *a, **k):
        pass


def test_an_empty_recording_is_read_once_until_it_changes(tmp_path, monkeypatch):
    clip = tmp_path / "CLIP_0001.MP4"
    clip.write_bytes(b"\0" * 1241)
    reads = []
    monkeypatch.setattr(extract, "_source_jpeg", lambda source: reads.append(source) or None)
    monkeypatch.setattr(extract, "_NO_PICTURE", {})
    for _ in range(3):
        assert extract.get_thumbnail(clip, _Cache(), ThumbnailSize.GRID) is None
    assert len(reads) == 1
    clip.write_bytes(b"\0" * 2048)                 # copied again from the card, say
    extract.get_thumbnail(clip, _Cache(), ThumbnailSize.GRID)
    assert len(reads) == 2


def test_ffmpegs_word_for_no_video_at_all():
    assert not has_video("[out#0/image2pipe] Output file does not contain any stream")
    assert not has_video("CLIP.MP4: Invalid data found when processing input")
    assert has_video("[h264] error while decoding MB 12 4")    # a bad frame: try another


def test_the_grid_shows_a_tile_for_a_file_with_no_picture(client, monkeypatch):
    library = Path(client.app.state.config.library_root)
    (library / "CLIP_0002.MP4").write_bytes(b"\0" * 1241)
    monkeypatch.setattr(extract, "_source_jpeg", lambda source: None)
    monkeypatch.setattr(extract, "_NO_PICTURE", {})
    res = client.get("/api/thumb/CLIP_0002.MP4?size=grid")
    assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"
    assert res.headers["x-photool-no-picture"] == "1" and res.content[:2] == b"\xff\xd8"
