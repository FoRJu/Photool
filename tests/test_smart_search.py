"""
Smart search (analysis.clip, the shows: term of catalog.search): photographs
found by what they show.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from analysis import clip  # noqa: E402
from catalog.database import CatalogDB, PhotoRecord  # noqa: E402
from catalog.search import SearchError, describe, gather, search, unclear  # noqa: E402


def rec(filepath, **kw):
    folder, filename = filepath.rsplit("/", 1)
    base, ext = filename.rsplit(".", 1)
    return PhotoRecord(filepath=filepath, filename=filename, folder=folder,
                       extension="." + ext.lower(), base_name=base, **kw)


LIBRARY = [
    rec("2024/beach/IMG_1.JPG", rating=4), rec("2024/beach/IMG_1.CR3"),
    rec("2024/beach/IMG_2.JPG", rating=1),
    rec("2024/city/IMG_3.JPG", rating=5),
    rec("2024/city/IMG_4.JPG"),
]
# How relevant each file is to "kids on a beach" (analysis.clip.scores): a pair
# is found by either of its files.
BEACH = {"2024/beach/IMG_1.CR3": 0.061, "2024/beach/IMG_2.JPG": 0.031,
         "2024/city/IMG_3.JPG": 0.004, "2024/city/IMG_4.JPG": 0.019}


def found(query, **kw):
    photos = gather(LIBRARY, {}, set(), set())
    return [p.record.filename for p in search(photos, query, scorer=lambda text: BEACH, **kw)]


class TestShows:
    def test_the_nearest_are_found_best_first(self):
        # At least the floor, best first.
        assert found('shows:"kids on a beach"') == ["IMG_1.JPG", "IMG_2.JPG"]

    def test_it_combines_with_the_other_terms(self):
        assert found('shows:"kids on a beach" rating:>=3') == ["IMG_1.JPG"]
        assert found('-shows:"kids on a beach"', sort="filename", desc=False) == ["IMG_3.JPG", "IMG_4.JPG"]

    def test_described_words_are_one_shows_term(self):
        assert found("kids on a beach rating:>=2", describe=True) == ["IMG_1.JPG"]
        assert describe("kids on a beach rating:>=2", True) == ["rating:>=2", 'shows:"kids on a beach"']
        # Without describe, plain words are still names.
        assert found("city", sort="filename", desc=False) == ["IMG_3.JPG", "IMG_4.JPG"]

    def test_nothing_near_enough_is_nothing_found(self):
        photos = gather(LIBRARY, {}, set(), set())
        assert search(photos, "shows:volcano", scorer=lambda text: {k: 0.015 for k in BEACH}) == []

    def test_only_the_closest_is_said_to_be_unclear(self):
        photos = gather(LIBRARY, {}, set(), set())
        weak = search(photos, "shows:giraffe", scorer=lambda text: {k: 0.03 for k in BEACH})
        assert len(weak) == 4 and unclear(weak) == ["giraffe"]
        clear = search(gather(LIBRARY, {}, set(), set()), "shows:beach", scorer=lambda text: BEACH)
        assert unclear(clear) == []

    def test_without_smart_search_it_says_so(self):
        photos = gather(LIBRARY, {}, set(), set())
        with pytest.raises(SearchError, match="smart search"):
            search(photos, "shows:beach")


class TestKept:
    @pytest.fixture()
    def db(self, tmp_path):
        db = CatalogDB(str(tmp_path / "catalog.db"))
        for name, size in (("A.JPG", 10), ("A.CR3", 30), ("B.CR3", 20)):
            db.upsert_photo(PhotoRecord(filepath=f"2024/x/{name}", filename=name, folder="2024/x",
                                        extension=Path(name).suffix.lower(), base_name=Path(name).stem,
                                        file_size=size))
        return db

    def test_one_file_per_pair_is_described_and_found_again_when_replaced(self, db):
        assert sorted(r["filepath"] for r in clip.pending(db)) == ["2024/x/A.JPG", "2024/x/B.CR3"]
        v = np.ones(512, np.float32) / np.sqrt(512)
        clip.store(db, [("2024/x/A.JPG", 10, v), ("2024/x/B.CR3", 20, v)])
        assert clip.pending(db) == []
        assert clip.counts(db) == {"photos": 2, "described": 2}
        clip.store(db, [("2024/x/B.CR3", 999, v)])          # a description of an older file
        assert [r["filepath"] for r in clip.pending(db)] == ["2024/x/B.CR3"]

    def test_scores_are_the_kept_descriptions_against_the_words(self, db, monkeypatch):
        a = np.zeros(512, np.float32)
        a[0] = 1.0
        b = np.zeros(512, np.float32)
        b[1] = 1.0
        clip.store(db, [("2024/x/A.JPG", 10, a), ("2024/x/B.CR3", 20, b)])
        # Relevance is nearness to the words less nearness to "a photo".
        monkeypatch.setattr(clip, "embed_text", lambda text: b if text == clip.NEUTRAL else a)
        assert clip.scores(db, "anything") == {"2024/x/A.JPG": 1.0, "2024/x/B.CR3": -1.0}


@pytest.fixture(scope="module")
def models():
    from config import load_config

    clip.configure(load_config().models_dir)
    if not clip.available():
        pytest.skip("smart search is not set up here")
    return clip


class TestModels:
    def test_the_tokenizer_is_clips(self, models):
        assert models.tokenize("a photo of a dog") == [49406, 320, 1125, 539, 320, 1929, 49407]

    def test_a_picture_is_nearest_the_words_that_say_what_it_is(self, models):
        h, w = 240, 320
        sky = np.zeros((h, w, 3), np.uint8)
        sky[:] = (90, 150, 235)
        sky[h // 2:] = (40, 110, 40)                        # blue above, green below
        red = np.zeros((h, w, 3), np.uint8)
        red[:] = (200, 20, 20)
        v = models.embed_images([sky, red])
        assert v.shape == (2, 512) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)
        field_words = models.embed_text("a green field under a blue sky")
        red_words = models.embed_text("a plain red square")
        assert v[0] @ field_words > v[0] @ red_words
        assert v[1] @ red_words > v[1] @ field_words

    def test_a_short_batch_is_the_same_as_a_full_one(self, models):
        pictures = [np.full((100, 100, 3), 25 * k, np.uint8) for k in range(1, 10)]
        each = models.embed_images(pictures)
        assert np.allclose(models.embed_images(pictures[:1])[0], each[0], atol=1e-4)
