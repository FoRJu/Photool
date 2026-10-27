"""
Exports are copies of photographs already in the library, written under its
export folder; they are not indexed as photographs of their own.
"""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from catalog.database import CatalogDB, PhotoRecord  # noqa: E402
from catalog.service import CatalogService, is_export_folder  # noqa: E402


def _photo(db, folder, name):
    db.upsert_photo(PhotoRecord(filepath=f"{folder}/{name}", filename=name, folder=folder,
                                extension=".jpg", base_name=Path(name).stem))


def test_the_export_folder_is_recognised_however_it_is_spelled():
    assert is_export_folder("Export") and is_export_folder("export/Hawaii 2026")
    assert is_export_folder("EXPORT\\a\\b")
    assert not is_export_folder("export_guard") and not is_export_folder("2026/export")


def test_forgetting_a_folder_tree_clears_every_table_and_nothing_beside_it(tmp_path):
    db = CatalogDB(str(tmp_path / "catalog.db"))
    for folder in ("Export/Parade", "Export", "export_guard", "2026/2026-10-17"):
        _photo(db, folder, "a.jpg")
    db.add_tag_to_photos(["Export/Parade/a.jpg", "2026/2026-10-17/a.jpg"], "family")
    db.set_photo_edit("Export/a.jpg", {"exposure": 1.0})
    db.add_to_basket("Export/Parade/a.jpg", "Export/Parade", "a.jpg")

    assert db.forget_folder_tree("export") == 2
    left = sorted(p.filepath for f in ("Export/Parade", "Export", "export_guard", "2026/2026-10-17")
                  for p in db.get_photos_in_folder(f))
    assert left == ["2026/2026-10-17/a.jpg", "export_guard/a.jpg"]
    with db._connect() as conn:
        for table in ("photo_tags", "photo_edits", "export_basket"):
            rows = conn.execute(f"SELECT filepath FROM {table}").fetchall()
            assert all(not r[0].lower().startswith("export/") for r in rows), table


def test_indexing_skips_the_export_folder(tmp_path):
    root = tmp_path / "lib"
    for folder in ("export/Parade", "2026/2026-10-17"):
        (root / folder).mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(root / folder / "a.jpg")
    service = CatalogService(CatalogDB(str(tmp_path / "catalog.db")), str(root))
    assert service.index_folder("export/Parade", [".jpg"]) == {
        "indexed": 0, "folder": "export/Parade", "skipped": "export"}
    assert service.index_folder("2026/2026-10-17", [".jpg"])["indexed"] == 1


def test_the_library_folder_list_leaves_exports_out(tmp_path):
    from api.routes import _discover_image_folders

    for folder in ("Export/Parade", "2026/2026-10-17", "2026/export"):
        (tmp_path / folder).mkdir(parents=True)
        (tmp_path / folder / "a.jpg").write_bytes(b"x")
    found = [f["path"] for f in _discover_image_folders(tmp_path, {".jpg"})]
    assert found == ["2026/export", "2026/2026-10-17"]
