"""The configuration: paths written from the home folder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from config import Config, load_config  # noqa: E402


def test_a_tilde_is_the_home_folder():
    config = Config(library_root="~/Photos", sd_mount_paths=["~/card", "/Volumes/EOS"])
    home = Path.home()
    assert config.library_root == str(home / "Photos")
    assert config.db_path == str(home / "Photos" / ".photomanager" / "catalog.db")
    assert config.thumbnail_cache == str(home / "Photos" / ".photomanager" / "thumbnails")
    assert config.sd_mount_paths == [str(home / "card"), "/Volumes/EOS"]


def test_other_paths_are_left_alone():
    config = Config(library_root="D:/Photos", db_path="~/catalog.db")
    assert config.library_root == "D:/Photos"
    assert config.db_path == str(Path.home() / "catalog.db")
    assert Config().library_root == ""


def test_a_file_s_tilde_is_expanded(tmp_path):
    file = tmp_path / "config.yaml"
    file.write_text('library_root: "~/Photos"\n')
    assert load_config(str(file)).library_root == str(Path.home() / "Photos")
