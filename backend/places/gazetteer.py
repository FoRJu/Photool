"""
Place names without a web service.

A photograph's coordinates are where someone was standing, which is about as
personal as data gets. Sending them to a geocoding service to learn the name of
the town would hand that service a trail of where the family has been. GeoNames
publishes every town of 5,000 people or more as one file (about 4 MB zipped),
so it is fetched once, like the face models, and every lookup after that is a
nearest-point search on this machine.

The same list answers the other direction: a folder with no GPS can be given a
place by typing its name, and the typed name becomes coordinates for the map.
"""

import io
import logging
import math
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

BASE_URL = "https://download.geonames.org/export/dump"
CITIES = "cities5000"
FILES = {
    f"{CITIES}.txt": f"{BASE_URL}/{CITIES}.zip",
    "admin1CodesASCII.txt": f"{BASE_URL}/admin1CodesASCII.txt",
    "countryInfo.txt": f"{BASE_URL}/countryInfo.txt",
}
_TIMEOUT = 60
EARTH_KM = 6371.0


@dataclass(frozen=True)
class Place:
    name: str
    region: str
    country: str
    country_code: str
    lat: float
    lon: float
    population: int
    km: float = 0.0
    zone: str = ""          # its time zone, as the tz database names it

    def label(self, home_country: str = "") -> str:
        """``Seattle, Washington`` at home, ``Paris, France`` abroad."""
        if self.country_code and self.country_code == home_country:
            tail = self.region or self.country
        else:
            tail = self.country or self.region
        return f"{self.name}, {tail}" if tail and tail != self.name else self.name

    def as_dict(self, home_country: str = "") -> dict:
        return {
            "name": self.name, "region": self.region, "country": self.country,
            "country_code": self.country_code, "lat": self.lat, "lon": self.lon,
            "km": round(self.km, 1), "label": self.label(home_country),
        }


def data_dir(models_dir) -> Path:
    return Path(models_dir) / "geonames"


def is_ready(models_dir) -> bool:
    folder = data_dir(models_dir)
    return all((folder / name).is_file() for name in FILES)


def download(models_dir) -> None:
    """Fetch whichever of the three files is missing. Raises ``OSError`` on failure."""
    folder = data_dir(models_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = folder / name
        if target.is_file():
            continue
        logger.info("Fetching %s", url)
        request = urllib.request.Request(url, headers={"User-Agent": "Photool"})
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            payload = response.read()
        if url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                payload = archive.read(name)
        # Written beside and renamed, so a half-downloaded file is never taken
        # for the real one.
        partial = target.with_suffix(".part")
        partial.write_bytes(payload)
        partial.replace(target)


class Gazetteer:
    """The town list in memory, with nearest-point and by-name lookups."""

    def __init__(self, folder: Path):
        countries: dict[str, str] = {}
        for line in (folder / "countryInfo.txt").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) > 4:
                countries[cols[0]] = cols[4]
        regions: dict[str, str] = {}
        for line in (folder / "admin1CodesASCII.txt").read_text(encoding="utf-8").splitlines():
            cols = line.split("\t")
            if len(cols) > 1:
                regions[cols[0]] = cols[1]

        names, ascii_keys, region_names, codes, pops, lats, lons, zones = [], [], [], [], [], [], [], []
        with open(folder / f"{CITIES}.txt", encoding="utf-8") as handle:
            for line in handle:
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 15:
                    continue
                # PPLX is a section of a town: "Greenwood" for a photo taken at
                # the university, "Paris 16 Passy" for one at the Eiffel Tower.
                # The town is the answer a person gives, so sections are left out.
                if cols[7] == "PPLX":
                    continue
                try:
                    lat, lon = float(cols[4]), float(cols[5])
                    pop = int(cols[14] or 0)
                except ValueError:
                    continue
                names.append(cols[1])
                ascii_keys.append(cols[2].casefold())
                codes.append(cols[8])
                region_names.append(regions.get(f"{cols[8]}.{cols[10]}", ""))
                pops.append(pop)
                lats.append(lat)
                lons.append(lon)
                zones.append(cols[17] if len(cols) > 17 else "")

        self.names = names
        self.keys = [n.casefold() for n in names]
        self.ascii_keys = ascii_keys
        self.regions = region_names
        self.codes = codes
        self.zones = zones
        self.countries = countries
        self.population = np.asarray(pops, dtype=np.int64)
        self.lat = np.asarray(lats, dtype=np.float64)
        self.lon = np.asarray(lons, dtype=np.float64)
        self._lat_r = np.radians(self.lat)
        self._lon_r = np.radians(self.lon)
        self._cos_lat = np.cos(self._lat_r)

    def __len__(self) -> int:
        return len(self.names)

    def _place(self, i: int, km: float = 0.0) -> Place:
        code = self.codes[i]
        return Place(
            name=self.names[i], region=self.regions[i],
            country=self.countries.get(code, code), country_code=code,
            lat=float(self.lat[i]), lon=float(self.lon[i]),
            population=int(self.population[i]), km=km, zone=self.zones[i],
        )

    # A town ten times bigger, at most this much further away than the nearest,
    # is the better name: Paris rather than "Paris 16 Passy", one of its
    # districts, which GeoNames lists as a town of its own.
    BIGGER_KM = 5.0
    BIGGER_RATIO = 10

    def nearest(self, lat: float, lon: float) -> Optional[Place]:
        """The town a person would name for these coordinates."""
        if not len(self) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        la, lo = math.radians(lat), math.radians(lon)
        a = (np.sin((self._lat_r - la) / 2) ** 2
             + math.cos(la) * self._cos_lat * np.sin((self._lon_r - lo) / 2) ** 2)
        km = 2 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        i = int(np.argmin(km))
        near = np.flatnonzero(
            (km <= km[i] + self.BIGGER_KM) & (self.population >= self.BIGGER_RATIO * max(1, self.population[i])))
        if near.size:
            i = int(near[np.argmax(self.population[near])])
        return self._place(i, float(km[i]))

    def search(self, query: str, limit: int = 8, prefer: str = "") -> list[Place]:
        """Towns whose name starts with what was typed: exact names first, then
        the ones in the ``prefer`` country (home), then the biggest.

        Cambridge in England is bigger than Cambridge in Massachusetts, and
        is not the one someone living in Massachusetts means.

        ``Portland, Maine`` narrows by what follows the comma, matched against
        the region or the country, so the smaller Portland can be reached.
        """
        text, _, qualifier = (query or "").partition(",")
        text, qualifier = text.strip().casefold(), qualifier.strip().casefold()
        if not text:
            return []
        hits = []
        for i, (key, ascii_key) in enumerate(zip(self.keys, self.ascii_keys)):
            if not (key.startswith(text) or ascii_key.startswith(text)):
                continue
            if qualifier:
                region = self.regions[i].casefold()
                country = self.countries.get(self.codes[i], "").casefold()
                if not (region.startswith(qualifier) or country.startswith(qualifier)
                        or self.codes[i].casefold() == qualifier):
                    continue
            exact = key == text or ascii_key == text
            elsewhere = bool(prefer) and self.codes[i] != prefer
            hits.append((not exact, elsewhere, -int(self.population[i]), i))
        hits.sort()
        return [self._place(hit[-1]) for hit in hits[: max(1, int(limit))]]


_lock = threading.Lock()
_loaded: dict[str, Gazetteer] = {}


def get(models_dir, fetch: bool = False) -> Optional[Gazetteer]:
    """The loaded list, or ``None`` when it is not on disk and not fetched."""
    key = str(data_dir(models_dir))
    with _lock:
        if key in _loaded:
            return _loaded[key]
        if not is_ready(models_dir):
            if not fetch:
                return None
            download(models_dir)
        gazetteer = Gazetteer(data_dir(models_dir))
        _loaded[key] = gazetteer
        return gazetteer
