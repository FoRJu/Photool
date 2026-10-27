"""
Anchors: the same things in two frames, to measure the light between them.

Two frames of one event rarely show the same picture. The photographer moves,
zooms, turns; one frame is the choir, the next a soloist. Comparing the frames as
wholes (their brightness, their histograms) then compares their content. But
most frames of an event share *things*: the same people, the same piano, the
same backdrop. The same surface under the same light comes out the same, so
whatever differs between two frames on those surfaces is the light and the
exposure, not what was in front of the camera, at any angle and focal length.

Two kinds of anchor are found:

  faces          the same person in both frames, known from the face clusters
                 (faces.service): the middle of the face (cheeks and nose), its
                 shadows, hair and highlights left out. Each person is compared
                 only with themselves, so complexion does not matter.
  scene points   points of the scene found in both frames and matched (SIFT,
                 with the matches checked against the geometry of two views):
                 a backdrop, the floor, an instrument.

Both are compared in the decode's linear light, before the camera look and
before any edit, and combined into one gain per channel: the frame's exposure
and white balance relative to the other.

Measured on a stage shoot (R6 II, the same person's skin across 52 pairs of
frames, that person held out of the anchors): faces matter most. Scene points
alone barely did better than pasting the edit as it was, because a backdrop is
lit differently from the people in front of it; with faces weighted above them,
the skin came closest to the hero's. A frame zoomed 1.33x and shifted half a
stop and warmer was brought back within 0.05 stop on every channel.
"""

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# A face counts this many scene points: a face is where the eye judges a set, and
# it is lit as the people are; a point on the backdrop is not.
FACE_WEIGHT = 10.0
# Fewer scene points than this are not used at all: between frames shot from
# different places only a handful of points match, and the geometry check that
# throws out false matches needs more than a handful to work (on a stage shoot,
# eight such points disagreed by up to six stops).
MIN_SCENE_POINTS = 30
# A colour from fewer anchors than this is not believed: one face may be at the
# edge of both frames, or half in a spotlight. Its brightness still counts.
MIN_FACES_FOR_COLOUR = 2
SCENE_POINTS = 400
# A face this small at the measuring size has too little skin to read.
MIN_FACE_PX = 16
_SIFT_FEATURES = 3000
_RATIO = 0.75
_RANSAC_PX = 1.5
_PATCH = 3                      # half-size of a scene point's patch
_CLIPPED = 0.85                 # a channel this bright may be clipped
_DARK = 0.003                   # this dark is mostly noise


def skin(rgb: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
    """The median linear colour of a face's skin: the middle of the box (cheeks
    and nose), its darkest fifth (shadow, hair) and brightest tenth (highlights)
    left out. ``box`` is (x, y, w, h) as fractions of the picture."""
    h, w = rgb.shape[:2]
    x, y, bw, bh = box
    x0, x1 = int((x + 0.3 * bw) * w), int((x + 0.7 * bw) * w)
    y0, y1 = int((y + 0.45 * bh) * h), int((y + 0.8 * bh) * h)
    patch = rgb[max(0, y0):max(0, y1), max(0, x0):max(0, x1)].reshape(-1, 3)
    if len(patch) < 20:
        return None
    luma = patch @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    keep = ((luma >= np.quantile(luma, 0.2)) & (luma <= np.quantile(luma, 0.9))
            & (patch.max(axis=1) < _CLIPPED))
    if keep.sum() < 12:
        return None
    return np.median(patch[keep], axis=0)


def face_pairs(hero_rgb, hero_faces, target_rgb, target_faces) -> list:
    """(hero colour, target colour) for every person in both frames. Faces are
    ``(person, x, y, w, h)``; a person of 0 or None is not known and not used."""
    hw, tw = hero_rgb.shape[1], target_rgb.shape[1]
    pairs = []
    for person, *box in hero_faces:
        if not person or box[2] * hw < MIN_FACE_PX:
            continue
        for other, *box2 in target_faces:
            if other == person and box2[2] * tw >= MIN_FACE_PX:
                a, b = skin(hero_rgb, box), skin(target_rgb, box2)
                if a is not None and b is not None:
                    pairs.append((a, b))
    return pairs


def _grey8(rgb):
    from adjust.render import LUMA, encode

    return (np.clip(encode(np.clip(rgb @ LUMA, 0.0, 1.0)), 0.0, 1.0) * 255).astype(np.uint8)


def features(rgb):
    """What scene_pairs needs of one frame, worked out once: its keypoints and
    their descriptors. ``None`` without OpenCV or with nothing to find."""
    try:
        import cv2
    except ImportError:
        return None
    sift = cv2.SIFT_create(nfeatures=_SIFT_FEATURES)
    points, described = sift.detectAndCompute(_grey8(rgb), None)
    if described is None or len(points) < 20:
        return None
    return np.float32([p.pt for p in points]), described


def _patch(rgb, x, y):
    h, w = rgb.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if xi < _PATCH or yi < _PATCH or xi >= w - _PATCH or yi >= h - _PATCH:
        return None
    c = np.median(rgb[yi - _PATCH:yi + _PATCH + 1, xi - _PATCH:xi + _PATCH + 1].reshape(-1, 3), axis=0)
    if c.max() > _CLIPPED or c @ np.array([0.2126, 0.7152, 0.0722]) < _DARK:
        return None
    return c


def scene_pairs(hero_rgb, hero_features, target_rgb, target_features, limit: int = SCENE_POINTS) -> list:
    """(hero colour, target colour) at points of the scene found in both frames."""
    if hero_features is None or target_features is None:
        return []
    import cv2

    (pa_all, da), (pb_all, db) = hero_features, target_features
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < _RATIO * m[1].distance]
    if len(good) < 15:
        return []
    pa = pa_all[[g.queryIdx for g in good]]
    pb = pb_all[[g.trainIdx for g in good]]
    _, inliers = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, _RANSAC_PX, 0.99)
    if inliers is None:
        return []
    pairs = []
    for (xa, ya), (xb, yb), ok in zip(pa, pb, inliers.ravel()):
        if ok:
            a, b = _patch(hero_rgb, xa, ya), _patch(target_rgb, xb, yb)
            if a is not None and b is not None:
                pairs.append((a, b))
    if len(pairs) > limit:
        keep = np.random.default_rng(0).choice(len(pairs), limit, replace=False)
        pairs = [pairs[i] for i in keep]
    return pairs


def solve(faces: list, scene: list) -> Optional[dict]:
    """The gain per channel (in stops) that takes the target to the hero, from the
    anchor pairs: a weighted median, faces weighted FACE_WEIGHT. ``None`` when the
    anchors say too little (no shared face and fewer than MIN_SCENE_POINTS).
    ``colour`` says whether there are enough of them to believe a white balance
    (MIN_FACES_FOR_COLOUR faces, or MIN_SCENE_POINTS points)."""
    if len(scene) < MIN_SCENE_POINTS:
        scene = []
    if not faces and not scene:
        return None
    pairs = [(a, b, FACE_WEIGHT) for a, b in faces] + [(a, b, 1.0) for a, b in scene]
    d = np.array([np.log2(np.maximum(a, 1e-5) / np.maximum(b, 1e-5)) for a, b, _ in pairs])
    weight = np.array([w for *_, w in pairs])

    def weighted_median(v):
        order = np.argsort(v)
        cum = np.cumsum(weight[order])
        return float(v[order][np.searchsorted(cum, cum[-1] / 2.0)])

    stops = np.array([weighted_median(d[:, c]) for c in range(3)])
    luma = np.array([0.2126, 0.7152, 0.0722])
    return {"stops": stops, "exposure": float(stops @ luma), "faces": len(faces), "points": len(scene),
            "colour": len(faces) >= MIN_FACES_FOR_COLOUR or len(scene) >= MIN_SCENE_POINTS,
            "spread": float(np.median(np.abs(d @ luma - stops @ luma)))}
