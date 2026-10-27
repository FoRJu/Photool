"""
The edit model for one photograph.

A package of its own rather than a corner of ``export`` because the same numbers
have three consumers: the export pipeline, the preview, and the XMP sidecar. The
catalog cannot import from ``export``, so anything both of them need lives here.

``develop`` describes an edit (the controls, their ranges, the sidecar names) and
``render`` performs it: a settings dictionary and a decoded frame go in, a
corrected 8-bit image comes out, in linear light so that exposure, highlights and
shadows can rescue a frame rather than re-scale one that was already clipped. The
preview, the library thumbnails and the exported file all go through that one
function, which is what keeps them showing the same picture.
"""

from adjust.develop import (
    adjustment,
    apply_develop,
    apply_tone,
    auto_tone,
    combine,
    describe,
    describe_settings,
    describe_tone,
    is_identity,
    normalize,
    point_of,
    to_crs,
    tone_table,
)

from adjust.render import LinearFrame, auto_develop, develop_frame

__all__ = [
    "LinearFrame",
    "auto_develop",
    "develop_frame",
    "adjustment",
    "apply_develop",
    "apply_tone",
    "auto_tone",
    "combine",
    "describe",
    "describe_settings",
    "describe_tone",
    "is_identity",
    "normalize",
    "point_of",
    "to_crs",
    "tone_table",
]
