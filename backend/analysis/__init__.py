"""
Photo measurements — the shared foundation for culling and for matching a set of
photos to a common exposure and colour before export.

``image_stats`` holds the measurements themselves and knows nothing about
databases or the library; ``groups`` clusters frames that are the same shot from
those measurements; ``hints`` turns them into the soft-frame and blown-highlight
warnings shown while culling; ``service`` runs them over a folder and stores the
result in the catalog.
"""

from analysis.groups import (  # noqa: F401
    MAX_DISTANCE,
    MAX_GAP_SECONDS,
    find_groups,
    suggest_keeper,
)
from analysis.harmonize import (  # noqa: F401
    apply_correction,
    correction,
    set_target,
)
from analysis.hints import (  # noqa: F401
    BLOWN_CLIP,
    MIN_FOLDER_SAMPLE,
    SOFT_RATIO,
    compute_hints,
    folder_baseline,
    photo_hints,
)
from analysis.image_stats import (  # noqa: F401
    STATS_VERSION,
    compute_image_stats,
    hamming_distance,
    stats_are_current,
    stats_from_jpeg,
)
