from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

Point2D = Tuple[int, int]


def _euclidean(a: Union[Point2D, np.ndarray], b: Union[Point2D, np.ndarray]) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a_arr - b_arr))


def calculate_ear(eye_points: Sequence[Point2D]) -> float:
    """
    Compute Eye Aspect Ratio (EAR) from 6 ordered points: [p1, p2, p3, p4, p5, p6]

    EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    """
    if len(eye_points) != 6:
        raise ValueError(f"Expected 6 eye points, got {len(eye_points)}")

    p1, p2, p3, p4, p5, p6 = eye_points
    horiz = _euclidean(p1, p4)
    if horiz <= 1e-6:
        return 0.0
    vert1 = _euclidean(p2, p6)
    vert2 = _euclidean(p3, p5)
    return float((vert1 + vert2) / (2.0 * horiz))

