import math

import pytest

from eye_tracking_project.ear import calculate_ear


def test_calculate_ear_open_eye_rectangle() -> None:
    # p1----p4
    #  |    |
    # p6----p5  (with p2/p3 top points)
    eye = [
        (0, 0),   # p1
        (1, 2),   # p2
        (3, 2),   # p3
        (4, 0),   # p4
        (3, -2),  # p5
        (1, -2),  # p6
    ]
    ear = calculate_ear(eye)
    assert math.isclose(ear, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_calculate_ear_closed_eye_zero_vertical() -> None:
    eye = [
        (0, 0),
        (1, 0),
        (3, 0),
        (4, 0),
        (3, 0),
        (1, 0),
    ]
    ear = calculate_ear(eye)
    assert math.isclose(ear, 0.0, rel_tol=1e-6, abs_tol=1e-6)


def test_calculate_ear_requires_6_points() -> None:
    with pytest.raises(ValueError):
        calculate_ear([(0, 0)])

