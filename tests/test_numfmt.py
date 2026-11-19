"""Platform-stable formatting: ties round the same way whatever the float summation order."""

from __future__ import annotations

import math

import pytest

from passk.numfmt import fixed


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        (0.1375, 3, "0.138"),
        (0.13749999999999998, 3, "0.138"),  # one ulp below the tie, as 3.12 summation can give
        (0.13750000000000004, 3, "0.138"),
        (0.5375, 3, "0.538"),
        (-0.1375, 3, "-0.138"),
        (0.0, 3, "0.000"),
        (-1e-12, 3, "0.000"),  # never prints a negative zero
        (1.0, 2, "1.00"),
    ],
)
def test_fixed_rounds_ties_the_same_everywhere(value: float, digits: int, expected: str) -> None:
    assert fixed(value, digits) == expected


def test_fixed_signed_prefix() -> None:
    assert fixed(0.258, 3, signed=True) == "+0.258"
    assert fixed(-0.258, 3, signed=True) == "-0.258"
    assert fixed(0.0, 3, signed=True) == "+0.000"


def test_fixed_matches_plain_formatting_away_from_ties() -> None:
    for value in (0.123456, 0.999, 0.0004, 12.3456):
        assert fixed(value, 3) == f"{value:.3f}"
    assert math.isclose(float(fixed(0.3333333, 3)), 0.333)
