"""Canonical 1/5/8-bit explicit-mantissa format; exact ties-to-even rounding."""

from dataclasses import dataclass
from fractions import Fraction

MAX_VALUE = Fraction(32640)
MIN_POSITIVE = Fraction(1, 2**24)
SIGN_TOKENS = ("S0", "S1")
EXPONENT_TOKENS = ("E0", "E1")
MANTISSA_TOKENS = ("M0", "M1")
ANSWER_LENGTH = 16  # 14 numeric bits and two separators.


def rational(value):
    return value if isinstance(value, Fraction) else Fraction(value)


@dataclass(frozen=True)
class Encoded:
    sign: int
    exponent: int
    mantissa: int

    def __post_init__(self):
        if self.sign not in (0, 1) or not 0 <= self.exponent <= 31 or not 0 <= self.mantissa <= 255:
            raise ValueError("Invalid sign/exponent/mantissa")
        if self.mantissa == 0 and (self.sign != 0 or self.exponent != 0):
            raise ValueError("Zero must be canonical positive zero")
        if self.exponent > 0 and self.mantissa < 128:
            raise ValueError("Noncanonical nonzero mantissa")

    @property
    def value(self):
        return (-1)**self.sign * Fraction(self.mantissa, 256) * Fraction(2)**(self.exponent-16)

    @property
    def tokens(self):
        return (SIGN_TOKENS[self.sign], "<EXP>",
                *(EXPONENT_TOKENS[int(b)] for b in f"{self.exponent:05b}"), "<MANT>",
                *(MANTISSA_TOKENS[int(b)] for b in f"{self.mantissa:08b}"))


def encode(value):
    value = rational(value)
    magnitude = abs(value)
    if magnitude > MAX_VALUE:
        raise ValueError("Integral outside supported magnitude range")
    if not magnitude:
        return Encoded(0, 0, 0)
    exponent = max(-16, magnitude.numerator.bit_length()-magnitude.denominator.bit_length()+1)
    scaled = magnitude * Fraction(2)**(8-exponent)
    if exponent > -16 and scaled < 128:
        exponent -= 1
        scaled *= 2
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    mantissa = quotient + int(2*remainder > scaled.denominator or
                              (2*remainder == scaled.denominator and quotient % 2 == 1))
    if mantissa == 0:
        return Encoded(0, 0, 0)
    if mantissa == 256:
        mantissa = 128
        exponent += 1
    return Encoded(int(value < 0), exponent+16, mantissa)


def decode(tokens):
    tokens = tuple(tokens)
    if len(tokens) != ANSWER_LENGTH or tokens[1] != "<EXP>" or tokens[7] != "<MANT>":
        raise ValueError("Invalid answer layout")
    if (tokens[0] not in SIGN_TOKENS or any(t not in EXPONENT_TOKENS for t in tokens[2:7])
            or any(t not in MANTISSA_TOKENS for t in tokens[8:])):
        raise ValueError("Invalid answer bit token")
    return Encoded(SIGN_TOKENS.index(tokens[0]),
                   int("".join(t[1] for t in tokens[2:7]), 2),
                   int("".join(t[1] for t in tokens[8:]), 2))


def stable_encoding(value, uncertainty):
    """Require the entire estimated uncertainty interval to share one encoding."""
    value, uncertainty = rational(value), rational(uncertainty)
    if uncertainty < 0:
        raise ValueError("Negative uncertainty")
    center, left, right = (encode(v) for v in (value, value-uncertainty, value+uncertainty))
    if left != center or right != center:
        raise ValueError("Numerical uncertainty crosses an answer rounding boundary")
    return center
