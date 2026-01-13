"""Utility functions module."""


def helper_function(x: int) -> int:
    """A helper function."""
    return x * 2


# Relative import test - same directory
from .submodule import MyClass
