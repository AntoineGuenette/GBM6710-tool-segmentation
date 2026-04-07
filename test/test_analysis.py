import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis import compute_IoU

def test_perfect_match():
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[1, 0], [0, 1]])
    assert compute_IoU(A, B) == 1.0

def test_no_overlap():
    A = np.array([[1, 0], [0, 0]])
    B = np.array([[0, 0], [0, 1]])
    assert compute_IoU(A, B) == 0.0

def test_partial_overlap():
    A = np.array([[1, 1], [0, 0]])
    B = np.array([[1, 0], [0, 0]])
    assert compute_IoU(A, B) == pytest.approx(0.5)

def test_both_empty():
    A = np.zeros((2, 2))
    B = np.zeros((2, 2))
    assert compute_IoU(A, B) == 1.0

def test_asymmetric_case():
    A = np.array([[1, 1], [1, 0]])
    B = np.array([[1, 0], [0, 0]])
    # Intersection = 1, Union = 3
    assert compute_IoU(A, B) == pytest.approx(1/3)