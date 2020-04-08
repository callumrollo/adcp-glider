from pathlib import Path
import sys
import numpy as np
from math import isclose

try:
    library_dir = Path(__file__).parent.parent.parent.absolute()
except NameError:
    library_dir = Path("/media/callum/storage/Documents/adcp-glider/")
sys.path.append(str(library_dir))

from src.data.beam_mapping import rotate_pitch, rotate_roll, rotate_head

isclose_vec = np.vectorize(isclose)
vel_xyz = np.array([1.0, -1.0, 2.0])


# first test with no rotation, should be no change
def test_no_rotations():
    assert (rotate_pitch(0).dot(vel_xyz) == vel_xyz).all()
    assert (rotate_roll(0).dot(vel_xyz) == vel_xyz).all()
    assert (rotate_head(90).dot(vel_xyz) == vel_xyz).all()


# test with reversing rotations, should be no change
def test_reverse_rotations():
    assert (rotate_pitch(-90).dot(rotate_pitch(90)).dot(vel_xyz) == vel_xyz).all()
    assert (rotate_roll(-30).dot(rotate_roll(30)).dot(vel_xyz) == vel_xyz).all()
    assert (rotate_head(180).dot(rotate_head(0)).dot(vel_xyz) == vel_xyz).all()


# test with single rotations. Using absolute tolerance of 1e-5
def test_single_rotations():
    assert (
        isclose_vec(
            rotate_pitch(90).dot(vel_xyz), np.array([-2.0, -1.0, 1.0]), abs_tol=1e-5
        )
    ).all()
    assert (
        isclose_vec(
            rotate_roll(90).dot(vel_xyz), np.array([1.0, -2.0, -1.0]), abs_tol=1e-5
        )
    ).all()
    assert (
        isclose_vec(
            rotate_head(0).dot(vel_xyz), np.array([1.0, 1.0, 2.0]), abs_tol=1e-5
        )
    ).all()


# test combined rotations
def test_combi_rotations():
    assert (
        isclose_vec(
            rotate_pitch(-90).dot(rotate_roll(90)).dot(vel_xyz),
            np.array([-1.0, -2.0, -1.0,]),
            abs_tol=1e-5,
        )
    ).all()
    assert (
        isclose_vec(
            rotate_head(0).dot(rotate_roll(180)).dot(vel_xyz),
            np.array([-1.0, 1.0, -2.0,]),
            abs_tol=1e-5,
        )
    ).all()
    assert (
        isclose_vec(
            rotate_head(270).dot(rotate_pitch(180)).dot(vel_xyz),
            np.array([1.0, 1.0, -2.0,]),
            abs_tol=1e-5,
        )
    ).all()
