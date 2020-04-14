from pathlib import Path
import sys
import numpy as np
from math import isclose

try:
    library_dir = Path(__file__).parent.parent.parent.absolute()
except NameError:
    library_dir = Path("/media/callum/storage/Documents/adcp-glider/")
sys.path.append(str(library_dir))

from src.data.beam_mapping import (
    calc_beam_angles,
    vert_miss,
    rotate_pitch,
    rotate_roll,
    rotate_head,
)

isclose_vec = np.vectorize(isclose)
vel_xyz = np.array([1.0, -1.0, 2.0])


def test_beam_angles():
    """First test with perfect angles"""
    assert isclose_vec(calc_beam_angles(0, 0), [47.5, 25, 47.5, 25], abs_tol=1e-5).all()
    """Pitch up"""
    assert isclose_vec(
        calc_beam_angles(90, 0), [42.5, 90, 137.5, 90], abs_tol=1e-5
    ).all()
    """ Roll to starboard"""
    assert isclose_vec(calc_beam_angles(0, 90), [90, 65, 90, 115], abs_tol=1e-5).all()


def test_vert_miss_perfect():
    """Test with perfect angles during dive and climb"""
    assert (vert_miss(17.4, 0, 15)[:2] == [0.0, 0.0]).all()
    assert np.isnan(vert_miss(17.4, 0, 15)[2])
    assert vert_miss(17.4, 0, 15)[3] == 0.0

    assert (vert_miss(-17.4, 0, 15)[1:] == [0.0, 0.0, 0.0]).all()
    assert np.isnan(vert_miss(-17.4, 0, 15)[0])


def test_vert_miss_pitch():
    """Suboptimal dive angles"""
    miss_down = vert_miss(-25, 0, 10)
    assert isclose(
        miss_down[2],
        10 * (np.cos(np.deg2rad(47.5 - 25)) - np.cos(np.deg2rad(47.5 - 17.4))),
        abs_tol=1e-5,
    )
    assert isclose(
        miss_down[1],
        10
        * (
            np.cos(np.deg2rad(25)) * (np.cos(np.deg2rad(25)) - np.cos(np.deg2rad(17.4)))
        ),
        abs_tol=1e-5,
    )
    assert np.isnan(miss_down[0])
    miss_up = vert_miss(25, 0, 10)
    assert isclose(
        miss_up[0],
        10 * (np.cos(np.deg2rad(47.5 - 25)) - np.cos(np.deg2rad(47.5 - 17.4))),
        abs_tol=1e-5,
    )
    assert isclose(
        miss_up[1],
        10
        * (
            np.cos(np.deg2rad(25)) * (np.cos(np.deg2rad(25)) - np.cos(np.deg2rad(17.4)))
        ),
        abs_tol=1e-5,
    )
    assert np.isnan(miss_up[2])


def test_vert_miss_roll():
    """Suboptimal roll angles"""
    miss_down = vert_miss(-17.4, 10, 10)
    assert isclose(
        miss_down[2],
        10 * (np.cos(np.deg2rad(10)) - 1) * np.cos(np.deg2rad(47.5 - 17.4)),
        abs_tol=1e-5,
    )
    assert isclose(
        miss_down[1],
        10
        * np.cos(np.deg2rad(17.4))
        * (np.cos(np.deg2rad(15)) - np.cos(np.deg2rad(25))),
        abs_tol=1e-5,
    )
    assert np.isnan(miss_down[0])
    miss_up = vert_miss(17.4, -10, 10)
    assert isclose(
        miss_up[0],
        10 * (np.cos(np.deg2rad(10)) - 1) * np.cos(np.deg2rad(47.5 - 17.4)),
        abs_tol=1e-5,
    )
    assert isclose(
        miss_up[1],
        10
        * np.cos(np.deg2rad(17.4))
        * (np.cos(np.deg2rad(35)) - np.cos(np.deg2rad(25))),
        abs_tol=1e-5,
    )
    assert np.isnan(miss_up[2])


def test_no_rotations():
    """First test with no rotation, should be no change"""
    assert (rotate_pitch(0).dot(vel_xyz) == vel_xyz).all()
    assert (rotate_roll(0).dot(vel_xyz) == vel_xyz).all()
    assert (rotate_head(90).dot(vel_xyz) == vel_xyz).all()


def test_reverse_rotations():
    """test with reversing rotations, should be no change"""
    assert (rotate_pitch(-90).dot(rotate_pitch(90)).dot(vel_xyz) == vel_xyz).all()
    assert (rotate_roll(-30).dot(rotate_roll(30)).dot(vel_xyz) == vel_xyz).all()
    assert (rotate_head(180).dot(rotate_head(0)).dot(vel_xyz) == vel_xyz).all()


def test_single_rotations():
    """test with single rotations. Using absolute tolerance of 1e-5"""
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


def test_combi_rotations():
    """test combined rotations"""
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
