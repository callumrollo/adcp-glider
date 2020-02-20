"""
Functions to calculate the beam angle and vertical miss sampling of anAD2CP
mounted on a glider for given pitch and roll
Fill rotation matrices from transducer along beam velocites to glider reference
frame and east north up frame
Pitch angle positive nose pitch up
Roll angle positive roll to starboard (starboard wing down)
as per Kongsberg

transducer head numbers:
1 aft
2 starboard
3 fore
4 port

"""
import numpy as np


def calc_beam_angles(pitch_angle, roll_angle):
    # Returns angles of each beam from the vertical
    z = np.empty(4)
    z[0] = np.cos(np.deg2rad(47.5 + pitch_angle)) * np.cos(
        np.deg2rad(roll_angle))
    z[1] = np.cos(np.deg2rad(25 - roll_angle)) * np.cos(np.deg2rad(pitch_angle))
    z[2] = np.cos(np.deg2rad(47.5 - pitch_angle)) * np.cos(
        np.deg2rad(roll_angle))
    z[3] = np.cos(np.deg2rad(25 + roll_angle)) * np.cos(np.deg2rad(pitch_angle))
    angles = np.rad2deg(np.arccos(z))
    return angles


def vert_miss(pitch_angle, roll_angle, sample_dist):
    # For the three beams in use, calculates the vertical distance between
    # the bin centre and where each beam samples in m
    beam_angles = calc_beam_angles(pitch_angle, roll_angle)
    if pitch_angle > 0:
        perfect_z = sample_dist * np.cos(np.deg2rad(calc_beam_angles(17.4, 0)))
        actual_z = sample_dist * np.cos(np.deg2rad(beam_angles))
        missmatch = actual_z - perfect_z
        missmatch[0] = np.nan
    else:
        perfect_z = sample_dist * np.cos(np.deg2rad(calc_beam_angles(-17.4, 0)))
        actual_z = sample_dist * np.cos(np.deg2rad(beam_angles))
        missmatch = actual_z - perfect_z
        missmatch[2] = np.nan

    return missmatch


def beam_from_center(pitch_angle, roll_angle, sample_dist):
    # The maximum vertical distance from bin centre
    return np.nanmax(np.abs(vert_miss(pitch_angle, roll_angle, sample_dist)))


beam_from_center = np.vectorize(beam_from_center)


def rotate_pitch(pitch):
    return np.array(
        ((np.cos(np.deg2rad(pitch)), 0, -np.sin(np.deg2rad(pitch))),
         (0, 1, 0),
         (np.sin(np.deg2rad(pitch)), 0, np.cos(np.deg2rad(pitch)))))


def rotate_roll(roll):
    return np.array(((1, 0, 0),
                     (0, np.cos(np.deg2rad(roll)), -np.sin(np.deg2rad(roll))),
                     (0, np.sin(np.deg2rad(roll)), np.cos(np.deg2rad(roll)))))


def rotate_head(heading):
    return np.array(((np.cos(np.deg2rad(-heading + 90)),
                      -np.sin(np.deg2rad(-heading + 90)), 0),
                     (np.sin(np.deg2rad(-heading + 90)),
                      np.cos(np.deg2rad(-heading + 90)), 0),
                     (0, 0, 1)))


# Coordinate transform from adcp files. config avg_beam2xyz_columns_description
beam2xyz_nor_desce = np.array(([ 1.3564, -0.5056, -0.5056],  [0.,     -1.1831,  1.1831],  [0.,      0.5518,  0.5518]))
beam2xyz_nor_climb = np.array(([ 0.5056, -1.3564,  0.5056], [-1.1831,  0.,      1.1831],  [0.5518,  0.,      0.5518]))

def beam2enu(beam_v, pitch, roll, heading, dive_limb="Descent"):
    # Combine all the matrices for a full BEAM to ENU conversion
    beam = np.transpose(np.array(beam_v))
    if dive_limb == "Descent":
        v_ENU_rot = rotate_head(heading) * rotate_roll(roll) * rotate_pitch(
            pitch) * beam2xyz_nor_desce
        v_ENU = v_ENU_rot.dot(beam)
    elif dive_limb == "Ascent":
        v_ENU_rot  = rotate_head(heading) * rotate_roll(roll) * rotate_pitch(
            pitch) * beam2xyz_nor_climb * beam
        v_ENU = v_ENU_rot.dot(beam)
    else:
        print('Must specify  dive direction')
        exit(1)

    return v_ENU
