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


uvw2beam_climb = np.array(
    ((np.sin(np.deg2rad(47.5)), 0., np.cos(np.deg2rad(47.5))),
     (0, np.sin(np.deg2rad(25)), np.cos(np.deg2rad(25))),
     (0, -np.sin(np.deg2rad(25)), np.cos(np.deg2rad(25)))))
beam2uvw_climb = np.linalg.inv(uvw2beam_climb)

uvw2beam_dive = np.array(
    ((0, np.sin(np.deg2rad(25)), np.cos(np.deg2rad(25))),
    (-np.sin(np.deg2rad(47.5)), 0.,np.cos(np.deg2rad(47.5))),
    (0, -np.sin(np.deg2rad(25)),np.cos(np.deg2rad(25)))))
beam2uvw_dive = np.linalg.inv(uvw2beam_dive)


def beam2enu(beam_v, pitch, roll, heading, dive_limb="dive"):
    # Combine all the matrices for a full BEAM to ENU conversion
    beam = np.transpose(np.array(beam_v))
    if dive_limb == "dive":
        v_ENU = rotate_head(heading) * rotate_roll(roll) * rotate_pitch(
            pitch) * beam2uvw_dive * beam
    else:
        v_ENU = rotate_head(heading) * rotate_roll(roll) * rotate_pitch(
            pitch) * beam2uvw_climb * beam
    return v_ENU
