"""
Functions to analyse the bulk data from the whole adcp_mission_overview
"""
import sys
import numpy as np
import pandas as pd
import copy
import gsw
from netCDF4 import Dataset
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    library_dir = Path(__file__).parent.parent.parent.absolute()
except NameError:
    library_dir = Path('/media/callum/storage/Documents/adcp-glider/')
sys.path.append(str(library_dir))
from src.data.beam_mapping import beam2enu, beam_from_center


def list_yos(working_dir):
    yos_path = working_dir.rglob("*cp*.nc")
    yos_list = []
    yos_identifier = []
    for path in yos_path:
        yos_list.append(str(path))
    if len(yos_list) == 0:
        print('Did not find any adcp files of pattern cp*.nc. Aborting')
        return
    yos = np.sort(yos_list)
    dive_limb = np.empty(len(yos), dtype=str)
    # Add column for descent (a) vs ascent (b)
    for i in range(len(yos)):
        yo = yos[i]
        yos_identifier.append(yo[-8:-3])
        dive_limb[i] = yo[-4]

    return yos, yos_identifier, dive_limb


def direction_num_to_climb_phase(num_in):
    if num_in == 1:
        return 'Descent'
    if num_in == 2:
        return 'Ascent'
    return 'Horizontal'

def rounder(float_in, n=2):
    # To neaten cell sizes and distances for plots
    return round(float_in, n)
rounder = np.vectorize(rounder)

@dataclass
class adcp_profile:
    """
    Class object to store data for a single adcp profile
    """
    name: str
    time: datetime
    cell_center: float
    pitch: float
    roll: float
    heading: float
    cor_beam: float
    amp_beam: float
    vel_beam: float
    beam_number: float
    pressure: float
    glider_z: float
    ad2cp_dict: dict

def adcp_import_data(working_dir):
    """

    :param working_dir: Path to the directory where your adcp *.nc files are stored
    :return: A dataframe of per profile info and a dictionary of adcp profiles with keys from the mission overview index
    """
    # First get the paths to the adcp files and some absic info
    yos_paths, yos_identifier, dive_limb = list_yos(working_dir)
    mission_summary = pd.DataFrame({'file_path': yos_paths,
                                    'dive_limb': dive_limb, },
                                   index=yos_identifier)

    # Make an empty dict for more detail
    adcp_profiles_dict_temp = {}
    # Dictionary for detailed info of each dive
    profiles_dict = {}
    # Expand mission_summary by adding extra columns for profile properties
    extras_list = ['averaging_interval', 'powerusage_mW', 'mem_usage_MB_per_hour', 'cell_size',
                   'measurement_interval', 'num_cells', 'num_pings',
                   'blank_dist', 'vert_direction']
    # Intialise the data dictionary
    for index, file_path in zip(mission_summary.index, mission_summary.file_path):
        adcp_dict = {}
        # Unpack the desired data from the nc files into the dict for each profile
        # Todo   config. then tab for autocomplete, has data on all config stuff
        ad2cp_dataset = Dataset(file_path, "r", format="NETCDF4")
        ad2cp_groups = ad2cp_dataset.groups
        config = ad2cp_groups['Config']
        adcp_dict['averaging_interval'] = config.avg_averagingInterval
        adcp_dict['powerusage_mW'] = config.avg_powerUsage
        adcp_dict['mem_usage_MB_per_hour'] = config.avg_memoryUsage
        adcp_dict['cell_size'] = config.avg_cellSize
        adcp_dict['measurement_interval'] = config.avg_measurementInterval
        adcp_dict['num_cells'] = config.avg_nCells
        adcp_dict['num_pings'] = config.avg_nPings
        adcp_dict['blank_dist'] = config.avg_blankingDistance
        adcp_dict['vert_direction'] = direction_num_to_climb_phase(config.plan_verticalDirection)
        adcp_profiles_dict_temp[index] = adcp_dict
        raw_data = ad2cp_groups["Data"]
        data_av = raw_data["Average"]
        # pretty time for plots

        time_secondsfrom1970 = data_av["time"][:]
        time = np.empty([len(time_secondsfrom1970)], dtype=datetime)

        for i in range(len(time_secondsfrom1970)):
            time[i] = datetime(1970, 1, 1, 00, 00) + timedelta(
                seconds=time_secondsfrom1970[i]
            )
        cell_center = adcp_dict['blank_dist'] + np.arange(adcp_dict['cell_size']/2, adcp_dict['cell_size']*15,
                                                           adcp_dict['cell_size'])
        cell_center_neat = rounder(cell_center)

        # extract velocity, correlation and amplitude data from the beams
        beam_array = np.empty(shape=(len(time_secondsfrom1970), len(cell_center), 3))
        beam_array[:] = np.nan
        cor_beam = copy.deepcopy(beam_array)
        amp_beam = copy.deepcopy(beam_array)
        vel_beam = copy.deepcopy(beam_array)
        beam_properties = ("VelocityBeam", "AmplitudeBeam", "CorrelationBeam")
        for i in range(3):
            vel_beam[:, :, i] = data_av[beam_properties[0] + str(i + 1)][:, :]
            amp_beam[:, :, i] = data_av[beam_properties[1] + str(i + 1)][:, :]
            cor_beam[:, :, i] = data_av[beam_properties[2] + str(i + 1)][:, :]

        # z determination from pressure (dbars) fudge factor offset used here.
        # should really use glider's density profile
        pressure = data_av["Pressure"][:]
        # glider_z = 0.7 + gsw.z_from_p(pressure, 50)
        # todo # Sort out gsw and document how to get it
        glider_z = pressure*10
        measurement_z = np.transpose(
            np.tile(np.array(glider_z), (len(cell_center), 1))
        ) - np.tile(np.array(cell_center), (len(glider_z), 1))

        # extract dive parameters

        pitch = data_av["Pitch"][:]
        roll = data_av["Roll"][:]
        heading = data_av["Heading"][:]
        beam_number = data_av["Physicalbeam"][:, :]
        ad2cp_dict = data_av.variables

        profile = adcp_profile(str(index), time, cell_center, pitch, roll, heading, cor_beam, amp_beam, vel_beam,
                               beam_number, pressure, glider_z, ad2cp_dict)
        profiles_dict[index] = profile
    # Add the per profile info to the mission summary
    for extra in extras_list:
        series_list = []
        for key in adcp_profiles_dict_temp.keys():
            adcp_dict = adcp_profiles_dict_temp[key]
            series_list.append(adcp_dict[extra])
        mission_summary[extra] = series_list
    # Dataframe can only accept things that are constant for each profile.
    # Everything else will go in a dictionary
    return mission_summary, profiles_dict


################################################################################


working_dir = library_dir / 'data' / '2019-12-12'
mission_summary, profiles_dict = adcp_import_data(working_dir)

def adcp_profile_data(mission_summary):
    return mission_summary
