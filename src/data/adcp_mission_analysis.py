"""
Functions to analyse the bulk data from the whole adcp_mission_overview
"""
import sys
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
from dataclasses import dataclass
try:
    library_dir = Path(__file__).parent.parent.parent.absolute()
except NameError:
    library_dir = Path('/media/callum/hd/Documents/adcp-glider/')
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
        exit(1)
    yos = np.sort(yos_list)
    dive_limb = np.empty(len(yos), dtype=str)
    dives = np.empty(len(yos))
    climbs = np.empty(len(yos))
    # create lists of profiles by dive and climb
    for i in range(len(yos)):
        yo = yos[i]
        yos_identifier.append(yo[-8:-3])
        if yo[-4] == "a":
            dive_limb[i] = "a"
            dives[i] = 1
            climbs[i] = np.nan
        else:
            dive_limb[i] = "b"
            dives[i] = np.nan
            climbs[i] = 1
    return yos, yos_identifier, dive_limb, dives, climbs

def direction_num_to_climb_phase(num_in):
    if num_in == 1:
        return 'Descent'
    if num_in == 2:
        return 'Ascent'
    return 'Horizontal'

################################################################################


working_dir = library_dir / 'data' / 'raw'


@dataclass
class adcp_profile:
    name: str
    lon: float = 0.0
    lat: float = 0.0

def adcp_import_data(working_dir):
    """

    :param working_dir: Path to the directory where your adcp *.nc files are stored
    :return: A dictionary of adcp profiles with keys from the mission overview index
    """
    # First get the paths to the adcp files and some absic info
    yos_paths, yos_identifier, dive_limb, dive, climb = list_yos(working_dir)
    mission_summary = pd.DataFrame({'file_path': yos_paths,
                                    'dive_limb': dive_limb,
                                    'descent': dive,
                                    'ascent': climb,},
                                    index=yos_identifier)
    # Make an empty dict for more detail
    adcp_profiles_dict = {}
    # Expand mission_summary by adding extra columns for profile properties
    extras_list = ['powerusage_mW', 'mem_usage_MB_per_hour', 'cell_size',
                   'measurement_interval', 'num_cells', 'num_pings',
                   'blank_dist', 'vert_direction']
    #header_list = list(mission_summary.columns) + extras_list
    #mission_summary = mission_summary.reindex(columns=header_list)
    # Intialise the data dictionary
    for index, file_path in zip(mission_summary.index, mission_summary.file_path):
        adcp_dict = {}
        # Unpack the desired data from the nc files into the dict for each profile
        ad2cp_dataset = Dataset(file_path, "r", format="NETCDF4")
        ad2cp_groups = ad2cp_dataset.groups
        config = ad2cp_groups['Config']
        adcp_dict['powerusage_mW'] = config.avg_powerUsage
        adcp_dict['mem_usage_MB_per_hour'] = config.avg_memoryUsage
        adcp_dict['cell_size'] = config.avg_cellSize
        adcp_dict['measurement_interval'] = config.avg_measurementInterval
        adcp_dict['num_cells'] = config.avg_nCells
        adcp_dict['num_pings'] = config.avg_nPings
        adcp_dict['blank_dist'] = config.avg_blankingDistance
        adcp_dict['vert_direction'] = direction_num_to_climb_phase(config.plan_verticalDirection)
        adcp_profiles_dict[index] = adcp_dict

    # Add this per profile info to the mission summary
    for extra in extras_list:
        series_list = []
        for key in adcp_profiles_dict.keys():
            adcp_dict = adcp_profiles_dict[key]
            series_list.append(adcp_dict[extra])
        mission_summary[extra] = series_list

    # Datafrmae can only accept things that are constant for each profile.
    # Everyting else will go in a dictionary
    return mission_summary
