"""
Functions to anlyse the bulk data from the whole adcp_mission_overview
"""
import sys
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
data_dir = Path(__file__).parent.absolute()
sys.path.append(str(data_dir))
from beam_mapping import beam2enu, beam_from_center
print(data_dir)


def list_yos(working_dir):
    yos_path = working_dir.rglob("*cp*.nc")
    yos_list = []
    yos_identifier = []
    for path in yos_path:
        yos_list.append(str(path))
    yos = np.sort(yos_list)
    dive_limb = np.empty(len(yos), dtype=str)
    dives = np.empty(len(yos))
    climbs = np.empty(len(yos))
    # create lists of profiles by dive and climb
    for i in range(len(yos)):
        yo = yos[i]
        yos_identifier.append(yo[-8:-3])
        if yo[-4] == "a":
            dive_limb[i] = "descent"
            dives[i] = 1
            climbs[i] = np.nan
        else:
            dive_limb[i] = "ascent"
            dives[i] = np.nan
            climbs[i] = 1
    return yos, yos_identifier, dive_limb, dives, climbs
working_dir = Path("/media/callum/hd/Documents/sg637/ad2cp_nc/2018-11-20_nosims/")
def adcp_mission_overview(working_dir):
    yos_paths, yos_identifier, dive_limb, dive, climb = list_yos(working_dir)
    mission_summary = pd.DataFrame({'file_path': yos_paths,
                                    'dive_limb': dive_limb,
                                    'descent': dive,
                                    'ascent': climb,},
                                    index=yos_identifier)

    return mission_summary
