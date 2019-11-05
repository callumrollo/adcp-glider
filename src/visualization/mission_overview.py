import sys
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import style
try:
    data_dir = Path(__file__).parent.parent.absolute()
except NameError:
    data_dir = Path('/media/callum/storage/Documents/adcp-glider/src/data')
sys.path.append(str(data_dir))
style_path = data_dir / 'visualisation' / 'presentation.mplstyle'
style.use(str(vis_dir))
fig, ax = plt.subplots()
ax.plot([1,2,3], [1, 2, 3])
ax.show()