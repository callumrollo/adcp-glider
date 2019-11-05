import sys
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import style
try:
    data_dir = Path(__file__).parent.parent.parent.absolute()
except NameError:
    data_dir = Path('/media/callum/storage/Documents/adcp-glider/')
sys.path.append(str(data_dir))
style_path = data_dir / 'src' / 'visualization' / 'presentation.mplstyle'
style.use(str(style_path))
fig, ax = plt.subplots()
ax.plot([1,2,3], [1, 2, 3])
ax.show()

