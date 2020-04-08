import sys
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import style

try:
    library_dir = Path(__file__).parent.parent.parent.absolute()
except NameError:
    library_dir = Path("/home/callum//Documents/adcp-glider/")
sys.path.append(str(library_dir))
style_path = library_dir / "src" / "visualization" / "presentation.mplstyle"
style.use(str(style_path))
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 2, 3])
ax.show()
