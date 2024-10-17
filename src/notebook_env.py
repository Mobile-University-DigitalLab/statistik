import sys
import socket

# set plotting font style
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
rcParams["figure.figsize"] = (12, 6)
rcParams["axes.labelsize"] = 16
rcParams["axes.titlesize"] = 17
rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["legend.fontsize"] = "large"

# set pandas display options
import pandas as pd

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

# clean namespace
del rcParams

# machine/server
print(
    f"\n---------------------------------\nWorking on the host: {socket.gethostname()}"
)

# Python version
print(f"\n---------------------------------\nPython version: {sys.version}")

# Python kernel
print(f"\n---------------------------------\nPython interpreter: {sys.executable}")


