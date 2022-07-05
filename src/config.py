from pathlib import Path
from os.path import dirname, realpath

ROOT_DIR = Path(dirname(realpath(__file__))).parent

# Data directory
DATA_DIR = ROOT_DIR / "data"

SRC_DIR = ROOT_DIR / "src"

ES_DIR = SRC_DIR / "es"

PLAYGROUND_DIR = SRC_DIR / "playground"
