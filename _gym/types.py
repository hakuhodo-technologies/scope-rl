"""Types."""
from typing import Dict, Any, Union
import numpy as np

# logged dataset
LoggedDataset = Dict[str, Any]
OPEInputDict = Dict[str, Dict[str, Any]]
Action = Union[int, float, np.integer, np.float, np.ndarray]
