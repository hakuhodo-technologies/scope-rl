"""Types."""
from typing import Dict, Any, Union
import numpy as np

LoggedDataset = Dict[str, Any]
OPEInputDict = Dict[str, Dict[str, Any]]
Action = Union[int, float, np.integer, np.float, np.float32, np.ndarray]

Numeric = (int, float, np.integer, np.float, np.float32)
