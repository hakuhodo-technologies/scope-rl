"""Types."""
from typing import Dict, Any, Union, Optional
import numpy as np

LoggedDataset = Dict[str, Any]
OPEInputDict = Dict[str, Dict[str, Optional[np.ndarray]]]
Action = Union[int, float, np.integer, np.float, np.float32, np.ndarray]

Numeric = (int, float, np.integer, np.float, np.float32)
