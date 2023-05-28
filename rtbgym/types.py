"""Types."""
from typing import Union
import numpy as np


Action = Union[int, float, np.integer, np.int64, np.int32, np.float64, np.float32, np.ndarray]
Numeric = (int, float, np.integer, np.int64, np.int32, np.float64, np.float32)
