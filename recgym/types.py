"""Types."""
from typing import Union
import numpy as np


Action = Union[int, float, np.integer, np.int32, np.int64, np.float64, np.float32, np.ndarray]
