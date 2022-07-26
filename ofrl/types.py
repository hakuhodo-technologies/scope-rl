"""Types."""
from typing import Dict, Any, Optional
import numpy as np


LoggedDataset = Dict[str, Any]
OPEInputDict = Dict[str, Dict[str, Optional[np.ndarray]]]
