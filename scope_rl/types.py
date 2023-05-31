# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Types."""
from typing import Dict, Any, Optional
import numpy as np


LoggedDataset = Dict[str, Any]
OPEInputDict = Dict[str, Dict[str, Optional[np.ndarray]]]
