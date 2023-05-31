# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Types."""
from typing import Union
import numpy as np


Action = Union[
    int, float, np.integer, np.int64, np.int32, np.float64, np.float32, np.ndarray
]
Numeric = (int, float, np.integer, np.int64, np.int32, np.float64, np.float32)
