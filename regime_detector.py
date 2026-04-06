from __future__ import annotations

import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Optional HMM support — gracefully degrade if hmmlearn is not installed
try:
    from hmmlearn import hmm as _hmm_module
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn(
        "hmmlearn is not installed. HMM-based regime detection will be unavailable. "
        "Install it with: pip install hmmlearn",
        ImportWarning,
        stacklevel=2,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
