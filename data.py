import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Union
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    print("[data.py] yfinance not installed — pip install yfinance")

try:
    from fredapi import Fred
    _FRED_AVAILABLE = True
except ImportError:
    _FRED_AVAILABLE = False
    print("[data.py] fredapi not installed — pip install fredapi")
