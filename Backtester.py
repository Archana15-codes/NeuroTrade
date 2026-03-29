import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum
import warnings
warnings.filterwarnings("ignore")

# ENUMS & DATA STRUCTURES

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    side: PositionSide
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    mae: float = 0.0        # Maximum Adverse Excursion
    mfe: float = 0.0        
    duration_bars: int = 0
    exit_reason: str = ""
    regime: str = ""
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Position:
    side: PositionSide = PositionSide.FLAT
    size: float = 0.0
    entry_price: float = 0.0
    entry_date: Optional[pd.Timestamp] = None
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    bars_held: int = 0
    mae: float = 0.0
    mfe: float = 0.0
