# MAML-based Algorithmic Trading Framework
# Combats model decay from financial market non-stationarity
# via Model-Agnostic Meta-Learning with bi-level optimization.

__version__ = "0.1.0"

from maml_trading.config import MAMLConfig
from maml_trading.data_pipeline import DataPipeline
from maml_trading.task_generator import MarketRegimeTaskGenerator
from maml_trading.model import AdaptiveTradingNetwork
from maml_trading.maml_engine import MAMLTradingEngine
from maml_trading.backtester import WalkForwardBacktester
from maml_trading.metrics import PerformanceMetrics

__all__ = [
    "MAMLConfig",
    "DataPipeline",
    "MarketRegimeTaskGenerator",
    "AdaptiveTradingNetwork",
    "MAMLTradingEngine",
    "WalkForwardBacktester",
    "PerformanceMetrics",
]
