"""
NSE Beast System - AI Trader Package
====================================

High-level package for intelligent options trading on NSE with:
- Automated strategy selection based on market regime
- Real-time execution optimization
- Machine learning parameter tuning
- Comprehensive backtesting integration
- Risk management and monitoring

Key Components:
- AITrader: Main trading orchestrator
- BacktesterIntegration: Integration with options_portfolio_backtester
- ExecutionOptimizer: Adaptive execution parameter optimization
- StrategyLearner: ML-based strategy selection
- ExecutionParams: Configuration for order execution
- MarketFeatures: Market condition descriptors

Usage:
    from nse_beast_system.ai_trader import AITrader, AITraderConfig
    
    config = AITraderConfig()
    trader = AITrader(config)
    
    await trader.run_trading_loop()

Author: NSE Beast System Team
Version: 1.0.0
Created: 2025-06-05
"""

from .ai_trader import (
    AITrader,
    AITraderConfig,
    TradeDecision,
    SignalType,
    BacktestResult,
    BacktestStatus
)

from .backtester_integration import (
    BacktesterIntegration,
    NSEOptionsDataAdapter,
    BacktestConfig
)

from .execution_optimizer import (
    ExecutionOptimizer,
    ExecutionParams,
    OptimizationState,
    AdaptiveLearningScheduler,
    FillModel,
    SignalSelector,
    OptimizationPhase
)

from .strategy_learner import (
    StrategyLearner,
    OnlineLearner,
    MarketFeatures,
    StrategyRecord,
    MarketRegime
)

__version__ = "1.0.0"
__author__ = "NSE Beast System Team"
__license__ = "MIT"

__all__ = [
    # AI Trader Core
    "AITrader",
    "AITraderConfig",
    "TradeDecision",
    "SignalType",
    "BacktestResult",
    "BacktestStatus",
    
    # Backtester Integration
    "BacktesterIntegration",
    "NSEOptionsDataAdapter",
    "BacktestConfig",
    
    # Execution Optimization
    "ExecutionOptimizer",
    "ExecutionParams",
    "OptimizationState",
    "AdaptiveLearningScheduler",
    "FillModel",
    "SignalSelector",
    "OptimizationPhase",
    
    # Strategy Learning
    "StrategyLearner",
    "OnlineLearner",
    "MarketFeatures",
    "StrategyRecord",
    "MarketRegime",
]
