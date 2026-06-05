"""
NSE Beast System - Execution Optimizer with Adaptive Learning
==============================================================

This module implements adaptive execution optimization for the AI Trader,
integrating backtester results to continuously improve trading parameters.

Key Features:
- ExecutionParams: Immutable configuration dataclass for all execution parameters
- OptimizationState: Tracks optimization history and best parameters
- ExecutionOptimizer: Main optimizer with gradient-descent and grid-search methods
- AdaptiveLearningScheduler: Determines when to trigger reoptimization

Optimization Strategies:
1. Adaptive Gradient: Nudge parameters when metrics improve, revert if degraded
2. Grid Search: Full parameter sweep across defined ranges
3. Kelly Criterion: Position sizing based on win/loss statistics
4. Dynamic Regime Adjustment: Adjust parameters based on market regime

Integration Points:
- Receives backtest results: Sharpe, max_dd, win_rate, fill_quality, hold_days, pnl
- Outputs optimized ExecutionParams for live trading
- Persists optimization state for recovery and analysis

Author: NSE Beast System Team
Created: 2025-06-05
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
from dataclasses import dataclass, asdict, field, replace
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import threading
import queue

import numpy as np
import pandas as pd


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_optimizer_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Setup JSON structured logging for execution optimizer.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("ExecutionOptimizer")
    logger.setLevel(logging.DEBUG)
    
    # JSON formatter for structured logging
    json_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        '"module": "%(name)s", "message": "%(message)s"}',
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "execution_optimizer.log"),
        maxBytes=50 * 1024 * 1024,
        backupCount=10
    )
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_optimizer_logging()


# ============================================================================
# ENUMS
# ============================================================================

class FillModel(Enum):
    """Available fill models for order execution"""
    MID_PRICE = "mid_price"
    VOLUME_AWARE = "volume_aware"
    MARKET_AT_BID_ASK = "market_at_bid_ask"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"


class SignalSelector(Enum):
    """Available signal selectors for option chain"""
    NEAREST_DELTA = "nearest_delta"
    MAX_OPEN_INTEREST = "max_open_interest"
    HIGHEST_VOLUME = "highest_volume"
    BEST_BID_ASK_SPREAD = "best_bid_ask_spread"


class OptimizationPhase(Enum):
    """Phases of optimization"""
    INITIALIZATION = "initialization"
    GRADIENT_DESCENT = "gradient_descent"
    GRID_SEARCH = "grid_search"
    VALIDATION = "validation"
    PRODUCTION = "production"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class ExecutionParams:
    """
    Immutable dataclass for execution parameters.
    All parameters needed for optimal trade execution and risk management.
    
    Attributes:
        fill_model: Order fill model (mid_price, volume_aware, etc.)
        signal_selector: Option selection strategy (nearest_delta, max_oi, etc.)
        delta_target_call: Target delta for short call legs
        delta_target_put: Target delta for short put legs
        position_size_pct: Position size as percentage of portfolio
        slippage_pct: Expected slippage in percentage
        commission_per_contract: Commission per option contract
        stop_loss_pct: Stop loss threshold in percentage
        profit_target_pct: Profit target threshold in percentage
        min_dte: Minimum days to expiration for entry
        max_dte: Maximum days to expiration for entry
        volume_threshold: Minimum volume threshold for option selection
    """
    fill_model: str = "mid_price"
    signal_selector: str = "nearest_delta"
    delta_target_call: float = -0.30
    delta_target_put: float = -0.30
    position_size_pct: float = 5.0
    slippage_pct: float = 0.5
    commission_per_contract: float = 20.0
    stop_loss_pct: float = -2.0
    profit_target_pct: float = 5.0
    min_dte: int = 7
    max_dte: int = 45
    volume_threshold: float = 100.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ExecutionParams to dictionary for serialization.
        
        Returns:
            Dictionary representation of parameters
        """
        try:
            return asdict(self)
        except Exception as e:
            logger.error(json.dumps({
                "event": "execution_params_to_dict_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ExecutionParams':
        """
        Create ExecutionParams from dictionary.
        
        Args:
            data: Dictionary with parameter values
            
        Returns:
            ExecutionParams instance
            
        Raises:
            ValueError: If required keys are missing
            TypeError: If parameter types are invalid
        """
        try:
            # Validate required keys
            required_keys = {
                'fill_model', 'signal_selector', 'delta_target_call',
                'delta_target_put', 'position_size_pct', 'slippage_pct',
                'commission_per_contract', 'stop_loss_pct', 'profit_target_pct',
                'min_dte', 'max_dte', 'volume_threshold'
            }
            
            missing_keys = required_keys - set(data.keys())
            if missing_keys:
                raise ValueError(f"Missing required keys: {missing_keys}")
            
            # Type validation
            type_checks = {
                'fill_model': str,
                'signal_selector': str,
                'delta_target_call': (int, float),
                'delta_target_put': (int, float),
                'position_size_pct': (int, float),
                'slippage_pct': (int, float),
                'commission_per_contract': (int, float),
                'stop_loss_pct': (int, float),
                'profit_target_pct': (int, float),
                'min_dte': int,
                'max_dte': int,
                'volume_threshold': (int, float)
            }
            
            for key, expected_type in type_checks.items():
                if not isinstance(data[key], expected_type):
                    raise TypeError(
                        f"Parameter '{key}' must be {expected_type}, "
                        f"got {type(data[key])}"
                    )
            
            return ExecutionParams(**data)
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "execution_params_from_dict_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "input_data": str(data)
            }))
            raise


@dataclass
class OptimizationState:
    """
    Tracks the state of optimization including history and best parameters.
    
    Attributes:
        iteration: Current optimization iteration number
        best_params: Best ExecutionParams found so far
        best_sharpe: Best Sharpe ratio achieved
        improvement_history: List of improvement records
        last_optimized: Timestamp of last optimization
    """
    iteration: int = 0
    best_params: ExecutionParams = field(default_factory=ExecutionParams)
    best_sharpe: float = -np.inf
    improvement_history: List[Dict[str, Any]] = field(default_factory=list)
    last_optimized: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        try:
            return {
                "iteration": self.iteration,
                "best_params": self.best_params.to_dict(),
                "best_sharpe": float(self.best_sharpe),
                "improvement_history": self.improvement_history,
                "last_optimized": self.last_optimized.isoformat()
            }
        except Exception as e:
            logger.error(json.dumps({
                "event": "optimization_state_to_dict_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'OptimizationState':
        """Create OptimizationState from dictionary."""
        try:
            return OptimizationState(
                iteration=int(data.get("iteration", 0)),
                best_params=ExecutionParams.from_dict(data.get("best_params", {})),
                best_sharpe=float(data.get("best_sharpe", -np.inf)),
                improvement_history=list(data.get("improvement_history", [])),
                last_optimized=datetime.fromisoformat(
                    data.get("last_optimized", datetime.now().isoformat())
                )
            )
        except Exception as e:
            logger.error(json.dumps({
                "event": "optimization_state_from_dict_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "input_data": str(data)
            }))
            raise


# ============================================================================
# EXECUTION OPTIMIZER
# ============================================================================

class ExecutionOptimizer:
    """
    Main optimizer class for adaptive execution improvement.
    
    Uses backtest results to:
    1. Adaptively nudge parameters when metrics improve
    2. Perform grid search over parameter ranges
    3. Calculate optimal position sizing using Kelly Criterion
    4. Adjust delta targets based on market regime
    5. Persist optimization state for recovery
    
    Attributes:
        initial_params: Starting ExecutionParams
        config: Optimization configuration
        state: Current OptimizationState
        lock: Thread lock for thread-safe operations
    """

    def __init__(
        self,
        initial_params: Optional[ExecutionParams] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ExecutionOptimizer.
        
        Args:
            initial_params: Starting execution parameters
            config: Optimization configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
            TypeError: If parameters are wrong type
        """
        try:
            self.initial_params = initial_params or ExecutionParams()
            self.config = config or self._get_default_config()
            
            # Validate config
            self._validate_config()
            
            # Initialize state
            self.state = OptimizationState(
                best_params=self.initial_params,
                best_sharpe=-np.inf,
                iteration=0
            )
            
            # Thread safety
            self.lock = threading.RLock()
            
            logger.info(json.dumps({
                "event": "execution_optimizer_initialized",
                "initial_params": self.initial_params.to_dict(),
                "config": self.config,
                "timestamp": datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "execution_optimizer_init_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default optimization configuration."""
        return {
            "learning_rate": 0.01,
            "gradient_step_size": 0.05,
            "improvement_threshold": 0.01,
            "revert_threshold": -0.05,
            "max_iterations": 100,
            "grid_search_levels": 5,
            "kelly_multiplier": 0.25,
            "max_position_size_pct": 10.0,
            "min_position_size_pct": 1.0,
        }

    def _validate_config(self) -> None:
        """Validate optimization configuration."""
        required_keys = {
            "learning_rate", "gradient_step_size", "improvement_threshold",
            "revert_threshold", "max_iterations", "grid_search_levels",
            "kelly_multiplier", "max_position_size_pct", "min_position_size_pct"
        }
        
        missing_keys = required_keys - set(self.config.keys())
        if missing_keys:
            raise ValueError(f"Missing config keys: {missing_keys}")
        
        # Validate ranges
        if not 0 < self.config["learning_rate"] < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if self.config["max_iterations"] < 1:
            raise ValueError("max_iterations must be >= 1")

    def update_from_backtest(
        self,
        sharpe: float,
        max_dd: float,
        win_rate: float,
        avg_fill_quality: float,
        avg_hold_days: float,
        total_pnl: float
    ) -> ExecutionParams:
        """
        Adaptively update parameters based on backtest results.
        
        Uses gradient descent approach:
        - If sharpe improves beyond threshold, nudge parameters in positive direction
        - If sharpe degrades beyond threshold, revert to previous parameters
        - Adjust parameters based on specific metrics (fill quality, hold days, etc.)
        
        Args:
            sharpe: Sharpe ratio from backtest
            max_dd: Maximum drawdown percentage
            win_rate: Winning trade percentage (0-1)
            avg_fill_quality: Average fill quality (0-1)
            avg_hold_days: Average holding period in days
            total_pnl: Total P&L from backtest
            
        Returns:
            Updated ExecutionParams
            
        Raises:
            ValueError: If metrics are out of valid ranges
        """
        try:
            with self.lock:
                # Validate input metrics
                if not 0 <= win_rate <= 1:
                    raise ValueError(f"win_rate must be 0-1, got {win_rate}")
                if not 0 <= avg_fill_quality <= 1:
                    raise ValueError(f"avg_fill_quality must be 0-1, got {avg_fill_quality}")
                if avg_hold_days < 0:
                    raise ValueError(f"avg_hold_days must be >= 0, got {avg_hold_days}")
                
                improvement_score = self._compute_improvement_score(
                    self.state.best_sharpe, sharpe, 0, max_dd
                )
                
                logger.info(json.dumps({
                    "event": "backtest_update_started",
                    "iteration": self.state.iteration,
                    "current_sharpe": self.state.best_sharpe,
                    "new_sharpe": sharpe,
                    "improvement_score": improvement_score,
                    "metrics": {
                        "max_dd": max_dd,
                        "win_rate": win_rate,
                        "avg_fill_quality": avg_fill_quality,
                        "avg_hold_days": avg_hold_days,
                        "total_pnl": total_pnl
                    },
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Check if improvement is significant
                if improvement_score > self.config["improvement_threshold"]:
                    # Update best metrics
                    self.state.best_sharpe = sharpe
                    
                    # Generate new parameters
                    new_params = replace(self.state.best_params)
                    
                    # Adjust parameters based on metrics
                    new_params = self._apply_adaptive_adjustments(
                        new_params, sharpe, max_dd, win_rate,
                        avg_fill_quality, avg_hold_days
                    )
                    
                    # Update best params
                    self.state.best_params = new_params
                    
                    # Record improvement
                    self.state.improvement_history.append({
                        "iteration": self.state.iteration,
                        "timestamp": datetime.now().isoformat(),
                        "sharpe": sharpe,
                        "max_dd": max_dd,
                        "improvement_score": improvement_score,
                        "new_params": new_params.to_dict()
                    })
                    
                    logger.info(json.dumps({
                        "event": "improvement_accepted",
                        "iteration": self.state.iteration,
                        "new_sharpe": sharpe,
                        "improvement_score": improvement_score,
                        "new_params": new_params.to_dict()
                    }))
                
                elif improvement_score < self.config["revert_threshold"]:
                    logger.warning(json.dumps({
                        "event": "degradation_detected",
                        "iteration": self.state.iteration,
                        "improvement_score": improvement_score,
                        "reverting_params": self.state.best_params.to_dict()
                    }))
                
                self.state.iteration += 1
                self.state.last_optimized = datetime.now()
                
                return self.state.best_params
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "update_from_backtest_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "metrics": {
                    "sharpe": sharpe,
                    "max_dd": max_dd,
                    "win_rate": win_rate,
                    "avg_fill_quality": avg_fill_quality,
                    "avg_hold_days": avg_hold_days
                }
            }))
            raise

    def _apply_adaptive_adjustments(
        self,
        params: ExecutionParams,
        sharpe: float,
        max_dd: float,
        win_rate: float,
        avg_fill_quality: float,
        avg_hold_days: float
    ) -> ExecutionParams:
        """
        Apply adaptive adjustments to parameters based on backtest metrics.
        
        Args:
            params: Current parameters
            sharpe: Sharpe ratio
            max_dd: Maximum drawdown
            win_rate: Winning trade percentage
            avg_fill_quality: Average fill quality
            avg_hold_days: Average holding days
            
        Returns:
            Adjusted ExecutionParams
        """
        try:
            adjustments = {}
            
            # Adjust fill model based on fill quality
            adjustments['fill_model'] = self._adjust_fill_model(
                win_rate, avg_fill_quality
            )
            
            # Adjust position sizing using Kelly Criterion
            adjustments['position_size_pct'] = self._kelly_position_size(
                win_rate, avg_fill_quality, max_dd
            )
            
            # Adjust delta targets based on win rate and drawdown
            delta_call, delta_put = self._adjust_delta_targets(win_rate, max_dd)
            adjustments['delta_target_call'] = delta_call
            adjustments['delta_target_put'] = delta_put
            
            # Adjust DTE range based on average holding days
            min_dte, max_dte = self._adjust_dte(avg_hold_days)
            adjustments['min_dte'] = min_dte
            adjustments['max_dte'] = max_dte
            
            # Adjust stop loss based on drawdown and Sharpe
            adjustments['stop_loss_pct'] = self._adjust_stop_loss(max_dd, sharpe)
            
            # Create new parameters with adjustments
            return replace(params, **adjustments)
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "apply_adaptive_adjustments_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    def run_full_optimization(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> ExecutionParams:
        """
        Perform full grid search optimization over parameter ranges.
        
        Evaluates all combinations of parameters in grid and selects best.
        Useful for periodic full re-optimization.
        
        Args:
            param_grid: Dictionary mapping parameter names to lists of values
                       e.g., {'delta_target_call': [-0.25, -0.30, -0.35],
                              'position_size_pct': [3.0, 5.0, 7.0]}
        
        Returns:
            Best ExecutionParams found
            
        Raises:
            ValueError: If param_grid is invalid
        """
        try:
            with self.lock:
                logger.info(json.dumps({
                    "event": "grid_search_started",
                    "param_grid": {k: str(v) for k, v in param_grid.items()},
                    "total_combinations": self._calculate_grid_size(param_grid),
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Get all combinations
                combinations = self._generate_parameter_combinations(param_grid)
                
                best_score = -np.inf
                best_params = self.state.best_params
                
                for idx, combo in enumerate(combinations):
                    try:
                        # Create parameter set from combination
                        current_dict = self.state.best_params.to_dict()
                        current_dict.update(combo)
                        test_params = ExecutionParams.from_dict(current_dict)
                        
                        # Score this parameter set (would be from backtest)
                        # For now, we use a heuristic score
                        score = self._score_parameters(test_params)
                        
                        if score > best_score:
                            best_score = score
                            best_params = test_params
                            
                            logger.debug(json.dumps({
                                "event": "grid_search_improvement",
                                "combination": idx,
                                "score": score,
                                "params": test_params.to_dict()
                            }))
                    
                    except Exception as e:
                        logger.warning(json.dumps({
                            "event": "grid_search_combination_error",
                            "combination": idx,
                            "error": str(e)
                        }))
                        continue
                
                # Update state with best parameters
                self.state.best_params = best_params
                self.state.iteration += 1
                self.state.last_optimized = datetime.now()
                
                logger.info(json.dumps({
                    "event": "grid_search_completed",
                    "best_score": best_score,
                    "best_params": best_params.to_dict(),
                    "timestamp": datetime.now().isoformat()
                }))
                
                return best_params
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "run_full_optimization_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    def _calculate_grid_size(self, param_grid: Dict[str, List[Any]]) -> int:
        """Calculate total number of parameter combinations."""
        size = 1
        for values in param_grid.values():
            size *= len(values)
        return size

    def _generate_parameter_combinations(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from grid."""
        try:
            import itertools
            
            keys = list(param_grid.keys())
            value_lists = [param_grid[k] for k in keys]
            combinations = []
            
            for values in itertools.product(*value_lists):
                combo = dict(zip(keys, values))
                combinations.append(combo)
            
            return combinations
        except Exception as e:
            logger.error(json.dumps({
                "event": "generate_combinations_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    def _score_parameters(self, params: ExecutionParams) -> float:
        """
        Score a parameter set using heuristic rules.
        Higher score is better.
        """
        score = 0.0
        
        # Prefer mid_price fill model
        if params.fill_model == "mid_price":
            score += 10.0
        elif params.fill_model == "volume_aware":
            score += 8.0
        
        # Prefer reasonable position sizing
        if 2.0 <= params.position_size_pct <= 8.0:
            score += 5.0
        
        # Prefer moderate slippage assumptions
        if 0.3 <= params.slippage_pct <= 1.0:
            score += 5.0
        
        # Prefer reasonable DTE range
        if 7 <= params.min_dte <= 14 and 30 <= params.max_dte <= 50:
            score += 5.0
        
        return score

    def _adjust_fill_model(self, win_rate: float, avg_fill_quality: float) -> str:
        """
        Adjust fill model based on win rate and fill quality.
        
        Args:
            win_rate: Winning trade percentage
            avg_fill_quality: Average fill quality (0-1)
            
        Returns:
            Recommended fill model string
        """
        try:
            # High quality fills with good win rate: use aggressive
            if avg_fill_quality > 0.8 and win_rate > 0.55:
                return FillModel.AGGRESSIVE.value
            # Moderate quality: volume aware
            elif avg_fill_quality > 0.6:
                return FillModel.VOLUME_AWARE.value
            # Default: mid price
            else:
                return FillModel.MID_PRICE.value
        except Exception as e:
            logger.error(json.dumps({
                "event": "adjust_fill_model_error",
                "error": str(e),
                "win_rate": win_rate,
                "avg_fill_quality": avg_fill_quality
            }))
            return FillModel.MID_PRICE.value

    def _kelly_position_size(
        self,
        win_rate: float,
        avg_fill_quality: float,
        max_dd: float
    ) -> float:
        """
        Calculate optimal position sizing using modified Kelly Criterion.
        
        Kelly % = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        
        Simplified with quality and drawdown factors.
        
        Args:
            win_rate: Winning trade percentage (0-1)
            avg_fill_quality: Average fill quality (0-1)
            max_dd: Maximum drawdown percentage
            
        Returns:
            Recommended position size as percentage of portfolio
        """
        try:
            # Base Kelly calculation
            if win_rate == 0 or win_rate == 1:
                base_kelly = 0.05
            else:
                loss_rate = 1 - win_rate
                # Assume 1:1 win/loss ratio for simplified Kelly
                base_kelly = (2 * win_rate - 1)
            
            # Apply quality factor (0 to 1)
            quality_adjusted = base_kelly * (0.5 + 0.5 * avg_fill_quality)
            
            # Apply drawdown penalty
            dd_factor = max(0.5, 1.0 - abs(max_dd) / 100.0)
            kelly_sized = quality_adjusted * dd_factor
            
            # Apply Kelly multiplier (conservative)
            kelly_multiplier = self.config.get("kelly_multiplier", 0.25)
            final_position_size = kelly_sized * kelly_multiplier
            
            # Constrain to limits
            max_pct = self.config.get("max_position_size_pct", 10.0)
            min_pct = self.config.get("min_position_size_pct", 1.0)
            
            final_position_size = np.clip(final_position_size, min_pct, max_pct)
            
            return float(final_position_size)
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "kelly_position_size_error",
                "error": str(e),
                "win_rate": win_rate,
                "avg_fill_quality": avg_fill_quality,
                "max_dd": max_dd
            }))
            return self.config.get("min_position_size_pct", 1.0)

    def _adjust_delta_targets(
        self,
        win_rate: float,
        max_dd: float
    ) -> Tuple[float, float]:
        """
        Adjust delta targets for calls and puts based on market performance.
        
        More conservative (closer to 0) if win rate is low or drawdown high.
        More aggressive (further from 0) if win rate is high.
        
        Args:
            win_rate: Winning trade percentage (0-1)
            max_dd: Maximum drawdown percentage
            
        Returns:
            Tuple of (delta_target_call, delta_target_put)
        """
        try:
            # Base deltas
            base_call = -0.30
            base_put = -0.30
            
            # Adjust aggressiveness based on win rate
            if win_rate > 0.60:
                # More aggressive: further from 0
                delta_multiplier = 1.2
            elif win_rate > 0.50:
                # Neutral
                delta_multiplier = 1.0
            else:
                # Conservative: closer to 0
                delta_multiplier = 0.75
            
            # Apply drawdown penalty
            if max_dd < -5:
                dd_penalty = 0.8
            elif max_dd < -10:
                dd_penalty = 0.6
            else:
                dd_penalty = 1.0
            
            delta_call = base_call * delta_multiplier * dd_penalty
            delta_put = base_put * delta_multiplier * dd_penalty
            
            # Clamp to reasonable bounds
            delta_call = np.clip(delta_call, -0.50, -0.10)
            delta_put = np.clip(delta_put, -0.50, -0.10)
            
            return float(delta_call), float(delta_put)
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "adjust_delta_targets_error",
                "error": str(e),
                "win_rate": win_rate,
                "max_dd": max_dd
            }))
            return -0.30, -0.30

    def _adjust_dte(self, avg_hold_days: float) -> Tuple[int, int]:
        """
        Adjust min and max DTE based on average holding period.
        
        If trades are held longer on average, use longer DTE options.
        If trades exit quickly, use shorter DTE options.
        
        Args:
            avg_hold_days: Average holding period in days
            
        Returns:
            Tuple of (min_dte, max_dte)
        """
        try:
            if avg_hold_days <= 1:
                # Very short term
                min_dte = 3
                max_dte = 14
            elif avg_hold_days <= 3:
                # Short term
                min_dte = 5
                max_dte = 21
            elif avg_hold_days <= 7:
                # Medium term
                min_dte = 7
                max_dte = 35
            else:
                # Longer term
                min_dte = 10
                max_dte = 45
            
            return min_dte, max_dte
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "adjust_dte_error",
                "error": str(e),
                "avg_hold_days": avg_hold_days
            }))
            return 7, 45

    def _adjust_stop_loss(self, max_dd: float, sharpe: float) -> float:
        """
        Adjust stop loss threshold based on drawdown and Sharpe ratio.
        
        Tighter stops if drawdown is high (volatile).
        Looser stops if Sharpe is good.
        
        Args:
            max_dd: Maximum drawdown percentage
            sharpe: Sharpe ratio
            
        Returns:
            Stop loss percentage (negative value)
        """
        try:
            base_stop = -2.0
            
            # Tighter stops with high drawdown
            if max_dd < -10:
                dd_factor = 1.5  # Make stop 1.5x tighter
            elif max_dd < -5:
                dd_factor = 1.2
            else:
                dd_factor = 1.0
            
            # Looser stops with high Sharpe
            if sharpe > 2.0:
                sharpe_factor = 0.8  # Make stop 0.8x (looser)
            elif sharpe > 1.0:
                sharpe_factor = 0.9
            else:
                sharpe_factor = 1.0
            
            stop_loss = base_stop * dd_factor * sharpe_factor
            
            # Clamp to reasonable bounds
            stop_loss = np.clip(stop_loss, -5.0, -0.5)
            
            return float(stop_loss)
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "adjust_stop_loss_error",
                "error": str(e),
                "max_dd": max_dd,
                "sharpe": sharpe
            }))
            return -2.0

    def _compute_improvement_score(
        self,
        old_sharpe: float,
        new_sharpe: float,
        old_dd: float,
        new_dd: float
    ) -> float:
        """
        Compute improvement score from old to new metrics.
        
        Positive score = improvement, Negative = degradation.
        Weighted combination of Sharpe improvement and drawdown reduction.
        
        Args:
            old_sharpe: Previous Sharpe ratio
            new_sharpe: New Sharpe ratio
            old_dd: Previous max drawdown
            new_dd: New max drawdown
            
        Returns:
            Improvement score
        """
        try:
            sharpe_improvement = new_sharpe - old_sharpe
            dd_improvement = old_dd - new_dd  # Positive is better (less drawdown)
            
            # Weight: 70% Sharpe, 30% drawdown
            score = 0.7 * sharpe_improvement + 0.3 * (dd_improvement / 100.0)
            
            return float(score)
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "compute_improvement_score_error",
                "error": str(e),
                "old_sharpe": old_sharpe,
                "new_sharpe": new_sharpe,
                "old_dd": old_dd,
                "new_dd": new_dd
            }))
            return 0.0

    def get_current_params(self) -> ExecutionParams:
        """
        Get current best ExecutionParams.
        
        Returns:
            Current optimal ExecutionParams
        """
        with self.lock:
            return self.state.best_params

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get optimization history.
        
        Returns:
            List of historical optimization records
        """
        with self.lock:
            return list(self.state.improvement_history)

    def save_state(self, path: str) -> None:
        """
        Save optimization state to JSON file.
        
        Args:
            path: File path to save state
            
        Raises:
            IOError: If file write fails
        """
        try:
            with self.lock:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                
                state_dict = self.state.to_dict()
                
                with open(path, 'w') as f:
                    json.dump(state_dict, f, indent=2, default=str)
                
                logger.info(json.dumps({
                    "event": "optimization_state_saved",
                    "path": path,
                    "timestamp": datetime.now().isoformat()
                }))
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "save_state_error",
                "path": path,
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    def load_state(self, path: str) -> None:
        """
        Load optimization state from JSON file.
        
        Args:
            path: File path to load state from
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If state format is invalid
        """
        try:
            with self.lock:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"State file not found: {path}")
                
                with open(path, 'r') as f:
                    state_dict = json.load(f)
                
                self.state = OptimizationState.from_dict(state_dict)
                
                logger.info(json.dumps({
                    "event": "optimization_state_loaded",
                    "path": path,
                    "iteration": self.state.iteration,
                    "best_sharpe": self.state.best_sharpe,
                    "timestamp": datetime.now().isoformat()
                }))
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "load_state_error",
                "path": path,
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise


# ============================================================================
# ADAPTIVE LEARNING SCHEDULER
# ============================================================================

class AdaptiveLearningScheduler:
    """
    Scheduler to determine when reoptimization should be triggered.
    
    Tracks:
    - Number of trades since last optimization
    - Current drawdown level
    - Market regime changes
    
    Reoptimizes when:
    - Fixed number of trades completed
    - Drawdown exceeds threshold
    - Market regime changes detected
    
    Attributes:
        reoptimize_every_n_trades: Trades between reoptimizations
        drawdown_trigger: Drawdown threshold to trigger reoptimization
        regime_change_trigger: Flag to trigger on regime change
        trade_count: Current trade count since last optimization
        lock: Thread lock for safety
    """

    def __init__(
        self,
        reoptimize_every_n_trades: int = 50,
        drawdown_trigger: float = -5.0,
        regime_change_trigger: bool = True
    ):
        """
        Initialize AdaptiveLearningScheduler.
        
        Args:
            reoptimize_every_n_trades: Number of trades between optimizations
            drawdown_trigger: Drawdown percentage to trigger optimization
            regime_change_trigger: Enable regime change based triggering
            
        Raises:
            ValueError: If parameters are invalid
        """
        try:
            if reoptimize_every_n_trades < 1:
                raise ValueError("reoptimize_every_n_trades must be >= 1")
            
            self.reoptimize_every_n_trades = reoptimize_every_n_trades
            self.drawdown_trigger = drawdown_trigger
            self.regime_change_trigger = regime_change_trigger
            
            self.trade_count = 0
            self.last_optimization_trade_count = 0
            self.lock = threading.RLock()
            
            logger.info(json.dumps({
                "event": "adaptive_learning_scheduler_initialized",
                "reoptimize_every_n_trades": reoptimize_every_n_trades,
                "drawdown_trigger": drawdown_trigger,
                "regime_change_trigger": regime_change_trigger,
                "timestamp": datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "scheduler_init_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    def should_reoptimize(
        self,
        trade_count: int,
        regime_changed: bool = False,
        current_drawdown: float = 0.0
    ) -> bool:
        """
        Determine if reoptimization should be triggered.
        
        Args:
            trade_count: Total number of trades executed
            regime_changed: Whether market regime has changed
            current_drawdown: Current drawdown percentage
            
        Returns:
            True if reoptimization should occur, False otherwise
        """
        try:
            with self.lock:
                self.trade_count = trade_count
                
                # Trigger 1: Fixed number of trades
                trades_since_last = trade_count - self.last_optimization_trade_count
                if trades_since_last >= self.reoptimize_every_n_trades:
                    logger.info(json.dumps({
                        "event": "reoptimization_triggered",
                        "trigger": "trade_count",
                        "trades_since_last": trades_since_last,
                        "current_trade_count": trade_count,
                        "timestamp": datetime.now().isoformat()
                    }))
                    self.last_optimization_trade_count = trade_count
                    return True
                
                # Trigger 2: Drawdown threshold
                if current_drawdown < self.drawdown_trigger:
                    logger.warning(json.dumps({
                        "event": "reoptimization_triggered",
                        "trigger": "drawdown_threshold",
                        "current_drawdown": current_drawdown,
                        "threshold": self.drawdown_trigger,
                        "timestamp": datetime.now().isoformat()
                    }))
                    self.last_optimization_trade_count = trade_count
                    return True
                
                # Trigger 3: Regime change
                if regime_changed and self.regime_change_trigger:
                    logger.info(json.dumps({
                        "event": "reoptimization_triggered",
                        "trigger": "regime_change",
                        "timestamp": datetime.now().isoformat()
                    }))
                    self.last_optimization_trade_count = trade_count
                    return True
                
                return False
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "should_reoptimize_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "parameters": {
                    "trade_count": trade_count,
                    "regime_changed": regime_changed,
                    "current_drawdown": current_drawdown
                }
            }))
            return False

    def reset(self) -> None:
        """Reset scheduler state."""
        try:
            with self.lock:
                self.trade_count = 0
                self.last_optimization_trade_count = 0
                
                logger.info(json.dumps({
                    "event": "adaptive_learning_scheduler_reset",
                    "timestamp": datetime.now().isoformat()
                }))
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "scheduler_reset_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_default_optimizer() -> ExecutionOptimizer:
    """
    Create ExecutionOptimizer with default parameters and configuration.
    
    Returns:
        Configured ExecutionOptimizer instance
    """
    try:
        default_params = ExecutionParams(
            fill_model="mid_price",
            signal_selector="nearest_delta",
            delta_target_call=-0.30,
            delta_target_put=-0.30,
            position_size_pct=5.0,
            slippage_pct=0.5,
            commission_per_contract=20.0,
            stop_loss_pct=-2.0,
            profit_target_pct=5.0,
            min_dte=7,
            max_dte=45,
            volume_threshold=100.0
        )
        
        return ExecutionOptimizer(initial_params=default_params)
        
    except Exception as e:
        logger.error(json.dumps({
            "event": "create_default_optimizer_error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of ExecutionOptimizer and AdaptiveLearningScheduler.
    """
    try:
        # Create optimizer
        optimizer = create_default_optimizer()
        
        # Example backtest update
        logger.info("Starting example optimization workflow...")
        
        # Simulate backtest results improving
        new_params = optimizer.update_from_backtest(
            sharpe=1.5,
            max_dd=-3.0,
            win_rate=0.58,
            avg_fill_quality=0.85,
            avg_hold_days=2.5,
            total_pnl=15000.0
        )
        
        logger.info(json.dumps({
            "event": "optimization_example",
            "updated_params": new_params.to_dict()
        }))
        
        # Save state
        optimizer.save_state("optimizer_state.json")
        
        # Create scheduler
        scheduler = AdaptiveLearningScheduler(
            reoptimize_every_n_trades=50,
            drawdown_trigger=-5.0
        )
        
        # Test scheduler
        should_reopt = scheduler.should_reoptimize(
            trade_count=55,
            regime_changed=False,
            current_drawdown=-2.0
        )
        
        logger.info(json.dumps({
            "event": "scheduler_example",
            "should_reoptimize": should_reopt
        }))
        
        logger.info("Example workflow completed successfully!")
        
    except Exception as e:
        logger.error(json.dumps({
            "event": "main_execution_error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)
