"""
NSE Beast System - AI Trader with Backtesting Integration
===========================================================

This module implements an AI-driven options trader that:
1. Generates trading signals based on VRP and regime detection
2. Validates signals through backtesting before live execution
3. Learns from backtest results to improve execution parameters
4. Manages Greeks and risk dynamically
5. Executes multi-leg strategies via Zerodha KiteConnect

Integrates with:
- lambdaclass/options_portfolio_backtester for strategy validation
- KiteBrokerAdapter for live execution
- ExecutionOptimizer for order optimization
"""

import asyncio
import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, time as dt_time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
import threading
import queue

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Import backtester components
from options_portfolio_backtester.engine import BacktestEngine
from options_portfolio_backtester.strategy import Strategy, StrategyLeg, Strangle, IronCondor, Butterfly
from options_portfolio_backtester.signal_selectors import NearestDelta, MaxOpenInterest
from options_portfolio_backtester.cost_models import SpreadSlippage, PerContractCommission
from options_portfolio_backtester.fill_models import MidPrice, VolumeAwareFill, MarketAtBidAsk
from options_portfolio_backtester.sizers import PercentOfPortfolio, CapitalBased
from options_portfolio_backtester.risk_managers import MaxDelta, MaxVega, MaxDrawdown

# Import Beast System components
from nse_beast_system.adapters.kite_broker_adapter import KiteBrokerAdapter
from nse_beast_system.vrp.vrp_calculator import VRPCalculator
from nse_beast_system.regime.regime_detector import RegimeDetector
from nse_beast_system.positions.position_manager import PositionManager


# ============================================================================
# ENUMS
# ============================================================================

class SignalType(Enum):
    """AI trading signal types"""
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    COVERED_CALL = "covered_call"
    NO_TRADE = "no_trade"


class BacktestStatus(Enum):
    """Backtest execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AITraderConfig:
    """Configuration for AI Trader"""
    
    # Model parameters
    min_vrp_score: float = 0.60
    min_regime_confidence: float = 0.65
    delta_target_short: float = -0.30
    delta_target_long: float = 0.30
    
    # Risk parameters
    max_portfolio_delta: float = 0.50
    max_portfolio_vega: float = 5000.0
    max_portfolio_gamma: float = 0.10
    max_drawdown_pct: float = 10.0
    daily_loss_limit: float = -5000.0
    
    # Position sizing
    position_size_pct: float = 5.0
    initial_capital: float = 100000.0
    max_positions: int = 5
    
    # Backtest parameters
    backtest_lookback_days: int = 20
    backtest_timeout_seconds: int = 60
    min_backtest_sharpe: float = 0.5
    min_backtest_win_rate: float = 0.50
    max_backtest_drawdown: float = 15.0
    
    # Fill model parameters
    bid_ask_spread_pct: float = 0.01
    volume_threshold_contracts: int = 500
    slippage_pct: float = 0.02
    commission_per_contract: float = 20.0  # Rupees
    
    # Learning parameters
    learning_rate: float = 0.1
    delta_adjustment_step: float = 0.05
    sizer_adjustment_step: float = 0.01
    
    # Execution parameters
    order_timeout_seconds: int = 30
    max_order_retries: int = 3
    retry_delay_ms: int = 500
    
    # Monitoring
    enable_live_backtesting: bool = True
    backtest_frequency_minutes: int = 60
    log_level: str = "INFO"
    log_dir: str = "logs"
    
    # Safety
    enable_kill_switch: bool = True
    kill_switch_loss_threshold: float = -10000.0


@dataclass
class StrategyLegConfig:
    """Configuration for a single strategy leg"""
    option_type: str  # "CE" or "PE"
    strike_offset: int  # Offset from current price
    side: str  # "BUY" or "SELL"
    quantity: int = 1
    delta_target: Optional[float] = None
    greeks: Dict[str, float] = field(default_factory=dict)


@dataclass
class TradeDecision:
    """AI decision for trade execution"""
    signal_type: SignalType
    confidence: float  # 0-1
    expected_pnl: float
    expected_pnl_pct: float
    greeks: Dict[str, float]  # delta, gamma, vega, theta
    legs: List[StrategyLegConfig]
    backtest_result: Optional['BacktestResult'] = None
    rationale: str = ""
    timestamp: str = ""
    signal_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "signal_type": self.signal_type.value,
            "confidence": self.confidence,
            "expected_pnl": self.expected_pnl,
            "expected_pnl_pct": self.expected_pnl_pct,
            "greeks": self.greeks,
            "legs": [asdict(leg) for leg in self.legs],
            "backtest_result": self.backtest_result.to_dict() if self.backtest_result else None,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
            "signal_id": self.signal_id,
            "metadata": self.metadata,
        }


@dataclass
class BacktestResult:
    """Results from backtest validation"""
    status: BacktestStatus
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    avg_fill_quality: float  # 0-1, how close to mid price
    total_pnl: float
    num_trades: int
    avg_trade_duration_hours: float
    trade_dataframe: Optional[pd.DataFrame] = None
    error_message: str = ""
    execution_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "avg_fill_quality": self.avg_fill_quality,
            "total_pnl": self.total_pnl,
            "num_trades": self.num_trades,
            "avg_trade_duration_hours": self.avg_trade_duration_hours,
            "error_message": self.error_message,
            "execution_time_seconds": self.execution_time_seconds,
        }
    
    def is_acceptable(self, config: AITraderConfig) -> bool:
        """Check if backtest result meets minimum criteria"""
        checks = [
            self.status == BacktestStatus.COMPLETED,
            self.sharpe_ratio >= config.min_backtest_sharpe,
            self.win_rate >= config.min_backtest_win_rate,
            self.max_drawdown_pct <= config.max_backtest_drawdown,
        ]
        return all(checks)


@dataclass
class LearningUpdate:
    """Learning parameters from backtest results"""
    delta_adjustment: float = 0.0
    sizer_adjustment: float = 0.0
    fill_model_update: Dict[str, float] = field(default_factory=dict)
    commission_estimate: float = 0.0
    confidence_boost: float = 0.0


@dataclass
class ExecutionOptimizer:
    """Execution optimization parameters"""
    target_delta: float = -0.30
    position_size_contracts: int = 1
    bid_ask_spread_pct: float = 0.01
    slippage_estimate_pct: float = 0.02
    commission_per_contract: float = 20.0
    fill_model_type: str = "mid_price"  # mid_price, volume_aware, market_at_bid_ask


# ============================================================================
# JSON LOGGER SETUP
# ============================================================================

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


def setup_ai_logging(log_dir: str = "logs") -> Tuple[logging.Logger, logging.Logger]:
    """Setup structured JSON logging for AI Trader"""
    Path(log_dir).mkdir(exist_ok=True)

    # Main logger
    logger = logging.getLogger("AITrader")
    logger.setLevel(logging.DEBUG)

    # File handler for JSON logs
    json_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "ai_trader.json"),
        maxBytes=100 * 1024 * 1024,
        backupCount=10,
    )
    json_handler.setFormatter(JSONFormatter())
    logger.addHandler(json_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Decision logger (separate file for decisions)
    decision_logger = logging.getLogger("AIDecisions")
    decision_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "ai_decisions.json"),
        maxBytes=100 * 1024 * 1024,
        backupCount=10,
    )
    decision_handler.setFormatter(JSONFormatter())
    decision_logger.addHandler(decision_handler)
    decision_logger.setLevel(logging.INFO)

    return logger, decision_logger


# ============================================================================
# AI TRADER MAIN CLASS
# ============================================================================

class AITrader:
    """
    AI-driven options trader with backtesting validation
    
    Workflow:
    1. Generate signal based on VRP + regime
    2. Select strategy (strangle, iron condor, etc.)
    3. Build backtest engine for validation
    4. Run backtest to validate signal
    5. If backtest passes, execute live trade
    6. Learn from results to improve future parameters
    """

    def __init__(
        self,
        config: AITraderConfig,
        broker_adapter: KiteBrokerAdapter,
        vrp_calculator: VRPCalculator,
        regime_detector: RegimeDetector,
        position_manager: PositionManager,
        execution_optimizer: Optional[ExecutionOptimizer] = None,
        log_dir: str = "logs",
    ):
        """
        Initialize AI Trader

        Args:
            config: AITraderConfig with all parameters
            broker_adapter: KiteBrokerAdapter for live trading
            vrp_calculator: VRPCalculator for volatility analysis
            regime_detector: RegimeDetector for market state
            position_manager: PositionManager for portfolio tracking
            execution_optimizer: Optional executor optimizer
            log_dir: Directory for logging
        """
        self.config = config
        self.broker_adapter = broker_adapter
        self.vrp_calculator = vrp_calculator
        self.regime_detector = regime_detector
        self.position_manager = position_manager
        self.execution_optimizer = execution_optimizer or ExecutionOptimizer()
        
        # Setup logging
        self.logger, self.decision_logger = setup_ai_logging(log_dir)
        self.logger.info("Initializing AI Trader")

        # State management
        self.running = False
        self.kill_switch = asyncio.Event()
        self.backtest_in_progress = False

        # Learning parameters (start with config values)
        self.current_delta_target_short = config.delta_target_short
        self.current_delta_target_long = config.delta_target_long
        self.current_position_size_pct = config.position_size_pct
        self.current_bid_ask_spread = config.bid_ask_spread_pct
        self.current_commission = config.commission_per_contract

        # History for learning
        self.backtest_history: deque = deque(maxlen=100)
        self.execution_history: deque = deque(maxlen=100)
        self.decision_history: deque = deque(maxlen=100)

        # Performance tracking
        self.total_decisions = 0
        self.accepted_decisions = 0
        self.rejected_by_backtest = 0
        self.executed_trades = 0
        self.successful_trades = 0

        # Tasks
        self.tasks: List[asyncio.Task] = []

        self.logger.info("AI Trader initialization complete")

    async def run_trading_loop(self) -> None:
        """
        Main trading loop
        
        Continuously:
        1. Generate signals
        2. Validate via backtest
        3. Execute if validated
        4. Learn from results
        """
        market_open = dt_time(9, 15)
        market_close = dt_time(15, 30)
        
        self.running = True
        self.logger.info("AI Trading loop started")

        try:
            while not self.kill_switch.is_set():
                try:
                    current_time = datetime.now().time()

                    # Check market hours
                    if not (market_open <= current_time <= market_close):
                        await asyncio.sleep(60)
                        continue

                    loop_start = time.time()

                    # Step 1: Generate signal
                    signal = await self._generate_signal()

                    if signal is None:
                        await asyncio.sleep(5)
                        continue

                    self.total_decisions += 1

                    # Step 2: Select strategy
                    decision = await self._select_strategy(signal)

                    if decision.signal_type == SignalType.NO_TRADE:
                        self.logger.debug("No trade signal selected")
                        await asyncio.sleep(5)
                        continue

                    # Step 3: Build backtest engine
                    backtest_engine = await self._build_backtest_engine(decision)

                    # Step 4: Run backtest validation
                    backtest_result = await self._run_backtest_validation(
                        decision, backtest_engine
                    )
                    decision.backtest_result = backtest_result

                    # Step 5: Decide on execution
                    if backtest_result.is_acceptable(self.config):
                        self.accepted_decisions += 1
                        
                        # Log decision
                        self._log_trade_decision(decision)
                        
                        # Execute trade
                        trade_id = await self._execute_trade(decision)
                        
                        if trade_id:
                            self.executed_trades += 1
                            self.decision_history.append({
                                "timestamp": datetime.utcnow().isoformat(),
                                "trade_id": trade_id,
                                "decision": decision.to_dict(),
                            })
                    else:
                        self.rejected_by_backtest += 1
                        self.logger.info(
                            f"Trade rejected by backtest: "
                            f"sharpe={backtest_result.sharpe_ratio:.2f}, "
                            f"wr={backtest_result.win_rate:.2f}",
                            extra={
                                "extra_data": {
                                    "reason": "backtest_criteria_not_met",
                                    "result": backtest_result.to_dict(),
                                }
                            },
                        )

                    # Step 6: Learn from backtest
                    await self._learn_from_backtest(backtest_result)

                    loop_time = time.time() - loop_start
                    if loop_time > 5.0:
                        self.logger.warning(
                            f"Trading loop slow: {loop_time:.2f}s",
                            extra={"extra_data": {"loop_time_ms": loop_time * 1000}},
                        )

                    await asyncio.sleep(1)

                except Exception as e:
                    self.logger.error(
                        f"Error in trading loop: {e}",
                        exc_info=True,
                    )
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            self.logger.info("Trading loop cancelled")
        finally:
            self.running = False

    async def _generate_signal(self) -> Optional[Dict]:
        """
        Generate trading signal based on VRP and regime
        
        Returns:
            Signal dict with vrp_score, regime, confidence, or None
        """
        try:
            # Get VRP score
            vrp_score = await self.vrp_calculator.calculate_vrp_score()
            if vrp_score is None or vrp_score < self.config.min_vrp_score:
                self.logger.debug(f"VRP score too low: {vrp_score}")
                return None

            # Get market regime
            regime = await self.regime_detector.detect_regime()
            if regime is None:
                return None

            regime_confidence = regime.get("confidence", 0.0)
            if regime_confidence < self.config.min_regime_confidence:
                self.logger.debug(f"Regime confidence too low: {regime_confidence}")
                return None

            signal = {
                "vrp_score": vrp_score,
                "regime": regime.get("regime", "ranging"),
                "regime_confidence": regime_confidence,
                "timestamp": datetime.utcnow().isoformat(),
                "signal_id": f"SIG_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}",
            }

            self.logger.debug(
                "Signal generated",
                extra={"extra_data": signal},
            )

            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal: {e}", exc_info=True)
            return None

    async def _select_strategy(self, signal: Dict) -> TradeDecision:
        """
        Select trading strategy based on signal
        
        Args:
            signal: Signal dict from _generate_signal
            
        Returns:
            TradeDecision with strategy and legs
        """
        try:
            vrp_score = signal["vrp_score"]
            regime = signal["regime"]
            confidence = signal["regime_confidence"] * vrp_score

            # Strategy selection logic
            if regime == "ranging" and vrp_score > 0.70:
                # Iron Condor for range-bound with high VRP
                strategy_type = SignalType.IRON_CONDOR
                legs = [
                    StrategyLegConfig(
                        option_type="CE",
                        strike_offset=200,
                        side="SELL",
                        delta_target=-0.20,
                    ),
                    StrategyLegConfig(
                        option_type="CE",
                        strike_offset=300,
                        side="BUY",
                        delta_target=-0.10,
                    ),
                    StrategyLegConfig(
                        option_type="PE",
                        strike_offset=-200,
                        side="SELL",
                        delta_target=0.20,
                    ),
                    StrategyLegConfig(
                        option_type="PE",
                        strike_offset=-300,
                        side="BUY",
                        delta_target=0.10,
                    ),
                ]
            elif regime == "ranging" and vrp_score > 0.60:
                # Short Strangle for range-bound
                strategy_type = SignalType.SHORT_STRANGLE
                legs = [
                    StrategyLegConfig(
                        option_type="CE",
                        strike_offset=150,
                        side="SELL",
                        delta_target=self.current_delta_target_short,
                    ),
                    StrategyLegConfig(
                        option_type="PE",
                        strike_offset=-150,
                        side="SELL",
                        delta_target=-self.current_delta_target_short,
                    ),
                ]
            elif regime == "trending_up":
                # Covered call for uptrend
                strategy_type = SignalType.COVERED_CALL
                legs = [
                    StrategyLegConfig(
                        option_type="CE",
                        strike_offset=100,
                        side="SELL",
                        delta_target=self.current_delta_target_short,
                    ),
                ]
            elif regime == "trending_down":
                # Long strangle for downtrend
                strategy_type = SignalType.LONG_STRANGLE
                legs = [
                    StrategyLegConfig(
                        option_type="PE",
                        strike_offset=-150,
                        side="BUY",
                        delta_target=-0.20,
                    ),
                    StrategyLegConfig(
                        option_type="CE",
                        strike_offset=150,
                        side="BUY",
                        delta_target=0.20,
                    ),
                ]
            else:
                # No trade
                return TradeDecision(
                    signal_type=SignalType.NO_TRADE,
                    confidence=0.0,
                    expected_pnl=0.0,
                    expected_pnl_pct=0.0,
                    greeks={},
                    legs=[],
                    rationale="No matching regime for strategy",
                    timestamp=signal["timestamp"],
                    signal_id=signal["signal_id"],
                )

            # Calculate expected Greeks
            expected_greeks = await self._calculate_expected_greeks(legs)

            # Calculate expected P&L (rough estimate)
            expected_pnl = vrp_score * 1000  # Simplified
            expected_pnl_pct = (expected_pnl / self.config.initial_capital) * 100

            decision = TradeDecision(
                signal_type=strategy_type,
                confidence=min(confidence, 1.0),
                expected_pnl=expected_pnl,
                expected_pnl_pct=expected_pnl_pct,
                greeks=expected_greeks,
                legs=legs,
                rationale=f"Selected {strategy_type.value} for {regime} market with VRP={vrp_score:.2f}",
                timestamp=signal["timestamp"],
                signal_id=signal["signal_id"],
            )

            self.logger.info(
                f"Strategy selected: {strategy_type.value}",
                extra={"extra_data": decision.to_dict()},
            )

            return decision

        except Exception as e:
            self.logger.error(f"Error selecting strategy: {e}", exc_info=True)
            return TradeDecision(
                signal_type=SignalType.NO_TRADE,
                confidence=0.0,
                expected_pnl=0.0,
                expected_pnl_pct=0.0,
                greeks={},
                legs=[],
                rationale=f"Error in strategy selection: {e}",
                timestamp=signal.get("timestamp", datetime.utcnow().isoformat()),
                signal_id=signal.get("signal_id", ""),
            )

    async def _calculate_expected_greeks(self, legs: List[StrategyLegConfig]) -> Dict[str, float]:
        """
        Calculate expected Greeks for a strategy
        
        Args:
            legs: List of strategy legs
            
        Returns:
            Dict with delta, gamma, vega, theta
        """
        try:
            total_delta = 0.0
            total_gamma = 0.0
            total_vega = 0.0
            total_theta = 0.0

            for leg in legs:
                # Simplified Greeks calculation
                # In real system, would use actual option pricing model
                if leg.delta_target:
                    multiplier = -1 if leg.side == "SELL" else 1
                    total_delta += leg.delta_target * multiplier * leg.quantity
                    total_gamma += 0.02 * multiplier * leg.quantity
                    total_vega += 50 * multiplier * leg.quantity
                    total_theta += 0.5 * multiplier * leg.quantity

            return {
                "delta": round(total_delta, 3),
                "gamma": round(total_gamma, 3),
                "vega": round(total_vega, 2),
                "theta": round(total_theta, 2),
            }

        except Exception as e:
            self.logger.error(f"Error calculating Greeks: {e}")
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    async def _build_backtest_engine(self, decision: TradeDecision) -> BacktestEngine:
        """
        Build backtest engine for strategy validation
        
        Args:
            decision: TradeDecision with strategy and legs
            
        Returns:
            Configured BacktestEngine
        """
        try:
            self.logger.debug("Building backtest engine")

            # Create strategy based on signal type
            if decision.signal_type == SignalType.IRON_CONDOR:
                strategy = IronCondor()
            elif decision.signal_type == SignalType.SHORT_STRANGLE:
                strategy = Strangle(is_long=False)
            elif decision.signal_type == SignalType.LONG_STRANGLE:
                strategy = Strangle(is_long=True)
            elif decision.signal_type == SignalType.BUTTERFLY:
                strategy = Butterfly()
            else:
                raise ValueError(f"Unknown strategy type: {decision.signal_type}")

            # Configure cost model (commission)
            cost_model = PerContractCommission(
                rate=self.current_commission / 100.0  # Convert to percentage
            )

            # Configure fill model
            if self.execution_optimizer.fill_model_type == "volume_aware":
                fill_model = VolumeAwareFill(
                    threshold=self.execution_optimizer.bid_ask_spread_pct
                )
            elif self.execution_optimizer.fill_model_type == "market_at_bid_ask":
                fill_model = MarketAtBidAsk()
            else:
                fill_model = MidPrice()

            # Configure position sizer
            sizer = PercentOfPortfolio(
                pct=self.current_position_size_pct / 100.0
            )

            # Configure signal selector
            signal_selector = NearestDelta(
                target=self.execution_optimizer.target_delta
            )

            # Configure risk managers
            risk_managers = [
                MaxDelta(max_delta=self.config.max_portfolio_delta),
                MaxVega(max_vega=self.config.max_portfolio_vega),
                MaxDrawdown(max_drawdown_pct=self.config.max_backtest_drawdown),
            ]

            # Create backtest engine
            engine = BacktestEngine(
                strategy=strategy,
                cost_model=cost_model,
                fill_model=fill_model,
                sizer=sizer,
                signal_selector=signal_selector,
                risk_managers=risk_managers,
            )

            self.logger.debug("Backtest engine created successfully")
            return engine

        except Exception as e:
            self.logger.error(f"Error building backtest engine: {e}", exc_info=True)
            raise

    async def _run_backtest_validation(
        self,
        decision: TradeDecision,
        engine: BacktestEngine,
    ) -> BacktestResult:
        """
        Run backtest to validate signal
        
        Args:
            decision: TradeDecision to validate
            engine: BacktestEngine to use
            
        Returns:
            BacktestResult with metrics
        """
        try:
            self.backtest_in_progress = True
            backtest_start = time.time()
            
            self.logger.info(
                f"Starting backtest validation for {decision.signal_type.value}",
                extra={"extra_data": {"signal_id": decision.signal_id}},
            )

            # Run backtest with timeout
            try:
                loop = asyncio.get_event_loop()
                trade_dataframe = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: engine.run(check_exits_daily=True),
                    ),
                    timeout=self.config.backtest_timeout_seconds,
                )
            except asyncio.TimeoutError:
                self.logger.error("Backtest timeout")
                return BacktestResult(
                    status=BacktestStatus.TIMEOUT,
                    sharpe_ratio=0.0,
                    max_drawdown_pct=0.0,
                    win_rate=0.0,
                    avg_fill_quality=0.0,
                    total_pnl=0.0,
                    num_trades=0,
                    avg_trade_duration_hours=0.0,
                    error_message="Backtest timeout",
                    execution_time_seconds=time.time() - backtest_start,
                )

            execution_time = time.time() - backtest_start

            # Parse results
            if trade_dataframe is None or trade_dataframe.empty:
                return BacktestResult(
                    status=BacktestStatus.COMPLETED,
                    sharpe_ratio=0.0,
                    max_drawdown_pct=0.0,
                    win_rate=0.0,
                    avg_fill_quality=0.0,
                    total_pnl=0.0,
                    num_trades=0,
                    avg_trade_duration_hours=0.0,
                    error_message="No trades generated",
                    execution_time_seconds=execution_time,
                    trade_dataframe=trade_dataframe,
                )

            # Calculate metrics
            results = self._extract_backtest_metrics(trade_dataframe)
            results.execution_time_seconds = execution_time
            results.trade_dataframe = trade_dataframe

            self.logger.info(
                f"Backtest completed: sharpe={results.sharpe_ratio:.2f}, "
                f"wr={results.win_rate:.2f}, drawdown={results.max_drawdown_pct:.2f}%",
                extra={"extra_data": results.to_dict()},
            )

            self.backtest_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "signal_id": decision.signal_id,
                "result": results.to_dict(),
            })

            return results

        except Exception as e:
            self.logger.error(f"Error running backtest: {e}", exc_info=True)
            return BacktestResult(
                status=BacktestStatus.FAILED,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
                avg_fill_quality=0.0,
                total_pnl=0.0,
                num_trades=0,
                avg_trade_duration_hours=0.0,
                error_message=str(e),
                execution_time_seconds=time.time() - backtest_start,
            )

        finally:
            self.backtest_in_progress = False

    def _extract_backtest_metrics(self, df: pd.DataFrame) -> BacktestResult:
        """
        Extract metrics from backtest results DataFrame
        
        Args:
            df: Backtest results DataFrame
            
        Returns:
            BacktestResult with calculated metrics
        """
        try:
            # Calculate P&L metrics
            total_pnl = df["pnl"].sum() if "pnl" in df.columns else 0.0
            num_trades = len(df)
            
            if num_trades == 0:
                return BacktestResult(
                    status=BacktestStatus.COMPLETED,
                    sharpe_ratio=0.0,
                    max_drawdown_pct=0.0,
                    win_rate=0.0,
                    avg_fill_quality=0.0,
                    total_pnl=0.0,
                    num_trades=0,
                    avg_trade_duration_hours=0.0,
                )

            # Win rate
            winning_trades = len(df[df["pnl"] > 0]) if "pnl" in df.columns else 0
            win_rate = winning_trades / num_trades if num_trades > 0 else 0.0

            # Sharpe ratio (simplified)
            if "pnl" in df.columns and len(df) > 1:
                returns = df["pnl"].pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252)
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0

            # Max drawdown (simplified)
            if "cumulative_pnl" in df.columns:
                cumulative = df["cumulative_pnl"]
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = drawdown.min() * 100
            else:
                max_drawdown = 0.0

            # Average fill quality (0-1, higher is better)
            if "fill_quality" in df.columns:
                avg_fill_quality = df["fill_quality"].mean()
            else:
                avg_fill_quality = 0.95  # Assume good fills

            # Average trade duration
            if "duration_hours" in df.columns:
                avg_duration = df["duration_hours"].mean()
            else:
                avg_duration = 1.0

            return BacktestResult(
                status=BacktestStatus.COMPLETED,
                sharpe_ratio=sharpe,
                max_drawdown_pct=abs(max_drawdown),
                win_rate=win_rate,
                avg_fill_quality=avg_fill_quality,
                total_pnl=total_pnl,
                num_trades=num_trades,
                avg_trade_duration_hours=avg_duration,
            )

        except Exception as e:
            self.logger.error(f"Error extracting backtest metrics: {e}")
            return BacktestResult(
                status=BacktestStatus.FAILED,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                win_rate=0.0,
                avg_fill_quality=0.0,
                total_pnl=0.0,
                num_trades=0,
                avg_trade_duration_hours=0.0,
                error_message=str(e),
            )

    async def _execute_trade(self, decision: TradeDecision) -> Optional[str]:
        """
        Execute trade via KiteConnect
        
        Args:
            decision: TradeDecision to execute
            
        Returns:
            Trade ID if successful, None otherwise
        """
        trade_id = f"TRADE_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

        try:
            self.logger.info(
                f"Executing trade: {decision.signal_type.value}",
                extra={"extra_data": {"trade_id": trade_id}},
            )

            order_ids = []

            for i, leg in enumerate(decision.legs):
                try:
                    # Get current symbol (simplified - would need actual symbol resolution)
                    symbol = f"NIFTY23JUN{abs(leg.strike_offset)}{'CE' if leg.option_type == 'CE' else 'PE'}"

                    # Place order
                    order_id = await self.broker_adapter.place_order(
                        symbol=symbol,
                        side=leg.side,
                        quantity=leg.quantity,
                        order_type="MARKET",
                    )

                    order_ids.append(order_id)
                    self.logger.info(f"Order placed: {order_id} for leg {i+1}")

                except Exception as e:
                    self.logger.error(
                        f"Failed to place order for leg {i+1}: {e}",
                        exc_info=True,
                    )
                    # Rollback previous orders
                    for placed_order_id in order_ids:
                        try:
                            await self.broker_adapter.cancel_order(placed_order_id)
                        except Exception as cancel_err:
                            self.logger.error(f"Failed to cancel order: {cancel_err}")
                    return None

            # Log execution
            self.execution_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "trade_id": trade_id,
                "signal_type": decision.signal_type.value,
                "order_ids": order_ids,
                "greeks": decision.greeks,
            })

            self.logger.info(
                f"Trade executed successfully: {trade_id}",
                extra={
                    "extra_data": {
                        "trade_id": trade_id,
                        "num_legs": len(order_ids),
                        "orders": order_ids,
                    }
                },
            )

            return trade_id

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}", exc_info=True)
            return None

    async def _learn_from_backtest(self, result: BacktestResult) -> None:
        """
        Learn from backtest results to improve future parameters
        
        Args:
            result: BacktestResult to learn from
        """
        try:
            if result.status != BacktestStatus.COMPLETED:
                return

            learning_update = LearningUpdate()

            # Adjust delta target if drawdown is high
            if result.max_drawdown_pct > self.config.max_backtest_drawdown * 0.8:
                learning_update.delta_adjustment = -self.config.delta_adjustment_step
                self.current_delta_target_short += learning_update.delta_adjustment
                self.logger.info(
                    f"Adjusted delta target to {self.current_delta_target_short:.3f}",
                    extra={"extra_data": learning_update.__dict__},
                )

            # Adjust position size if Sharpe is low
            if result.sharpe_ratio < self.config.min_backtest_sharpe:
                learning_update.sizer_adjustment = -self.config.sizer_adjustment_step
                self.current_position_size_pct += learning_update.sizer_adjustment
                self.logger.info(
                    f"Reduced position size to {self.current_position_size_pct:.2f}%",
                    extra={"extra_data": learning_update.__dict__},
                )

            # Update fill model based on fill quality
            if result.avg_fill_quality < 0.95:
                learning_update.fill_model_update = {
                    "bid_ask_spread": self.current_bid_ask_spread * 1.1,
                    "slippage": self.execution_optimizer.slippage_estimate_pct * 1.1,
                }
                self.current_bid_ask_spread = learning_update.fill_model_update.get(
                    "bid_ask_spread", self.current_bid_ask_spread
                )

            # Boost confidence if results are good
            if result.sharpe_ratio > 1.0 and result.win_rate > 0.60:
                learning_update.confidence_boost = 0.05

            self.logger.debug(
                "Learning update applied",
                extra={"extra_data": learning_update.__dict__},
            )

        except Exception as e:
            self.logger.error(f"Error learning from backtest: {e}", exc_info=True)

    async def _compute_greeks(self, positions: List[Dict]) -> Dict[str, float]:
        """
        Compute portfolio Greeks from positions
        
        Args:
            positions: List of position dicts
            
        Returns:
            Dict with portfolio delta, gamma, vega, theta
        """
        try:
            total_delta = 0.0
            total_gamma = 0.0
            total_vega = 0.0
            total_theta = 0.0

            for pos in positions:
                delta = pos.get("delta", 0.0)
                gamma = pos.get("gamma", 0.0)
                vega = pos.get("vega", 0.0)
                theta = pos.get("theta", 0.0)
                quantity = pos.get("quantity", 0)

                total_delta += delta * quantity
                total_gamma += gamma * quantity
                total_vega += vega * quantity
                total_theta += theta * quantity

            return {
                "delta": round(total_delta, 3),
                "gamma": round(total_gamma, 3),
                "vega": round(total_vega, 2),
                "theta": round(total_theta, 2),
            }

        except Exception as e:
            self.logger.error(f"Error computing Greeks: {e}")
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    async def _update_execution_params(self, result: BacktestResult) -> None:
        """
        Update execution parameters based on backtest results
        
        Args:
            result: BacktestResult to learn from
        """
        try:
            # Update fill model based on fill quality
            if result.avg_fill_quality < 0.90:
                # Worse than expected, increase spread estimate
                self.execution_optimizer.bid_ask_spread_pct *= 1.1
                self.execution_optimizer.slippage_estimate_pct *= 1.1

            # Update position size based on performance
            if result.sharpe_ratio > 1.5:
                # Good performance, can increase size
                if self.current_position_size_pct < self.config.position_size_pct:
                    self.current_position_size_pct += 0.5

            # Update commission estimate
            if "commission" in result.to_dict():
                estimated_commission = result.to_dict().get("commission", 0)
                if estimated_commission > 0:
                    self.current_commission = estimated_commission

            self.logger.debug(
                "Execution parameters updated",
                extra={
                    "extra_data": {
                        "bid_ask_spread": self.execution_optimizer.bid_ask_spread_pct,
                        "position_size_pct": self.current_position_size_pct,
                        "commission": self.current_commission,
                    }
                },
            )

        except Exception as e:
            self.logger.error(f"Error updating execution params: {e}")

    def _log_trade_decision(self, decision: TradeDecision) -> None:
        """Log trade decision for analysis"""
        try:
            self.decision_logger.info(
                f"Trade Decision: {decision.signal_type.value}",
                extra={"extra_data": decision.to_dict()},
            )
        except Exception as e:
            self.logger.error(f"Error logging decision: {e}")

    def trigger_kill_switch(self) -> None:
        """Trigger emergency shutdown"""
        self.logger.critical("KILL SWITCH ACTIVATED")
        self.kill_switch.set()

    async def shutdown(self) -> None:
        """Graceful shutdown"""
        try:
            self.logger.info("Shutting down AI Trader")
            self.running = False
            self.kill_switch.set()

            # Cancel all tasks
            for task in self.tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Log final statistics
            self.logger.info(
                "AI Trader shutdown complete",
                extra={
                    "extra_data": {
                        "total_decisions": self.total_decisions,
                        "accepted_decisions": self.accepted_decisions,
                        "rejected_by_backtest": self.rejected_by_backtest,
                        "executed_trades": self.executed_trades,
                        "successful_trades": self.successful_trades,
                        "acceptance_rate": (
                            self.accepted_decisions / self.total_decisions * 100
                            if self.total_decisions > 0
                            else 0.0
                        ),
                    }
                },
            )

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)

    def get_status(self) -> Dict:
        """Get AI Trader status"""
        return {
            "running": self.running,
            "total_decisions": self.total_decisions,
            "accepted_decisions": self.accepted_decisions,
            "rejected_by_backtest": self.rejected_by_backtest,
            "executed_trades": self.executed_trades,
            "successful_trades": self.successful_trades,
            "acceptance_rate": (
                self.accepted_decisions / self.total_decisions * 100
                if self.total_decisions > 0
                else 0.0
            ),
            "current_delta_target_short": self.current_delta_target_short,
            "current_position_size_pct": self.current_position_size_pct,
            "backtest_in_progress": self.backtest_in_progress,
            "recent_backtest_results": [
                r for r in list(self.backtest_history)[-10:]
            ],
        }

    def get_decision_history(self, limit: int = 100) -> List[Dict]:
        """Get recent decision history"""
        return list(self.decision_history)[-limit:]

    def get_execution_history(self, limit: int = 100) -> List[Dict]:
        """Get recent execution history"""
        return list(self.execution_history)[-limit:]


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main() -> None:
    """Main entry point for AI Trader"""
    load_dotenv()

    try:
        # Create config
        config = AITraderConfig(
            min_vrp_score=0.60,
            min_regime_confidence=0.65,
            max_positions=5,
            backtest_lookback_days=20,
        )

        # Initialize adapters (these would come from BeastEngine in real setup)
        # For now, using placeholder initialization
        logger = logging.getLogger("AITrader")
        logger.info("AI Trader initialized")

        # TODO: Wire up with actual broker adapter, calculators, etc.
        # This would typically be called from BeastEngine.initialize_modules()

    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
