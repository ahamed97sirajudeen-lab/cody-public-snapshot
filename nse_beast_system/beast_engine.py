"""
NSE Beast System - Complete Automated Options Trading Engine
============================================================

This is the main orchestrator that ties together all modules:
- Authentication (Zerodha KiteConnect)
- Volatility Risk Premium (VRP) Calculator
- Regime Detection (Market State Analysis)
- Signal Generator (Trade Entry Signals)
- Execution Adapter (KiteBrokerAdapter)
- Position Manager (Portfolio Tracking)
- Exit Engine (Trade Exit Logic)
- Trade Monitor (Real-time Monitoring)

Features:
- Async event-driven architecture
- JSON logging and telemetry
- Kill-switch support
- Robust error handling and recovery
- Real-time position management
- Multi-leg strategy execution
"""

import asyncio
import json
import logging
import logging.handlers
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta, time as dt_time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import threading
import queue

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Import all Beast System modules
from auth.zerodha_auth import ZerodhaAuthenticator
from adapters.kite_broker_adapter import KiteBrokerAdapter
from vrp.vrp_calculator import VRPCalculator
from regime.regime_detector import RegimeDetector
from signals.signal_generator import SignalGenerator
from positions.position_manager import PositionManager
from exits.exit_engine import ExitEngine
from monitor.trade_monitor import TradeMonitor


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class EngineState(Enum):
    """Engine operational states"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    RECOVERING = "recovering"
    SHUTDOWN = "shutdown"
    ERROR = "error"


class SignalType(Enum):
    """Trade signal types"""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"
    SHORT_PUT = "short_put"
    STRANGLE = "strangle"
    STRADDLE = "straddle"
    IRON_CONDOR = "iron_condor"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TradeTelemetry:
    """Telemetry data for a single trade"""
    trade_id: str
    timestamp: str
    signal_type: str
    entry_price: float
    entry_quantity: int
    entry_leg_count: int
    regime: str
    vrp_score: float
    market_condition: str
    entry_order_ids: List[str] = field(default_factory=list)
    status: str = "active"
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    max_pnl: float = 0.0
    exit_reason: Optional[str] = None
    exit_timestamp: Optional[str] = None
    execution_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class EngineMetrics:
    """Real-time engine metrics"""
    total_trades: int = 0
    active_trades: int = 0
    closed_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_pnl_per_trade: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    signal_count: int = 0
    execution_success_rate: float = 0.0
    last_update: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class SystemHealth:
    """System health and connectivity status"""
    broker_connected: bool = False
    websocket_connected: bool = False
    data_feed_active: bool = False
    auth_valid: bool = False
    last_heartbeat: str = ""
    error_count: int = 0
    recovery_count: int = 0
    uptime_seconds: float = 0.0
    api_call_count: int = 0
    api_error_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


# ============================================================================
# JSON LOGGER
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


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup structured JSON logging"""
    Path(log_dir).mkdir(exist_ok=True)

    logger = logging.getLogger("BeastEngine")
    logger.setLevel(logging.DEBUG)

    # File handler for JSON logs
    json_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "beast_engine.json"),
        maxBytes=100 * 1024 * 1024,  # 100MB
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

    # Telemetry handler
    telemetry_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "telemetry.json"),
        maxBytes=100 * 1024 * 1024,
        backupCount=10,
    )
    telemetry_handler.setFormatter(JSONFormatter())
    telemetry_logger = logging.getLogger("Telemetry")
    telemetry_logger.addHandler(telemetry_handler)
    telemetry_logger.setLevel(logging.INFO)

    return logger


# ============================================================================
# MAIN BEAST ENGINE
# ============================================================================

class BeastEngine:
    """
    NSE Beast System - Main Orchestrator Engine
    
    Coordinates all trading components in an async event-driven architecture.
    Handles authentication, signal generation, execution, position management,
    and monitoring in a unified system.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        access_token: str,
        user_id: str,
        log_dir: str = "logs",
        config_path: Optional[str] = None,
        enable_live_trading: bool = False,
    ):
        """
        Initialize the Beast Engine

        Args:
            api_key: Zerodha API key
            api_secret: Zerodha API secret
            access_token: Zerodha access token
            user_id: Zerodha user ID
            log_dir: Directory for logs
            config_path: Path to configuration JSON file
            enable_live_trading: Enable live trading mode
        """
        self.logger = setup_logging(log_dir)
        self.telemetry_logger = logging.getLogger("Telemetry")

        self.logger.info("Initializing NSE Beast System Engine")

        # Core configuration
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.user_id = user_id
        self.enable_live_trading = enable_live_trading
        self.log_dir = log_dir

        # Load configuration
        self.config = self._load_config(config_path)

        # State management
        self.state = EngineState.INITIALIZED
        self.kill_switch_active = False
        self.paused = False
        self.start_time = None
        self.last_heartbeat = datetime.utcnow()

        # Async components
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.tasks: List[asyncio.Task] = []

        # Module instances
        self.authenticator: Optional[ZerodhaAuthenticator] = None
        self.broker_adapter: Optional[KiteBrokerAdapter] = None
        self.vrp_calculator: Optional[VRPCalculator] = None
        self.regime_detector: Optional[RegimeDetector] = None
        self.signal_generator: Optional[SignalGenerator] = None
        self.position_manager: Optional[PositionManager] = None
        self.exit_engine: Optional[ExitEngine] = None
        self.trade_monitor: Optional[TradeMonitor] = None

        # Data stores
        self.active_trades: Dict[str, TradeTelemetry] = {}
        self.closed_trades: Dict[str, TradeTelemetry] = {}
        self.pending_signals: deque = deque(maxlen=1000)
        self.trade_history: deque = deque(maxlen=10000)

        # Metrics and health
        self.metrics = EngineMetrics()
        self.health = SystemHealth()
        self.pnl_history: deque = deque(maxlen=10000)
        self.drawdown_history: deque = deque(maxlen=10000)

        # Thread-safe queues
        self.signal_queue = queue.Queue()
        self.execution_queue = queue.Queue()
        self.alert_queue = queue.Queue()

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info(
            "Beast Engine initialization complete",
            extra={"extra_data": {"state": self.state.value}},
        )

    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "trading": {
                "max_positions": 5,
                "max_position_size": 10,
                "max_daily_loss": -5000,
                "max_daily_profit": 50000,
                "position_timeout_hours": 4,
            },
            "signals": {
                "min_vrp_score": 0.6,
                "min_regime_confidence": 0.65,
                "max_signals_per_hour": 10,
                "signal_expiry_minutes": 5,
            },
            "execution": {
                "order_timeout_seconds": 30,
                "retry_attempts": 3,
                "retry_delay_ms": 500,
            },
            "monitoring": {
                "heartbeat_interval_seconds": 5,
                "metrics_update_interval_seconds": 10,
                "trade_check_interval_seconds": 1,
            },
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")

        return default_config

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle OS signals for graceful shutdown"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.kill_switch_active = True

    async def initialize_modules(self) -> bool:
        """Initialize all trading modules asynchronously"""
        try:
            self.logger.info("Initializing trading modules")

            # Authentication
            self.authenticator = ZerodhaAuthenticator(
                self.api_key,
                self.api_secret,
                self.access_token,
                self.user_id,
                self.logger,
            )
            if not await self.authenticator.validate_session():
                raise Exception("Authentication validation failed")
            self.health.auth_valid = True

            # Broker adapter
            self.broker_adapter = KiteBrokerAdapter(
                self.authenticator,
                self.logger,
                enable_live_trading=self.enable_live_trading,
            )
            await self.broker_adapter.initialize()
            self.health.broker_connected = True

            # VRP calculator
            self.vrp_calculator = VRPCalculator(self.broker_adapter, self.logger)

            # Regime detector
            self.regime_detector = RegimeDetector(
                self.broker_adapter, self.logger
            )

            # Signal generator
            self.signal_generator = SignalGenerator(
                self.vrp_calculator, self.regime_detector, self.logger
            )

            # Position manager
            self.position_manager = PositionManager(
                self.broker_adapter, self.logger
            )
            await self.position_manager.load_positions()

            # Exit engine
            self.exit_engine = ExitEngine(self.broker_adapter, self.logger)

            # Trade monitor
            self.trade_monitor = TradeMonitor(
                self.broker_adapter, self.position_manager, self.logger
            )

            self.logger.info("All modules initialized successfully")
            self.state = EngineState.RUNNING
            return True

        except Exception as e:
            self.logger.error(
                f"Module initialization failed: {e}",
                exc_info=True,
            )
            self.state = EngineState.ERROR
            return False

    async def run(self) -> None:
        """
        Main entry point to run the Beast Engine
        
        Orchestrates all async tasks and handles the event loop
        """
        try:
            self.logger.info("Starting Beast Engine")
            self.start_time = datetime.utcnow()
            self.state = EngineState.RUNNING

            # Initialize all modules
            if not await self.initialize_modules():
                raise Exception("Module initialization failed")

            # Create main event loop
            self.loop = asyncio.get_event_loop()

            # Schedule all coroutines
            self.tasks = [
                asyncio.create_task(self._run_loop()),
                asyncio.create_task(self._heartbeat_loop()),
                asyncio.create_task(self._metrics_loop()),
                asyncio.create_task(self._health_check_loop()),
            ]

            # Wait for all tasks
            await asyncio.gather(*self.tasks)

        except Exception as e:
            self.logger.error(f"Beast Engine fatal error: {e}", exc_info=True)
            self.state = EngineState.ERROR
        finally:
            await self.shutdown()

    async def _run_loop(self) -> None:
        """
        Main trading loop
        
        Processes signals, manages positions, and checks exits continuously
        """
        market_open = dt_time(9, 15)
        market_close = dt_time(15, 30)

        while not self.kill_switch_active:
            try:
                current_time = datetime.now().time()

                # Check market hours
                if not (market_open <= current_time <= market_close):
                    self.logger.debug("Market closed, sleeping")
                    await asyncio.sleep(60)
                    continue

                # Skip if paused
                if self.paused:
                    await asyncio.sleep(1)
                    continue

                # Main trading cycle
                loop_start = time.time()

                # Step 1: Process signals
                await self._process_signals()

                # Step 2: Manage existing positions
                await self._manage_positions()

                # Step 3: Check exits
                await self._check_exits()

                # Step 4: Collect telemetry
                await self._collect_telemetry()

                loop_time = time.time() - loop_start
                if loop_time > 1.0:
                    self.logger.warning(
                        f"Trading loop took {loop_time:.2f}s (slow)",
                        extra={
                            "extra_data": {
                                "loop_time_ms": loop_time * 1000
                            }
                        },
                    )

                # Sleep to prevent busy-waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(
                    f"Error in main trading loop: {e}",
                    exc_info=True,
                )
                self.health.error_count += 1
                await asyncio.sleep(1)

    async def _process_signals(self) -> None:
        """
        Process incoming trade signals
        
        Validates signals, checks position limits, and executes trades
        """
        try:
            # Generate new signals
            signals = await self.signal_generator.generate_signals(
                self.config["signals"]
            )

            for signal in signals:
                try:
                    # Check if signal is valid
                    if not await self._validate_signal(signal):
                        continue

                    # Check position limits
                    if len(self.active_trades) >= self.config["trading"]["max_positions"]:
                        self.logger.info("Max positions reached, skipping signal")
                        continue

                    # Check VRP score
                    if signal.vrp_score < self.config["signals"]["min_vrp_score"]:
                        self.logger.debug(
                            f"VRP score too low: {signal.vrp_score}"
                        )
                        continue

                    # Check regime confidence
                    if (
                        signal.regime_confidence
                        < self.config["signals"]["min_regime_confidence"]
                    ):
                        self.logger.debug(
                            f"Regime confidence too low: {signal.regime_confidence}"
                        )
                        continue

                    # Execute signal
                    trade_id = await self._execute_signal(signal)
                    if trade_id:
                        self.metrics.signal_count += 1
                        self.pending_signals.append(
                            {
                                "trade_id": trade_id,
                                "signal": signal,
                                "timestamp": datetime.utcnow(),
                            }
                        )

                except Exception as e:
                    self.logger.error(
                        f"Error processing signal: {e}",
                        exc_info=True,
                    )

        except Exception as e:
            self.logger.error(f"Error in signal processing: {e}", exc_info=True)

    async def _execute_signal(self, signal: Any) -> Optional[str]:
        """
        Execute a trade signal
        
        Args:
            signal: Trade signal object
            
        Returns:
            Trade ID if successful, None otherwise
        """
        execution_start = time.time()
        trade_id = f"TRADE_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

        try:
            self.logger.info(
                f"Executing trade signal: {signal.signal_type}",
                extra={
                    "extra_data": {
                        "trade_id": trade_id,
                        "signal_type": signal.signal_type,
                    }
                },
            )

            # Execute multi-leg orders
            order_ids = []
            for leg in signal.legs:
                try:
                    order_id = await self.broker_adapter.place_order(
                        symbol=leg.symbol,
                        order_type=leg.order_type,
                        quantity=leg.quantity,
                        price=leg.price,
                        side=leg.side,
                    )
                    order_ids.append(order_id)
                    self.logger.info(f"Order placed: {order_id}")

                except Exception as e:
                    self.logger.error(f"Failed to place order for leg: {e}")
                    # Rollback previous orders if leg fails
                    for placed_order_id in order_ids:
                        try:
                            await self.broker_adapter.cancel_order(placed_order_id)
                        except Exception as cancel_err:
                            self.logger.error(f"Failed to cancel order: {cancel_err}")
                    return None

            # Create trade telemetry
            execution_latency = (time.time() - execution_start) * 1000
            telemetry = TradeTelemetry(
                trade_id=trade_id,
                timestamp=datetime.utcnow().isoformat(),
                signal_type=signal.signal_type,
                entry_price=signal.entry_price,
                entry_quantity=sum(leg.quantity for leg in signal.legs),
                entry_leg_count=len(signal.legs),
                regime=signal.regime,
                vrp_score=signal.vrp_score,
                market_condition=signal.market_condition,
                entry_order_ids=order_ids,
                execution_latency_ms=execution_latency,
                metadata={
                    "signal_id": getattr(signal, "id", None),
                    "confidence": signal.confidence,
                },
            )

            self.active_trades[trade_id] = telemetry
            self.trade_history.append(telemetry)
            self.metrics.total_trades += 1
            self.metrics.active_trades = len(self.active_trades)

            # Log telemetry
            self._log_telemetry(telemetry)

            return trade_id

        except Exception as e:
            self.logger.error(f"Error executing signal: {e}", exc_info=True)
            return None

    async def _validate_signal(self, signal: Any) -> bool:
        """Validate signal before execution"""
        try:
            # Check signal has required attributes
            required_attrs = [
                "signal_type",
                "entry_price",
                "legs",
                "vrp_score",
                "regime",
            ]
            for attr in required_attrs:
                if not hasattr(signal, attr):
                    self.logger.warning(f"Signal missing attribute: {attr}")
                    return False

            # Check signal not expired
            if hasattr(signal, "created_at"):
                age = (
                    datetime.utcnow() - signal.created_at
                ).total_seconds()
                if age > self.config["signals"]["signal_expiry_minutes"] * 60:
                    self.logger.debug("Signal expired")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False

    async def _manage_positions(self) -> None:
        """
        Manage existing positions
        
        Updates P&L, checks position limits, and manages risk
        """
        try:
            # Update all position P&L
            for trade_id, telemetry in self.active_trades.items():
                try:
                    # Get current market prices
                    positions = await self.position_manager.get_positions()

                    for position in positions:
                        if position.get("trade_id") == trade_id:
                            # Update telemetry
                            telemetry.pnl = position.get("pnl", 0)
                            telemetry.pnl_percentage = position.get(
                                "pnl_percentage", 0
                            )
                            telemetry.max_pnl = max(
                                telemetry.max_pnl, telemetry.pnl
                            )

                            # Check daily loss limit
                            if (
                                self.metrics.total_pnl
                                < self.config["trading"]["max_daily_loss"]
                            ):
                                self.logger.warning(
                                    "Daily loss limit reached, closing all positions"
                                )
                                await self._emergency_exit_all()
                                return

                            # Check daily profit limit
                            if (
                                self.metrics.total_pnl
                                > self.config["trading"]["max_daily_profit"]
                            ):
                                self.logger.info(
                                    "Daily profit target reached, closing all positions"
                                )
                                await self._emergency_exit_all()
                                return

                            # Check position timeout
                            position_duration = (
                                datetime.utcnow()
                                - datetime.fromisoformat(telemetry.timestamp)
                            ).total_seconds()
                            if (
                                position_duration
                                > self.config["trading"]["position_timeout_hours"]
                                * 3600
                            ):
                                self.logger.info(
                                    f"Position timeout for {trade_id}"
                                )
                                await self._close_position(trade_id, "timeout")

                except Exception as e:
                    self.logger.error(
                        f"Error updating position {trade_id}: {e}",
                        exc_info=True,
                    )

        except Exception as e:
            self.logger.error(f"Error managing positions: {e}", exc_info=True)

    async def _check_exits(self) -> None:
        """
        Check exit conditions for open positions
        
        Evaluates technical indicators, risk management rules, and exit signals
        """
        try:
            for trade_id, telemetry in list(self.active_trades.items()):
                try:
                    # Check technical exit conditions
                    exit_signal = await self.exit_engine.check_exit(trade_id)

                    if exit_signal:
                        await self._close_position(
                            trade_id,
                            exit_signal.exit_reason,
                        )
                        continue

                    # Check stop-loss
                    stop_loss_pct = -2.0  # 2% stop loss
                    if telemetry.pnl_percentage < stop_loss_pct:
                        self.logger.info(f"Stop-loss triggered for {trade_id}")
                        await self._close_position(trade_id, "stop_loss")
                        continue

                    # Check take-profit
                    take_profit_pct = 3.0  # 3% take profit
                    if telemetry.pnl_percentage > take_profit_pct:
                        self.logger.info(f"Take-profit triggered for {trade_id}")
                        await self._close_position(trade_id, "take_profit")
                        continue

                except Exception as e:
                    self.logger.error(
                        f"Error checking exit for {trade_id}: {e}",
                        exc_info=True,
                    )

        except Exception as e:
            self.logger.error(f"Error in exit check: {e}", exc_info=True)

    async def _close_position(
        self,
        trade_id: str,
        exit_reason: str,
    ) -> None:
        """
        Close an open position
        
        Args:
            trade_id: Trade ID to close
            exit_reason: Reason for exit
        """
        try:
            if trade_id not in self.active_trades:
                return

            telemetry = self.active_trades.pop(trade_id)
            telemetry.status = "closed"
            telemetry.exit_reason = exit_reason
            telemetry.exit_timestamp = datetime.utcnow().isoformat()

            # Close all orders for this trade
            for order_id in telemetry.entry_order_ids:
                try:
                    await self.broker_adapter.cancel_order(order_id)
                except Exception as e:
                    self.logger.error(f"Failed to cancel order {order_id}: {e}")

            # Move to closed trades
            self.closed_trades[trade_id] = telemetry

            # Update metrics
            self.metrics.active_trades = len(self.active_trades)
            self.metrics.closed_trades += 1
            self.metrics.total_pnl += telemetry.pnl

            # Update win/loss stats
            if telemetry.pnl > 0:
                self.metrics.consecutive_wins += 1
                self.metrics.consecutive_losses = 0
            else:
                self.metrics.consecutive_losses += 1
                self.metrics.consecutive_wins = 0

            self.logger.info(
                f"Position closed: {trade_id}",
                extra={
                    "extra_data": {
                        "exit_reason": exit_reason,
                        "pnl": telemetry.pnl,
                        "pnl_percentage": telemetry.pnl_percentage,
                    }
                },
            )

            # Log telemetry
            self._log_telemetry(telemetry)

        except Exception as e:
            self.logger.error(f"Error closing position {trade_id}: {e}", exc_info=True)

    async def _emergency_exit_all(self) -> None:
        """Close all positions immediately (emergency procedure)"""
        self.logger.critical("EMERGENCY EXIT - Closing all positions")

        for trade_id in list(self.active_trades.keys()):
            try:
                await self._close_position(trade_id, "emergency_exit")
            except Exception as e:
                self.logger.error(f"Error in emergency exit {trade_id}: {e}")

        self.state = EngineState.PAUSED
        self.paused = True

    async def _collect_telemetry(self) -> None:
        """Collect system telemetry for monitoring"""
        try:
            # Update metrics
            self.metrics.last_update = datetime.utcnow().isoformat()

            if self.metrics.closed_trades > 0:
                self.metrics.avg_pnl_per_trade = (
                    self.metrics.total_pnl / self.metrics.closed_trades
                )
                self.metrics.win_rate = (
                    self.metrics.consecutive_wins
                    / (self.metrics.consecutive_wins + self.metrics.consecutive_losses)
                    * 100
                )

            # Update health
            self.health.last_heartbeat = datetime.utcnow().isoformat()
            if self.start_time:
                self.health.uptime_seconds = (
                    datetime.utcnow() - self.start_time
                ).total_seconds()

            # Store PnL history
            self.pnl_history.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "pnl": self.metrics.total_pnl,
                }
            )

        except Exception as e:
            self.logger.error(f"Error collecting telemetry: {e}")

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat to check system health"""
        while not self.kill_switch_active:
            try:
                self.last_heartbeat = datetime.utcnow()

                # Check broker connectivity
                try:
                    await self.broker_adapter.get_funds()
                    self.health.broker_connected = True
                except Exception:
                    self.health.broker_connected = False
                    self.logger.warning("Broker connection lost")

                # Check websocket
                try:
                    # Check data feed status
                    self.health.data_feed_active = (
                        await self.trade_monitor.is_feed_active()
                    )
                except Exception:
                    self.health.data_feed_active = False

                await asyncio.sleep(
                    self.config["monitoring"]["heartbeat_interval_seconds"]
                )

            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)

    async def _metrics_loop(self) -> None:
        """Periodic metrics update and logging"""
        while not self.kill_switch_active:
            try:
                # Log current metrics
                self.telemetry_logger.info(
                    "Metrics update",
                    extra={"extra_data": self.metrics.to_dict()},
                )

                # Calculate Sharpe ratio
                if len(self.pnl_history) > 1:
                    pnl_values = [
                        p["pnl"] for p in list(self.pnl_history)[-252:]
                    ]  # Daily
                    if pnl_values:
                        returns = np.diff(pnl_values)
                        if len(returns) > 0 and np.std(returns) > 0:
                            self.metrics.sharpe_ratio = (
                                np.mean(returns) / np.std(returns) * np.sqrt(252)
                            )

                # Calculate max drawdown
                if len(self.pnl_history) > 1:
                    cumulative_pnl = [
                        p["pnl"] for p in list(self.pnl_history)
                    ]
                    if cumulative_pnl:
                        running_max = np.maximum.accumulate(cumulative_pnl)
                        drawdown = (
                            np.array(cumulative_pnl) - running_max
                        ) / running_max
                        self.metrics.max_drawdown = np.min(drawdown) * 100

                await asyncio.sleep(
                    self.config["monitoring"]["metrics_update_interval_seconds"]
                )

            except Exception as e:
                self.logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(5)

    async def _health_check_loop(self) -> None:
        """Periodic system health check and recovery"""
        while not self.kill_switch_active:
            try:
                # Check if engine is still responsive
                time_since_heartbeat = (
                    datetime.utcnow() - self.last_heartbeat
                ).total_seconds()

                if time_since_heartbeat > 30:
                    self.logger.warning(
                        "Engine not responsive, attempting recovery"
                    )
                    await self._attempt_recovery()

                # Check error rate
                if self.health.error_count > 100:
                    self.logger.error("Error rate too high, entering recovery mode")
                    self.state = EngineState.RECOVERING
                    await self._attempt_recovery()

                await asyncio.sleep(10)

            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(10)

    async def _attempt_recovery(self) -> None:
        """Attempt to recover from errors"""
        try:
            self.logger.info("Attempting system recovery")
            self.state = EngineState.RECOVERING
            self.health.recovery_count += 1

            # Re-authenticate
            if self.authenticator:
                if await self.authenticator.validate_session():
                    self.health.auth_valid = True
                    self.logger.info("Re-authentication successful")
                else:
                    self.logger.error("Re-authentication failed")
                    return

            # Reconnect broker
            if self.broker_adapter:
                try:
                    await self.broker_adapter.initialize()
                    self.health.broker_connected = True
                    self.logger.info("Broker reconnection successful")
                except Exception as e:
                    self.logger.error(f"Broker reconnection failed: {e}")
                    return

            self.state = EngineState.RUNNING
            self.logger.info("System recovery complete")

        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}", exc_info=True)

    def _log_telemetry(self, telemetry: TradeTelemetry) -> None:
        """Log trade telemetry"""
        try:
            self.telemetry_logger.info(
                f"Trade telemetry: {telemetry.trade_id}",
                extra={"extra_data": telemetry.to_dict()},
            )
        except Exception as e:
            self.logger.error(f"Error logging telemetry: {e}")

    async def pause(self) -> None:
        """Pause trading"""
        self.logger.info("Pausing trading")
        self.paused = True
        self.state = EngineState.PAUSED

    async def resume(self) -> None:
        """Resume trading"""
        self.logger.info("Resuming trading")
        self.paused = False
        self.state = EngineState.RUNNING

    async def shutdown(self) -> None:
        """
        Graceful shutdown of the Beast Engine
        
        Closes all positions, cancels pending orders, and cleans up resources
        """
        try:
            self.logger.info("Shutting down Beast Engine")
            self.state = EngineState.SHUTDOWN

            # Close all open positions
            for trade_id in list(self.active_trades.keys()):
                try:
                    await self._close_position(trade_id, "engine_shutdown")
                except Exception as e:
                    self.logger.error(f"Error closing position on shutdown: {e}")

            # Cancel pending orders
            try:
                await self.broker_adapter.cancel_all_orders()
            except Exception as e:
                self.logger.error(f"Error cancelling orders on shutdown: {e}")

            # Cancel all async tasks
            for task in self.tasks:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            # Disconnect broker
            try:
                await self.broker_adapter.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting broker: {e}")

            self.logger.info(
                "Beast Engine shutdown complete",
                extra={
                    "extra_data": {
                        "total_trades": self.metrics.total_trades,
                        "total_pnl": self.metrics.total_pnl,
                        "uptime_seconds": self.health.uptime_seconds,
                    }
                },
            )

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}", exc_info=True)

    def get_status(self) -> Dict:
        """Get current engine status"""
        return {
            "state": self.state.value,
            "paused": self.paused,
            "kill_switch_active": self.kill_switch_active,
            "metrics": self.metrics.to_dict(),
            "health": self.health.to_dict(),
            "active_trades": len(self.active_trades),
            "closed_trades": len(self.closed_trades),
            "uptime_seconds": (
                (datetime.utcnow() - self.start_time).total_seconds()
                if self.start_time
                else 0
            ),
        }

    def get_active_trades(self) -> List[Dict]:
        """Get list of active trades"""
        return [t.to_dict() for t in self.active_trades.values()]

    def get_closed_trades(self) -> List[Dict]:
        """Get list of closed trades"""
        return [t.to_dict() for t in list(self.closed_trades.values())[-100:]]


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


async def main() -> None:
    """Main entry point"""
    load_dotenv()

    # Get credentials from environment
    api_key = os.getenv("ZERODHA_API_KEY")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
    user_id = os.getenv("ZERODHA_USER_ID")

    if not all([api_key, api_secret, access_token, user_id]):
        print("Error: Missing required environment variables")
        sys.exit(1)

    # Create and run engine
    engine = BeastEngine(
        api_key=api_key,
        api_secret=api_secret,
        access_token=access_token,
        user_id=user_id,
        config_path="config/beast_engine_config.json",
        enable_live_trading=True,
    )

    try:
        await engine.run()
    except KeyboardInterrupt:
        print("\nShutdown signal received")
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
