"""
NSE Beast System - AI Trader Entry Point Script
================================================

Main entry point for running the AI trading system.

Responsibilities:
1. Load configuration from environment variables
2. Initialize all subsystems (broker, backtester, optimizer, learner)
3. Wire components together
4. Handle graceful shutdown on signals
5. Run the main trading loop

Usage:
    python -m nse_beast_system.ai_trader.run_ai_trader --symbol NIFTY50 --dry-run
    python run_ai_trader.py --log-level DEBUG

Environment Variables:
    ZERODHA_API_KEY: Zerodha API key
    ZERODHA_API_SECRET: Zerodha API secret
    ZERODHA_ACCESS_TOKEN: Zerodha access token
    ENABLE_LIVE_TRADING: Enable live trading (true/false)
    ENGINE_MODE: Engine mode (live/paper/backtest)
    TRADING_TIMEZONE: Trading timezone
    LOG_LEVEL: Logging level (DEBUG/INFO/WARNING/ERROR)

Author: NSE Beast System Team
Created: 2025-06-05
"""

import argparse
import asyncio
import json
import logging
import logging.handlers
import os
import signal
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv

# Import AI Trader components
from nse_beast_system.ai_trader import (
    AITrader,
    AITraderConfig,
    BacktesterIntegration,
    BacktestConfig,
    ExecutionOptimizer,
    ExecutionParams,
    AdaptiveLearningScheduler,
    StrategyLearner,
    OnlineLearner
)

# Import broker adapter (assuming it exists in main Beast System)
try:
    from nse_beast_system.adapters.kite_broker_adapter import KiteBrokerAdapter
except ImportError:
    KiteBrokerAdapter = None

try:
    from nse_beast_system.auth.zerodha_auth import ZerodhaAuthenticator
except ImportError:
    ZerodhaAuthenticator = None


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """
    Setup comprehensive JSON logging for AI Trader.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Parse log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("AITraderRunner")
    logger.setLevel(numeric_level)
    
    # JSON formatter
    json_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        '"module": "%(name)s", "message": "%(message)s"}',
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "ai_trader_runner.log"),
        maxBytes=50 * 1024 * 1024,
        backupCount=10
    )
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(numeric_level)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

@dataclass
class RunnerConfig:
    """Configuration for AI Trader runner."""
    # Zerodha credentials
    api_key: str
    api_secret: str
    access_token: str
    user_id: str
    
    # Trading configuration
    symbol: str
    enable_live_trading: bool
    engine_mode: str
    trading_timezone: str
    
    # Logging
    log_level: str
    dry_run: bool
    
    # System paths
    model_path: str = "models"
    log_dir: str = "logs"
    data_dir: str = "data"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging (with credentials masked)."""
        config_dict = {
            "symbol": self.symbol,
            "enable_live_trading": self.enable_live_trading,
            "engine_mode": self.engine_mode,
            "trading_timezone": self.trading_timezone,
            "log_level": self.log_level,
            "dry_run": self.dry_run,
            "model_path": self.model_path,
            "log_dir": self.log_dir,
            "data_dir": self.data_dir,
            "api_key": self.api_key[:8] + "***" if self.api_key else None,
            "user_id": self.user_id
        }
        return config_dict


def load_config_from_env() -> RunnerConfig:
    """
    Load configuration from environment variables.
    
    Returns:
        RunnerConfig instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    try:
        # Load .env file if it exists
        load_dotenv()
        
        # Required credentials
        api_key = os.getenv("ZERODHA_API_KEY")
        api_secret = os.getenv("ZERODHA_API_SECRET")
        access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
        user_id = os.getenv("ZERODHA_USER_ID")
        
        if not all([api_key, api_secret, access_token, user_id]):
            raise ValueError(
                "Missing required Zerodha credentials in environment variables. "
                "Set ZERODHA_API_KEY, ZERODHA_API_SECRET, ZERODHA_ACCESS_TOKEN, "
                "ZERODHA_USER_ID"
            )
        
        # Optional configuration
        symbol = os.getenv("TRADING_SYMBOL", "NIFTY50")
        enable_live_trading = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
        engine_mode = os.getenv("ENGINE_MODE", "paper")
        trading_timezone = os.getenv("TRADING_TIMEZONE", "Asia/Kolkata")
        log_level = os.getenv("LOG_LEVEL", "INFO")
        
        config = RunnerConfig(
            api_key=api_key,
            api_secret=api_secret,
            access_token=access_token,
            user_id=user_id,
            symbol=symbol,
            enable_live_trading=enable_live_trading,
            engine_mode=engine_mode,
            trading_timezone=trading_timezone,
            log_level=log_level,
            dry_run=False
        )
        
        logger.info(json.dumps({
            "event": "config_loaded_from_env",
            "config": config.to_dict(),
            "timestamp": datetime.now().isoformat()
        }))
        
        return config
        
    except Exception as e:
        logger.error(json.dumps({
            "event": "load_config_from_env_error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }))
        raise


# ============================================================================
# SUBSYSTEM INITIALIZATION
# ============================================================================

def setup_kite_adapter(config: RunnerConfig) -> Optional[KiteBrokerAdapter]:
    """
    Initialize KiteBrokerAdapter with Zerodha authentication.
    
    Args:
        config: RunnerConfig instance
        
    Returns:
        Initialized KiteBrokerAdapter or None if not available
        
    Raises:
        RuntimeError: If adapter initialization fails
    """
    try:
        if KiteBrokerAdapter is None:
            logger.warning(json.dumps({
                "event": "kite_adapter_not_available",
                "message": "KiteBrokerAdapter not found in nse_beast_system.adapters",
                "timestamp": datetime.now().isoformat()
            }))
            return None
        
        logger.info(json.dumps({
            "event": "initializing_kite_adapter",
            "user_id": config.user_id,
            "timestamp": datetime.now().isoformat()
        }))
        
        adapter = KiteBrokerAdapter(
            api_key=config.api_key,
            api_secret=config.api_secret,
            access_token=config.access_token,
            user_id=config.user_id,
            enable_live_trading=config.enable_live_trading
        )
        
        logger.info(json.dumps({
            "event": "kite_adapter_initialized",
            "timestamp": datetime.now().isoformat()
        }))
        
        return adapter
        
    except Exception as e:
        logger.error(json.dumps({
            "event": "setup_kite_adapter_error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }))
        raise


def setup_backtester_integration(
    config: RunnerConfig,
    initial_params: ExecutionParams
) -> BacktesterIntegration:
    """
    Create and initialize BacktesterIntegration.
    
    Args:
        config: RunnerConfig instance
        initial_params: Initial execution parameters
        
    Returns:
        Initialized BacktesterIntegration
        
    Raises:
        RuntimeError: If initialization fails
    """
    try:
        logger.info(json.dumps({
            "event": "initializing_backtester_integration",
            "symbol": config.symbol,
            "timestamp": datetime.now().isoformat()
        }))
        
        backtest_config = BacktestConfig(
            initial_capital=100000.0,
            symbol=config.symbol,
            start_date="2025-01-01",
            end_date="2025-12-31",
            use_cache=True,
            cache_dir=os.path.join(config.data_dir, "cache")
        )
        
        integration = BacktesterIntegration(
            config=backtest_config,
            initial_params=initial_params
        )
        
        logger.info(json.dumps({
            "event": "backtester_integration_initialized",
            "timestamp": datetime.now().isoformat()
        }))
        
        return integration
        
    except Exception as e:
        logger.error(json.dumps({
            "event": "setup_backtester_integration_error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }))
        raise


def setup_execution_optimizer() -> ExecutionOptimizer:
    """
    Create and initialize ExecutionOptimizer.
    
    Returns:
        Initialized ExecutionOptimizer
        
    Raises:
        RuntimeError: If initialization fails
    """
    try:
        logger.info(json.dumps({
            "event": "initializing_execution_optimizer",
            "timestamp": datetime.now().isoformat()
        }))
        
        initial_params = ExecutionParams(
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
        
        optimizer = ExecutionOptimizer(initial_params=initial_params)
        
        logger.info(json.dumps({
            "event": "execution_optimizer_initialized",
            "initial_params": initial_params.to_dict(),
            "timestamp": datetime.now().isoformat()
        }))
        
        return optimizer, initial_params
        
    except Exception as e:
        logger.error(json.dumps({
            "event": "setup_execution_optimizer_error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }))
        raise


def setup_strategy_learner(model_path: str) -> StrategyLearner:
    """
    Create or load StrategyLearner.
    
    Args:
        model_path: Path for model storage
        
    Returns:
        StrategyLearner instance (loaded if available, new otherwise)
        
    Raises:
        RuntimeError: If initialization fails
    """
    try:
        logger.info(json.dumps({
            "event": "initializing_strategy_learner",
            "model_path": model_path,
            "timestamp": datetime.now().isoformat()
        }))
        
        learner = StrategyLearner(model_path=model_path)
        
        # Try to load existing model
        if learner.load_model():
            logger.info(json.dumps({
                "event": "strategy_learner_model_loaded",
                "timestamp": datetime.now().isoformat()
            }))
        else:
            logger.info(json.dumps({
                "event": "strategy_learner_model_not_found",
                "message": "Will train new model as records accumulate",
                "timestamp": datetime.now().isoformat()
            }))
        
        return learner
        
    except Exception as e:
        logger.error(json.dumps({
            "event": "setup_strategy_learner_error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }))
        raise


# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

class ShutdownHandler:
    """Handles graceful shutdown on signals."""
    
    def __init__(self):
        self.should_shutdown = False
        self.shutdown_event: Optional[asyncio.Event] = None
    
    def handle_signal(self, signum, frame):
        """Signal handler for SIGINT and SIGTERM."""
        logger.warning(json.dumps({
            "event": "shutdown_signal_received",
            "signal": signal.Signals(signum).name,
            "timestamp": datetime.now().isoformat()
        }))
        self.should_shutdown = True
        if self.shutdown_event:
            self.shutdown_event.set()
    
    def setup_handlers(self):
        """Register signal handlers."""
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        
        logger.info(json.dumps({
            "event": "signal_handlers_registered",
            "signals": ["SIGINT", "SIGTERM"],
            "timestamp": datetime.now().isoformat()
        }))


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main(args: argparse.Namespace):
    """
    Main entry point for AI Trader.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        Exception: If trading loop fails
    """
    shutdown_handler = ShutdownHandler()
    shutdown_handler.setup_handlers()
    
    try:
        # Load configuration
        logger.info(json.dumps({
            "event": "ai_trader_startup",
            "timestamp": datetime.now().isoformat()
        }))
        
        config = load_config_from_env()
        
        # Override with command line arguments
        if args.symbol:
            config.symbol = args.symbol
        if args.log_level:
            config.log_level = args.log_level
        if args.dry_run:
            config.dry_run = True
            config.enable_live_trading = False
        
        # Setup execution optimizer (needed for backtester integration)
        optimizer, initial_params = setup_execution_optimizer()
        
        # Setup backtester integration
        backtester = setup_backtester_integration(config, initial_params)
        
        # Setup broker adapter (optional)
        kite_adapter = None
        if not config.dry_run and config.enable_live_trading:
            try:
                kite_adapter = setup_kite_adapter(config)
            except Exception as e:
                logger.error(json.dumps({
                    "event": "kite_adapter_setup_failed",
                    "error": str(e),
                    "message": "Continuing in dry-run mode",
                    "timestamp": datetime.now().isoformat()
                }))
                config.dry_run = True
        
        # Setup strategy learner
        strategy_learner = setup_strategy_learner(config.model_path)
        
        # Setup adaptive learning scheduler
        scheduler = AdaptiveLearningScheduler(
            reoptimize_every_n_trades=50,
            drawdown_trigger=-5.0,
            regime_change_trigger=True
        )
        
        # Create AI Trader configuration
        trader_config = AITraderConfig(
            enable_live_trading=config.enable_live_trading and not config.dry_run,
            engine_mode=config.engine_mode,
            trading_timezone=config.trading_timezone,
            symbol=config.symbol
        )
        
        # Initialize AI Trader
        logger.info(json.dumps({
            "event": "initializing_ai_trader",
            "config": {
                "symbol": trader_config.symbol,
                "engine_mode": trader_config.engine_mode,
                "enable_live_trading": trader_config.enable_live_trading,
                "dry_run": config.dry_run
            },
            "timestamp": datetime.now().isoformat()
        }))
        
        ai_trader = AITrader(
            config=trader_config,
            kite_adapter=kite_adapter,
            backtester_integration=backtester,
            execution_optimizer=optimizer,
            strategy_learner=strategy_learner,
            scheduler=scheduler
        )
        
        # Create shutdown event for async handling
        shutdown_handler.shutdown_event = asyncio.Event()
        
        # Run trading loop
        logger.info(json.dumps({
            "event": "starting_trading_loop",
            "mode": "DRY_RUN" if config.dry_run else "LIVE",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Create task for trading loop
        trading_task = asyncio.create_task(ai_trader.run_trading_loop())
        
        # Create task to wait for shutdown signal
        async def wait_for_shutdown():
            await shutdown_handler.shutdown_event.wait()
        
        shutdown_task = asyncio.create_task(wait_for_shutdown())
        
        # Wait for either trading loop to end or shutdown signal
        done, pending = await asyncio.wait(
            [trading_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        logger.info(json.dumps({
            "event": "trading_loop_ended",
            "timestamp": datetime.now().isoformat()
        }))
        
    except KeyboardInterrupt:
        logger.warning(json.dumps({
            "event": "keyboard_interrupt",
            "timestamp": datetime.now().isoformat()
        }))
    except Exception as e:
        logger.error(json.dumps({
            "event": "main_error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }))
        sys.exit(1)
    finally:
        logger.info(json.dumps({
            "event": "ai_trader_shutdown",
            "timestamp": datetime.now().isoformat()
        }))


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create command line argument parser.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="NSE Beast System - AI Trader Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ai_trader.py --symbol NIFTY50 --dry-run
  python run_ai_trader.py --log-level DEBUG --enable-live
  python -m nse_beast_system.ai_trader.run_ai_trader --symbol BANKNIFTY
        """
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading symbol (e.g., NIFTY50, BANKNIFTY)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no real trades)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level"
    )
    
    parser.add_argument(
        "--enable-live",
        action="store_true",
        help="Enable live trading (overrides env var)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="NSE Beast System AI Trader v1.0.0"
    )
    
    return parser


# ============================================================================
# SCRIPT EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # Parse arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Setup logging with specified level
        if args.log_level:
            logger = setup_logging(log_level=args.log_level)
        
        logger.info(json.dumps({
            "event": "script_started",
            "args": {
                "symbol": args.symbol,
                "dry_run": args.dry_run,
                "log_level": args.log_level,
                "enable_live": args.enable_live
            },
            "timestamp": datetime.now().isoformat()
        }))
        
        # Run main async function
        asyncio.run(main(args))
        
    except KeyboardInterrupt:
        logger.info(json.dumps({
            "event": "script_interrupted",
            "timestamp": datetime.now().isoformat()
        }))
        sys.exit(0)
    except Exception as e:
        logger.error(json.dumps({
            "event": "script_error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }))
        sys.exit(1)
