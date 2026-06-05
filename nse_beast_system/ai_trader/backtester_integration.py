"""
NSE Beast System - Backtester Integration Bridge
=================================================

This module bridges AITrader and lambdaclass/options_portfolio_backtester.

Handles:
1. NSE options chain data conversion to backtester schema
2. Strategy building (Strangle, IronCondor, Butterfly)
3. BacktestEngine configuration with NSE-specific parameters
4. Validation backtests before live execution
5. Parameter sweep to find optimal Greeks targets
6. Fill quality analysis and model selection
7. Execution metrics extraction

Key Classes:
- BacktesterIntegration: Main bridge class
- NSEOptionsDataAdapter: Converts KiteConnect data to backtester format
- StrategyBuilder: Builds multi-leg strategies
- EngineConfigurator: Wires up engine components
"""

import asyncio
import json
import logging
import logging.handlers
import os
import sys
import traceback
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv

# Backtester imports
from options_portfolio_backtester.engine import BacktestEngine
from options_portfolio_backtester.strategy import Strategy, StrategyLeg, Strangle, IronCondor, Butterfly
from options_portfolio_backtester.signal_selectors import NearestDelta, MaxOpenInterest
from options_portfolio_backtester.cost_models import SpreadSlippage, PerContractCommission
from options_portfolio_backtester.fill_models import MidPrice, VolumeAwareFill, MarketAtBidAsk
from options_portfolio_backtester.sizers import PercentOfPortfolio, CapitalBased
from options_portfolio_backtester.risk_managers import MaxDelta, MaxVega, MaxDrawdown

# Beast System imports
from nse_beast_system.ai_trader.ai_trader import (
    AITraderConfig,
    SignalType,
    TradeDecision,
    BacktestResult,
    BacktestStatus,
    ExecutionOptimizer,
)


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

NSE_INDICES = {
    "NIFTY": {"spot_symbol": "NIFTY 50", "prefix": "NIFTY", "underlying": "NIFTY50"},
    "BANKNIFTY": {"spot_symbol": "NIFTY BANK", "prefix": "BANKNIFTY", "underlying": "BANKNIFTY"},
    "FINNIFTY": {"spot_symbol": "NIFTY FIN SERVICE", "prefix": "FINNIFTY", "underlying": "FINNIFTY"},
}

TICK_SIZES = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "FINNIFTY": 50,
}

# Options expiry schedule (NSE standard)
NSE_WEEKLY_EXPIRY_DAYS = [3]  # Wednesday
NSE_MONTHLY_EXPIRY_DAYS = [24, 25, 26, 27, 28, 29, 30, 31]  # Last Thursday

STRATEGY_CONFIGS = {
    SignalType.IRON_CONDOR: {
        "num_legs": 4,
        "strikes": [-200, -300, 200, 300],
        "sides": ["SELL", "BUY", "SELL", "BUY"],
        "types": ["PE", "PE", "CE", "CE"],
    },
    SignalType.SHORT_STRANGLE: {
        "num_legs": 2,
        "strikes": [-150, 150],
        "sides": ["SELL", "SELL"],
        "types": ["PE", "CE"],
    },
    SignalType.LONG_STRANGLE: {
        "num_legs": 2,
        "strikes": [-150, 150],
        "sides": ["BUY", "BUY"],
        "types": ["PE", "CE"],
    },
    SignalType.BUTTERFLY: {
        "num_legs": 4,
        "strikes": [-100, -200, 100, 200],
        "sides": ["BUY", "SELL", "SELL", "BUY"],
        "types": ["CE", "CE", "CE", "CE"],
    },
    SignalType.COVERED_CALL: {
        "num_legs": 1,
        "strikes": [100],
        "sides": ["SELL"],
        "types": ["CE"],
    },
}


# ============================================================================
# LOGGING SETUP
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


def setup_backtester_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup structured logging for backtester integration"""
    Path(log_dir).mkdir(exist_ok=True)

    logger = logging.getLogger("BacktesterIntegration")
    logger.setLevel(logging.DEBUG)

    # File handler for JSON logs
    json_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "backtester_integration.json"),
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

    return logger


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class StrategyLegSpec:
    """Specification for a single strategy leg"""
    option_type: str  # "CE" or "PE"
    strike: int  # Absolute strike price
    strike_offset: int  # Offset from ATM
    side: str  # "BUY" or "SELL"
    quantity: int = 1
    delta_target: float = 0.0
    iv: float = 0.0
    bid_price: float = 0.0
    ask_price: float = 0.0
    open_interest: int = 0


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics"""
    win_rate: float
    loss_rate: float
    avg_pnl: float
    median_pnl: float
    total_pnl: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_hold_days: float
    total_trades: int
    total_winners: int
    total_losers: int
    profit_factor: float
    avg_fill_slippage: float
    execution_quality: float


@dataclass
class FillQualityMetrics:
    """Fill quality analysis"""
    avg_slippage_pct: float
    median_slippage_pct: float
    best_fill_quality_pct: float
    worst_fill_quality_pct: float
    fill_execution_score: float  # 0-1, higher is better


# ============================================================================
# NSE OPTIONS DATA ADAPTER
# ============================================================================

class NSEOptionsDataAdapter:
    """
    Converts KiteConnect options chain response to backtester-compatible format

    NSE Options Chain Response Format:
    {
        "data": {
            "options": [
                {
                    "instrument_token": 12345,
                    "tradingsymbol": "NIFTY23JUN20000CE",
                    "strike": 20000,
                    "expiry": "2024-06-20",
                    "option_type": "CE",
                    "bid": 150.0,
                    "ask": 155.0,
                    "last_price": 152.5,
                    "volume": 1000,
                    "oi": 50000,
                    "iv": 0.18,
                    ...
                },
                ...
            ]
        }
    }
    """

    def __init__(self, logger: logging.Logger):
        """Initialize adapter"""
        self.logger = logger
        self.underlying_price: Optional[float] = None
        self.expiry_date: Optional[datetime] = None
        self.index_name: Optional[str] = None

    def adapt_options_chain(
        self,
        options_chain_response: Dict,
        underlying_price: float,
        index_name: str,
    ) -> pd.DataFrame:
        """
        Convert KiteConnect options chain to backtester DataFrame

        Args:
            options_chain_response: Raw response from KiteConnect
            underlying_price: Current spot price
            index_name: "NIFTY", "BANKNIFTY", or "FINNIFTY"

        Returns:
            DataFrame with columns:
            [strike, expiry, option_type, bid, ask, last_price, volume, oi, iv, delta]
        """
        try:
            self.logger.info(f"Adapting options chain for {index_name} at {underlying_price}")

            self.underlying_price = underlying_price
            self.index_name = index_name

            if "data" not in options_chain_response:
                raise ValueError("Invalid options chain response: missing 'data' key")

            data = options_chain_response["data"]

            if "options" not in data or not isinstance(data["options"], list):
                raise ValueError("Invalid options chain response: missing 'options' list")

            options_list = data["options"]
            rows = []

            for option in options_list:
                try:
                    # Extract fields
                    strike = option.get("strike")
                    expiry = option.get("expiry")
                    option_type = option.get("option_type")  # "CE" or "PE"
                    bid = option.get("bid", 0.0)
                    ask = option.get("ask", 0.0)
                    last_price = option.get("last_price", 0.0)
                    volume = option.get("volume", 0)
                    oi = option.get("oi", 0)
                    iv = option.get("iv", 0.0)
                    tradingsymbol = option.get("tradingsymbol", "")

                    if not all([strike, expiry, option_type]):
                        self.logger.warning(f"Skipping option with missing fields: {option}")
                        continue

                    # Set expiry date from first option
                    if self.expiry_date is None:
                        self.expiry_date = datetime.strptime(expiry, "%Y-%m-%d")

                    # Calculate Greeks (simplified)
                    delta = self._calculate_delta(
                        underlying_price,
                        strike,
                        iv,
                        option_type,
                        self.expiry_date,
                    )

                    rows.append({
                        "strike": strike,
                        "expiry": expiry,
                        "option_type": option_type,
                        "bid": bid,
                        "ask": ask,
                        "mid_price": (bid + ask) / 2 if bid > 0 and ask > 0 else last_price,
                        "last_price": last_price,
                        "volume": volume,
                        "oi": oi,
                        "iv": iv,
                        "delta": delta,
                        "tradingsymbol": tradingsymbol,
                    })

                except Exception as e:
                    self.logger.warning(f"Error processing option: {e}")
                    continue

            if not rows:
                raise ValueError("No valid options found in chain")

            df = pd.DataFrame(rows)

            self.logger.info(
                f"Adapted {len(df)} options",
                extra={"extra_data": {
                    "index": index_name,
                    "spot": underlying_price,
                    "expiry": str(self.expiry_date),
                    "num_calls": len(df[df["option_type"] == "CE"]),
                    "num_puts": len(df[df["option_type"] == "PE"]),
                }},
            )

            return df

        except Exception as e:
            self.logger.error(f"Error adapting options chain: {e}", exc_info=True)
            raise

    def _calculate_delta(
        self,
        spot: float,
        strike: float,
        iv: float,
        option_type: str,
        expiry_date: datetime,
        risk_free_rate: float = 0.06,
    ) -> float:
        """
        Calculate option delta using Black-Scholes model

        Args:
            spot: Current spot price
            strike: Strike price
            iv: Implied volatility (0-1)
            option_type: "CE" or "PE"
            expiry_date: Expiration date
            risk_free_rate: Risk-free rate (default 6%)

        Returns:
            Delta value (-1 to 1)
        """
        try:
            if iv <= 0 or spot <= 0 or strike <= 0:
                return 0.0

            # Time to expiry in years
            time_to_expiry = (expiry_date - datetime.now()).days / 365.0
            if time_to_expiry <= 0:
                return 1.0 if option_type == "CE" and spot > strike else 0.0

            # Black-Scholes d1
            d1 = (
                np.log(spot / strike) +
                (risk_free_rate + 0.5 * iv ** 2) * time_to_expiry
            ) / (iv * np.sqrt(time_to_expiry))

            # Delta calculation
            if option_type == "CE":
                delta = stats.norm.cdf(d1)
            else:
                delta = stats.norm.cdf(d1) - 1

            return float(delta)

        except Exception as e:
            self.logger.warning(f"Error calculating delta: {e}")
            return 0.0


# ============================================================================
# STRATEGY BUILDER
# ============================================================================

class StrategyBuilder:
    """Builds multi-leg strategies from signal parameters"""

    def __init__(self, logger: logging.Logger):
        """Initialize builder"""
        self.logger = logger

    def build_strategy(
        self,
        signal_type: SignalType,
        options_df: pd.DataFrame,
        spot_price: float,
        config: AITraderConfig,
    ) -> Tuple[Strategy, List[StrategyLegSpec]]:
        """
        Build strategy with legs

        Args:
            signal_type: SignalType enum
            options_df: Adapted options DataFrame
            spot_price: Current spot price
            config: AITraderConfig

        Returns:
            Tuple of (Strategy, list of StrategyLegSpec)
        """
        try:
            self.logger.info(f"Building strategy: {signal_type.value}", extra={
                "extra_data": {"spot_price": spot_price}
            })

            # Create strategy object
            if signal_type == SignalType.IRON_CONDOR:
                strategy = IronCondor()
            elif signal_type == SignalType.SHORT_STRANGLE:
                strategy = Strangle(is_long=False)
            elif signal_type == SignalType.LONG_STRANGLE:
                strategy = Strangle(is_long=True)
            elif signal_type == SignalType.BUTTERFLY:
                strategy = Butterfly()
            else:
                raise ValueError(f"Unknown strategy type: {signal_type}")

            # Get strategy config
            strat_config = STRATEGY_CONFIGS.get(signal_type)
            if not strat_config:
                raise ValueError(f"No config for strategy: {signal_type}")

            # Build legs
            legs = []
            leg_specs = []

            for i, strike_offset in enumerate(strat_config["strikes"]):
                leg_side = strat_config["sides"][i]
                leg_type = strat_config["types"][i]

                # Select strike
                strike = int(spot_price) + strike_offset
                strike = self._round_strike(strike)

                # Find option in chain
                option = self._find_option(options_df, strike, leg_type)
                if option is None:
                    self.logger.warning(f"No option found for {leg_type} {strike}")
                    continue

                # Determine delta target
                delta_target = self._get_delta_target(
                    leg_type, leg_side, config
                )

                # Create leg spec
                leg_spec = StrategyLegSpec(
                    option_type=leg_type,
                    strike=strike,
                    strike_offset=strike_offset,
                    side=leg_side,
                    quantity=1,
                    delta_target=delta_target,
                    iv=float(option["iv"]),
                    bid_price=float(option["bid"]),
                    ask_price=float(option["ask"]),
                    open_interest=int(option["oi"]),
                )
                leg_specs.append(leg_spec)

                # Create and add strategy leg
                strategy_leg = StrategyLeg(
                    strike=strike,
                    option_type=leg_type,
                    side=leg_side,
                    quantity=1,
                )
                strategy.add_leg(strategy_leg)

                self.logger.debug(
                    f"Added leg: {leg_type} {strike} {leg_side}",
                    extra={"extra_data": asdict(leg_spec)},
                )

            if not leg_specs:
                raise ValueError(f"No legs built for strategy: {signal_type}")

            self.logger.info(
                f"Strategy built with {len(leg_specs)} legs",
                extra={"extra_data": {
                    "strategy": signal_type.value,
                    "num_legs": len(leg_specs),
                    "legs": [asdict(leg) for leg in leg_specs],
                }},
            )

            return strategy, leg_specs

        except Exception as e:
            self.logger.error(f"Error building strategy: {e}", exc_info=True)
            raise

    def _round_strike(self, strike: float) -> int:
        """Round strike to nearest valid NSE strike"""
        # NSE strikes are typically in 50/100 increments
        return int(round(strike / 50) * 50)

    def _find_option(
        self,
        options_df: pd.DataFrame,
        strike: int,
        option_type: str,
    ) -> Optional[pd.Series]:
        """Find option by strike and type"""
        try:
            mask = (options_df["strike"] == strike) & \
                   (options_df["option_type"] == option_type)
            matching = options_df[mask]

            if matching.empty:
                # Try to find nearest strike
                if option_type == "CE":
                    calls = options_df[options_df["option_type"] == "CE"]
                    nearest = calls.loc[(calls["strike"] - strike).abs().idxmin()]
                    return nearest
                else:
                    puts = options_df[options_df["option_type"] == "PE"]
                    nearest = puts.loc[(puts["strike"] - strike).abs().idxmin()]
                    return nearest

            return matching.iloc[0]

        except Exception as e:
            self.logger.error(f"Error finding option: {e}")
            return None

    def _get_delta_target(
        self,
        option_type: str,
        side: str,
        config: AITraderConfig,
    ) -> float:
        """Get delta target based on option type and side"""
        if option_type == "CE":
            if side == "BUY":
                return config.delta_target_long
            else:
                return config.delta_target_short
        else:  # PE
            if side == "BUY":
                return -config.delta_target_long
            else:
                return -config.delta_target_short


# ============================================================================
# ENGINE CONFIGURATOR
# ============================================================================

class EngineConfigurator:
    """Configures BacktestEngine components"""

    def __init__(self, logger: logging.Logger):
        """Initialize configurator"""
        self.logger = logger

    def build_engine(
        self,
        strategy: Strategy,
        execution_optimizer: ExecutionOptimizer,
        config: AITraderConfig,
    ) -> BacktestEngine:
        """
        Build configured BacktestEngine

        Args:
            strategy: Built strategy
            execution_optimizer: Current execution parameters
            config: AITraderConfig

        Returns:
            Configured BacktestEngine
        """
        try:
            self.logger.info("Building BacktestEngine", extra={
                "extra_data": {
                    "fill_model": execution_optimizer.fill_model_type,
                    "target_delta": execution_optimizer.target_delta,
                }
            })

            # Cost model
            cost_model = PerContractCommission(
                rate=execution_optimizer.commission_per_contract / 100.0
            )

            # Fill model
            fill_model = self._get_fill_model(
                execution_optimizer.fill_model_type,
                execution_optimizer.bid_ask_spread_pct,
            )

            # Position sizer
            sizer = PercentOfPortfolio(pct=config.position_size_pct / 100.0)

            # Signal selector
            signal_selector = NearestDelta(
                target=execution_optimizer.target_delta
            )

            # Risk managers
            risk_managers = [
                MaxDelta(max_delta=config.max_portfolio_delta),
                MaxVega(max_vega=config.max_portfolio_vega),
                MaxDrawdown(max_drawdown_pct=config.max_drawdown_pct),
            ]

            # Create engine
            engine = BacktestEngine(
                strategy=strategy,
                cost_model=cost_model,
                fill_model=fill_model,
                sizer=sizer,
                signal_selector=signal_selector,
                risk_managers=risk_managers,
            )

            self.logger.info("BacktestEngine created successfully")
            return engine

        except Exception as e:
            self.logger.error(f"Error building engine: {e}", exc_info=True)
            raise

    def _get_fill_model(self, model_type: str, spread: float):
        """Get fill model by type"""
        if model_type == "volume_aware":
            return VolumeAwareFill(threshold=spread)
        elif model_type == "market_at_bid_ask":
            return MarketAtBidAsk()
        else:
            return MidPrice()


# ============================================================================
# BACKTEST RESULTS ANALYZER
# ============================================================================

class BacktestResultsAnalyzer:
    """Analyzes backtest results and extracts metrics"""

    def __init__(self, logger: logging.Logger):
        """Initialize analyzer"""
        self.logger = logger

    def extract_metrics(self, backtest_df: pd.DataFrame) -> BacktestMetrics:
        """
        Extract comprehensive metrics from backtest results

        Args:
            backtest_df: Results DataFrame from engine.run()

        Returns:
            BacktestMetrics with all statistics
        """
        try:
            if backtest_df.empty:
                return self._empty_metrics()

            # P&L metrics
            total_pnl = backtest_df["pnl"].sum() if "pnl" in backtest_df.columns else 0.0
            avg_pnl = backtest_df["pnl"].mean() if "pnl" in backtest_df.columns else 0.0
            median_pnl = backtest_df["pnl"].median() if "pnl" in backtest_df.columns else 0.0

            # Win rate
            if "pnl" in backtest_df.columns:
                winners = len(backtest_df[backtest_df["pnl"] > 0])
                losers = len(backtest_df[backtest_df["pnl"] < 0])
            else:
                winners = 0
                losers = 0

            total_trades = len(backtest_df)
            win_rate = winners / total_trades if total_trades > 0 else 0.0
            loss_rate = losers / total_trades if total_trades > 0 else 0.0

            # Sharpe ratio
            if "pnl" in backtest_df.columns and len(backtest_df) > 1:
                returns = backtest_df["pnl"].pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    sharpe = returns.mean() / returns.std() * np.sqrt(252)
                else:
                    sharpe = 0.0
            else:
                sharpe = 0.0

            # Sortino ratio
            if "pnl" in backtest_df.columns and len(backtest_df) > 1:
                returns = backtest_df["pnl"].pct_change().dropna()
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0 and downside_returns.std() > 0:
                    sortino = returns.mean() / downside_returns.std() * np.sqrt(252)
                else:
                    sortino = 0.0
            else:
                sortino = 0.0

            # Drawdown
            if "cumulative_pnl" in backtest_df.columns:
                cumulative = backtest_df["cumulative_pnl"]
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = abs(drawdown.min())
                max_drawdown_pct = max_drawdown * 100
            else:
                max_drawdown = 0.0
                max_drawdown_pct = 0.0

            # Hold time
            if "duration_hours" in backtest_df.columns:
                avg_hold = backtest_df["duration_hours"].mean()
            else:
                avg_hold = 0.0

            # Profit factor
            if winners > 0 and losers > 0:
                gross_profit = backtest_df[backtest_df["pnl"] > 0]["pnl"].sum()
                gross_loss = abs(backtest_df[backtest_df["pnl"] < 0]["pnl"].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            else:
                profit_factor = 0.0

            # Fill quality
            if "fill_quality" in backtest_df.columns:
                avg_slippage = (1 - backtest_df["fill_quality"].mean()) * 100
            else:
                avg_slippage = 0.0

            metrics = BacktestMetrics(
                win_rate=win_rate,
                loss_rate=loss_rate,
                avg_pnl=avg_pnl,
                median_pnl=median_pnl,
                total_pnl=total_pnl,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_drawdown,
                max_drawdown_pct=max_drawdown_pct,
                avg_hold_days=avg_hold / 24,  # Convert hours to days
                total_trades=total_trades,
                total_winners=winners,
                total_losers=losers,
                profit_factor=profit_factor,
                avg_fill_slippage=avg_slippage,
                execution_quality=(1 - avg_slippage / 100) if avg_slippage > 0 else 1.0,
            )

            self.logger.info(
                "Metrics extracted",
                extra={"extra_data": {
                    "total_trades": total_trades,
                    "win_rate": f"{win_rate:.2%}",
                    "sharpe": f"{sharpe:.2f}",
                    "max_dd": f"{max_drawdown_pct:.2f}%",
                    "total_pnl": f"{total_pnl:.2f}",
                }},
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error extracting metrics: {e}", exc_info=True)
            return self._empty_metrics()

    def extract_fill_quality(self, backtest_df: pd.DataFrame) -> float:
        """
        Extract fill quality (0-1, higher is better)

        Args:
            backtest_df: Backtest results

        Returns:
            Fill quality score
        """
        try:
            if backtest_df.empty or "fill_quality" not in backtest_df.columns:
                return 0.95  # Assume good fills

            quality = backtest_df["fill_quality"].mean()
            return max(0.0, min(1.0, quality))

        except Exception as e:
            self.logger.warning(f"Error computing fill quality: {e}")
            return 0.95

    def _empty_metrics(self) -> BacktestMetrics:
        """Return empty metrics"""
        return BacktestMetrics(
            win_rate=0.0,
            loss_rate=0.0,
            avg_pnl=0.0,
            median_pnl=0.0,
            total_pnl=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            avg_hold_days=0.0,
            total_trades=0,
            total_winners=0,
            total_losers=0,
            profit_factor=0.0,
            avg_fill_slippage=0.0,
            execution_quality=0.0,
        )


# ============================================================================
# MAIN BACKTESTER INTEGRATION CLASS
# ============================================================================

class BacktesterIntegration:
    """
    Main bridge between AITrader and options_portfolio_backtester

    Provides:
    - NSE options chain adaptation
    - Strategy building
    - Engine configuration
    - Backtest execution
    - Parameter sweep
    - Results analysis
    """

    def __init__(self, config: AITraderConfig, log_dir: str = "logs"):
        """
        Initialize BacktesterIntegration

        Args:
            config: AITraderConfig
            log_dir: Logging directory
        """
        self.config = config
        self.logger = setup_backtester_logging(log_dir)

        # Components
        self.data_adapter = NSEOptionsDataAdapter(self.logger)
        self.strategy_builder = StrategyBuilder(self.logger)
        self.engine_configurator = EngineConfigurator(self.logger)
        self.results_analyzer = BacktestResultsAnalyzer(self.logger)

        # State
        self.last_options_df: Optional[pd.DataFrame] = None
        self.last_engine: Optional[BacktestEngine] = None
        self.backtest_history: List[Dict] = []

        self.logger.info("BacktesterIntegration initialized")

    async def build_nse_schema(
        self,
        options_chain_response: Dict,
        spot_price: float,
        index_name: str,
    ) -> pd.DataFrame:
        """
        Convert NSE options chain to backtester schema

        Args:
            options_chain_response: KiteConnect options chain response
            spot_price: Current spot price
            index_name: "NIFTY", "BANKNIFTY", or "FINNIFTY"

        Returns:
            Schema DataFrame compatible with backtester
        """
        try:
            self.logger.info(f"Building NSE schema for {index_name}")

            if index_name not in NSE_INDICES:
                raise ValueError(f"Unknown index: {index_name}")

            # Adapt options chain
            schema_df = self.data_adapter.adapt_options_chain(
                options_chain_response,
                spot_price,
                index_name,
            )

            self.last_options_df = schema_df
            return schema_df

        except Exception as e:
            self.logger.error(f"Error building NSE schema: {e}", exc_info=True)
            raise

    async def build_strategy(
        self,
        signal_type: SignalType,
        options_chain_response: Dict,
        spot_price: float,
        index_name: str,
    ) -> Tuple[Strategy, List[StrategyLegSpec]]:
        """
        Build strategy from signal

        Args:
            signal_type: SignalType to build
            options_chain_response: Options chain data
            spot_price: Current spot price
            index_name: Index name

        Returns:
            Tuple of (Strategy, list of StrategyLegSpecs)
        """
        try:
            # Build schema if not cached
            if self.last_options_df is None:
                await self.build_nse_schema(
                    options_chain_response,
                    spot_price,
                    index_name,
                )

            # Build strategy
            strategy, leg_specs = self.strategy_builder.build_strategy(
                signal_type,
                self.last_options_df,
                spot_price,
                self.config,
            )

            return strategy, leg_specs

        except Exception as e:
            self.logger.error(f"Error building strategy: {e}", exc_info=True)
            raise

    async def build_engine(
        self,
        strategy: Strategy,
        execution_optimizer: ExecutionOptimizer,
    ) -> BacktestEngine:
        """
        Build configured BacktestEngine

        Args:
            strategy: Built strategy
            execution_optimizer: Execution parameters

        Returns:
            Configured BacktestEngine
        """
        try:
            engine = self.engine_configurator.build_engine(
                strategy,
                execution_optimizer,
                self.config,
            )

            self.last_engine = engine
            return engine

        except Exception as e:
            self.logger.error(f"Error building engine: {e}", exc_info=True)
            raise

    async def run_validation_backtest(
        self,
        decision: TradeDecision,
        options_chain_response: Dict,
        spot_price: float,
        index_name: str,
        execution_optimizer: ExecutionOptimizer,
        historical_data: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run full validation backtest before live execution

        Args:
            decision: TradeDecision to validate
            options_chain_response: Options chain data
            spot_price: Current spot price
            index_name: Index name
            execution_optimizer: Execution parameters
            historical_data: Optional historical data

        Returns:
            BacktestResult
        """
        backtest_start = time.time()

        try:
            self.logger.info(
                f"Starting validation backtest for {decision.signal_type.value}",
                extra={"extra_data": {"signal_id": decision.signal_id}},
            )

            # Build schema and strategy
            strategy, leg_specs = await self.build_strategy(
                decision.signal_type,
                options_chain_response,
                spot_price,
                index_name,
            )

            # Build engine
            engine = await self.build_engine(strategy, execution_optimizer)

            # Run backtest with timeout
            try:
                loop = asyncio.get_event_loop()
                backtest_df = await asyncio.wait_for(
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

            # Extract metrics
            metrics = self.results_analyzer.extract_metrics(backtest_df)
            fill_quality = self.results_analyzer.extract_fill_quality(backtest_df)

            # Build result
            result = BacktestResult(
                status=BacktestStatus.COMPLETED,
                sharpe_ratio=metrics.sharpe_ratio,
                max_drawdown_pct=metrics.max_drawdown_pct,
                win_rate=metrics.win_rate,
                avg_fill_quality=fill_quality,
                total_pnl=metrics.total_pnl,
                num_trades=metrics.total_trades,
                avg_trade_duration_hours=metrics.avg_hold_days * 24,
                trade_dataframe=backtest_df,
                execution_time_seconds=execution_time,
            )

            # Log result
            self.backtest_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "signal_id": decision.signal_id,
                "result": {
                    "sharpe": metrics.sharpe_ratio,
                    "wr": metrics.win_rate,
                    "dd": metrics.max_drawdown_pct,
                    "pnl": metrics.total_pnl,
                    "trades": metrics.total_trades,
                },
            })

            self.logger.info(
                "Backtest completed",
                extra={"extra_data": {
                    "sharpe": metrics.sharpe_ratio,
                    "win_rate": f"{metrics.win_rate:.2%}",
                    "max_dd": f"{metrics.max_drawdown_pct:.2f}%",
                    "total_pnl": metrics.total_pnl,
                    "execution_time": execution_time,
                }},
            )

            return result

        except Exception as e:
            self.logger.error(f"Error running validation backtest: {e}", exc_info=True)
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

    async def run_parameter_sweep(
        self,
        signal_type: SignalType,
        options_chain_response: Dict,
        spot_price: float,
        index_name: str,
        execution_optimizer: ExecutionOptimizer,
        param_grid: Dict[str, List],
    ) -> pd.DataFrame:
        """
        Sweep parameters to find optimal configuration

        Args:
            signal_type: Strategy type to sweep
            options_chain_response: Options chain
            spot_price: Spot price
            index_name: Index name
            execution_optimizer: Base execution params
            param_grid: Dict of params to sweep, e.g.
                {
                    "delta_target": [-0.25, -0.30, -0.35],
                    "position_size_pct": [3.0, 5.0, 7.0],
                }

        Returns:
            DataFrame with sweep results
        """
        try:
            self.logger.info(
                f"Starting parameter sweep for {signal_type.value}",
                extra={"extra_data": {"param_grid": param_grid}},
            )

            results = []

            # Generate all combinations
            param_names = list(param_grid.keys())
            param_values = [param_grid[name] for name in param_names]

            total_combinations = len(list(product(*param_values)))
            self.logger.info(f"Testing {total_combinations} parameter combinations")

            for combo_idx, combo in enumerate(product(*param_values)):
                try:
                    # Create parameter dict
                    params = dict(zip(param_names, combo))

                    # Update optimizer
                    test_optimizer = ExecutionOptimizer(
                        target_delta=params.get(
                            "delta_target", execution_optimizer.target_delta
                        ),
                        position_size_contracts=params.get(
                            "position_size_contracts",
                            execution_optimizer.position_size_contracts,
                        ),
                        bid_ask_spread_pct=params.get(
                            "bid_ask_spread_pct",
                            execution_optimizer.bid_ask_spread_pct,
                        ),
                    )

                    # Build and run strategy
                    strategy, _ = await self.build_strategy(
                        signal_type,
                        options_chain_response,
                        spot_price,
                        index_name,
                    )

                    engine = await self.build_engine(strategy, test_optimizer)

                    # Run backtest
                    loop = asyncio.get_event_loop()
                    backtest_df = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: engine.run(check_exits_daily=True),
                        ),
                        timeout=self.config.backtest_timeout_seconds,
                    )

                    # Extract metrics
                    metrics = self.results_analyzer.extract_metrics(backtest_df)

                    result_row = {
                        "combo_idx": combo_idx,
                        **params,
                        "sharpe": metrics.sharpe_ratio,
                        "win_rate": metrics.win_rate,
                        "max_drawdown": metrics.max_drawdown_pct,
                        "total_pnl": metrics.total_pnl,
                        "num_trades": metrics.total_trades,
                        "profit_factor": metrics.profit_factor,
                    }
                    results.append(result_row)

                    if (combo_idx + 1) % max(1, total_combinations // 10) == 0:
                        self.logger.info(
                            f"Sweep progress: {combo_idx + 1}/{total_combinations}"
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Error in parameter combo {combo_idx}: {e}"
                    )
                    continue

            if not results:
                raise ValueError("No successful parameter combinations")

            results_df = pd.DataFrame(results)

            self.logger.info(
                f"Parameter sweep completed",
                extra={"extra_data": {
                    "total_combos": total_combinations,
                    "successful": len(results),
                    "best_sharpe": results_df["sharpe"].max(),
                    "best_wr": results_df["win_rate"].max(),
                }},
            )

            return results_df

        except Exception as e:
            self.logger.error(f"Error running parameter sweep: {e}", exc_info=True)
            raise

    def extract_fill_quality(self, backtest_df: pd.DataFrame) -> float:
        """Extract fill quality score (0-1)"""
        return self.results_analyzer.extract_fill_quality(backtest_df)

    def extract_execution_metrics(self, backtest_df: pd.DataFrame) -> Dict:
        """
        Extract comprehensive execution metrics

        Returns:
            Dict with win_rate, avg_pnl, sharpe, sortino, max_dd, avg_hold_days
        """
        try:
            metrics = self.results_analyzer.extract_metrics(backtest_df)

            return {
                "win_rate": metrics.win_rate,
                "loss_rate": metrics.loss_rate,
                "avg_pnl": metrics.avg_pnl,
                "median_pnl": metrics.median_pnl,
                "total_pnl": metrics.total_pnl,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "avg_hold_days": metrics.avg_hold_days,
                "total_trades": metrics.total_trades,
                "total_winners": metrics.total_winners,
                "total_losers": metrics.total_losers,
                "profit_factor": metrics.profit_factor,
                "avg_fill_slippage": metrics.avg_fill_slippage,
                "execution_quality": metrics.execution_quality,
            }

        except Exception as e:
            self.logger.error(f"Error extracting metrics: {e}")
            return {}

    def select_best_fill_model(self, results_df: pd.DataFrame) -> str:
        """
        Select best fill model based on backtest results

        Args:
            results_df: Parameter sweep results

        Returns:
            "MidPrice", "VolumeAwareFill", or "MarketAtBidAsk"
        """
        try:
            # Sort by Sharpe ratio (higher is better)
            best_result = results_df.nlargest(1, "sharpe").iloc[0]

            # Simple heuristic based on metrics
            if best_result["execution_quality"] > 0.97:
                return "MidPrice"
            elif best_result["execution_quality"] > 0.94:
                return "VolumeAwareFill"
            else:
                return "MarketAtBidAsk"

        except Exception as e:
            self.logger.warning(f"Error selecting fill model: {e}")
            return "MidPrice"  # Default

    def select_best_signal_selector(self, results_df: pd.DataFrame) -> str:
        """
        Select best signal selector based on results

        Args:
            results_df: Parameter sweep results

        Returns:
            "NearestDelta" or "MaxOpenInterest"
        """
        try:
            # For now, use NearestDelta as default
            # Could enhance with heuristic based on OI distribution
            return "NearestDelta"

        except Exception as e:
            self.logger.warning(f"Error selecting signal selector: {e}")
            return "NearestDelta"

    def get_backtest_history(self, limit: int = 100) -> List[Dict]:
        """Get recent backtest history"""
        return self.backtest_history[-limit:]

    def get_status(self) -> Dict:
        """Get integration status"""
        return {
            "last_options_df": self.last_options_df is not None,
            "last_engine": self.last_engine is not None,
            "backtest_count": len(self.backtest_history),
            "recent_backtests": self.get_backtest_history(5),
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_options_chain(
    options_chain_response: Dict,
    required_fields: List[str] = None,
) -> bool:
    """Validate options chain response structure"""
    if required_fields is None:
        required_fields = ["strike", "expiry", "option_type", "bid", "ask", "iv"]

    try:
        if "data" not in options_chain_response:
            return False

        if "options" not in options_chain_response["data"]:
            return False

        options = options_chain_response["data"]["options"]
        if not options or not isinstance(options, list):
            return False

        # Check first option has required fields
        if not options[0]:
            return False

        for field in required_fields:
            if field not in options[0]:
                return False

        return True

    except Exception:
        return False


def merge_backtest_results(
    results_list: List[pd.DataFrame],
) -> pd.DataFrame:
    """Merge multiple backtest results"""
    try:
        if not results_list:
            return pd.DataFrame()

        return pd.concat(results_list, ignore_index=True)

    except Exception as e:
        print(f"Error merging backtest results: {e}")
        return pd.DataFrame()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main() -> None:
    """Main entry point for testing"""
    load_dotenv()

    try:
        config = AITraderConfig()
        integration = BacktesterIntegration(config)

        print(f"BacktesterIntegration initialized: {integration.get_status()}")

    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import time
    asyncio.run(main())
