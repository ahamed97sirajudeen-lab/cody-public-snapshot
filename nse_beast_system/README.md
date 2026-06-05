# NSE Beast System - Automated Options Trading Engine

> **A complete, production-grade automated options trading engine for NSE (National Stock Exchange) using Zerodha KiteConnect API. Built with async Python for high-performance real-time trading.**

![Status](https://img.shields.io/badge/status-beta-yellow)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Zerodha](https://img.shields.io/badge/zerodha-kiteconnect-blue)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Authentication](#authentication)
8. [Quick Start](#quick-start)
9. [Kill-Switch Usage](#kill-switch-usage)
10. [Logging & Monitoring](#logging--monitoring)
11. [Module Reference](#module-reference)
12. [Risk Warnings](#risk-warnings)
13. [Troubleshooting](#troubleshooting)
14. [Development](#development)
15. [Contributing](#contributing)
16. [License](#license)

---

## 🎯 Overview

The **NSE Beast System** is a sophisticated automated options trading platform designed for NSE traders. It combines multiple trading strategies with intelligent risk management, real-time monitoring, and async event-driven architecture for maximum performance.

### Key Capabilities

- **Multi-Leg Strategy Execution** - Execute complex option spreads (Iron Condor, Straddle, Strangle, etc.)
- **Volatility Risk Premium Trading** - Identify premium selling opportunities using VRP analysis
- **Intelligent Regime Detection** - Trade according to market conditions (trending, ranging, volatile)
- **Real-Time Signal Generation** - Generate entry signals based on technical and volatility analysis
- **Automated Risk Management** - Kill-switch support, daily loss limits, position sizing
- **Async Event-Driven** - Handle thousands of events per second without blocking
- **Comprehensive Monitoring** - JSON telemetry, Prometheus metrics, real-time dashboards
- **Graceful Error Recovery** - Automatic reconnection, position recovery, error resilience

---

## 🏗️ Architecture

### System Components

The Beast System is composed of 8 core modules orchestrated by the main BeastEngine:

```
┌─────────────────────────────────────────────────────────────────┐
│                     NSE BEAST SYSTEM ARCHITECTURE                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           BEAST ENGINE ORCHESTRATOR                       │   │
│  │  (Async event loop, task coordination, lifecycle mgmt)    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                       │
│         ┌─────────────────┼─────────────────┐                   │
│         ▼                 ▼                 ▼                   │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐     │
│  │  SIGNAL LOOP   │ │ POSITION LOOP  │ │  EXIT LOOP     │     │
│  │                │ │                │ │                │     │
│  │ • Generate     │ │ • Manage P&L   │ │ • Check exits  │     │
│  │ • Validate     │ │ • Rebalance    │ │ • Close trades │     │
│  │ • Execute      │ │ • Monitor      │ │ • Risk limits  │     │
│  └────────────────┘ └────────────────┘ └────────────────┘     │
│         │                 │                 │                   │
└─────────┼─────────────────┼─────────────────┼─────────────────┘
          │                 │                 │
    ┌─────▼──────────────────▼─────────────────▼─────┐
    │                                                  │
    │         EXECUTION ADAPTER (KiteBrokerAdapter)   │
    │  • Order placement, cancellation, status        │
    │  • Real-time quote streaming                    │
    │  • Account & fund management                    │
    │                                                  │
    └────────────────────┬──────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │    VRP     │  │  REGIME    │  │   SIGNAL   │
    │ CALCULATOR │  │ DETECTOR   │  │ GENERATOR  │
    │            │  │            │  │            │
    │ • IV/HV    │  │ • ADX      │  │ • Entry    │
    │ • Premium  │  │ • Trends   │  │ • Greeks   │
    │ • Scoring  │  │ • Momentum │  │ • Filters  │
    └────────────┘  └────────────┘  └────────────┘
        ▼                ▼                ▼
    ┌──────────────────────────────────────────┐
    │     ZERODHA KITECONNECT API                │
    │  (Quotes, Orders, Account, Positions)     │
    └──────────────────────────────────────────┘
```

### Core Modules

| Module | Responsibility | Key Features |
|--------|-----------------|--------------|
| **BeastEngine** | Main orchestrator | Async coordination, task management, lifecycle control |
| **ZerodhaAuthenticator** | Authentication | Credential validation, session management, token refresh |
| **KiteBrokerAdapter** | Broker integration | Order execution, quote streaming, account management |
| **VRPCalculator** | Volatility analysis | IV/HV calculation, premium scoring, VRP identification |
| **RegimeDetector** | Market state | Trend detection, volatility regime, ADX analysis |
| **SignalGenerator** | Trade signals | Entry signal generation, multi-leg construction |
| **PositionManager** | Position tracking | Portfolio management, P&L calculation, rebalancing |
| **ExitEngine** | Trade exits | Exit signal generation, technical analysis, risk limits |
| **TradeMonitor** | Real-time monitoring | Performance tracking, health checks, telemetry |

---

## ✨ Features

### Trading Capabilities

- ✅ **Multiple Option Strategies**
  - Long/Short Calls & Puts
  - Iron Condor
  - Strangle
  - Straddle
  - Bull/Bear Call Spreads
  - Calendar Spreads

- ✅ **Smart Signal Generation**
  - Volatility-based signals (VRP)
  - Regime-aware trading (trending/ranging/volatile)
  - Technical analysis filters (ADX, MACD, RSI)
  - Multi-timeframe confirmation

- ✅ **Advanced Risk Management**
  - Daily loss/profit limits
  - Position size controls
  - Greeks-based hedging (Delta, Gamma, Vega, Theta)
  - Correlation monitoring
  - Emergency kill-switch

### Performance & Reliability

- ✅ **Async Architecture**
  - Non-blocking event loop
  - High-throughput signal processing
  - Minimal latency overhead

- ✅ **Robust Error Handling**
  - Automatic recovery
  - Multi-level retry logic
  - Graceful degradation
  - Comprehensive error logging

- ✅ **Real-Time Monitoring**
  - JSON telemetry (structured logging)
  - Prometheus metrics
  - Trade analytics dashboard
  - Health checks & alerts

- ✅ **Data Persistence**
  - SQLite/PostgreSQL/MongoDB support
  - Trade history archival
  - Position recovery from crashes
  - Backtest data storage

---

## 📋 Prerequisites

### System Requirements

- **Python**: 3.10 or higher
- **OS**: Linux, macOS, or Windows
- **Memory**: Minimum 2GB RAM (4GB+ recommended)
- **Network**: Stable internet connection (crucial for live trading)

### Required Accounts & APIs

1. **Zerodha Account**
   - Active NSE trading account
   - Derivatives (options) trading enabled
   - API access enabled
   - Visit: https://kite.zerodha.com

2. **Zerodha API Credentials**
   - API Key
   - API Secret
   - Access Token (or ability to generate via OAuth)

### Optional Services

- **PostgreSQL**: For high-volume data storage
- **Redis**: For caching and pub/sub
- **Sentry**: For error tracking
- **Prometheus**: For metrics collection

---

## 🔧 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/nse-beast-system.git
cd nse-beast-system
```

### Step 2: Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify activation
which python  # Should show venv path
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Dependencies

```bash
# Install all required packages
pip install -r nse_beast_system/requirements.txt

# Verify installation
pip list | grep -E "kiteconnect|pandas|numpy|asyncio"
```

### Step 5: Verify Installation

```bash
# Test imports
python -c "import kiteconnect; import pandas; print('✓ All imports successful')"
```

---

## ⚙️ Configuration

### Step 1: Create Environment File

```bash
# Copy template
cp nse_beast_system/.env.example nse_beast_system/.env

# Edit with your credentials
nano nse_beast_system/.env  # or use your favorite editor
```

### Step 2: Essential Configuration

Minimum required variables (in `.env`):

```bash
# ============================================================================
# ZERODHA CREDENTIALS (REQUIRED)
# ============================================================================
ZERODHA_API_KEY=your_api_key_here
ZERODHA_API_SECRET=your_api_secret_here
ZERODHA_ACCESS_TOKEN=your_access_token_here
ZERODHA_USER_ID=your_user_id_here

# ============================================================================
# ENGINE SETTINGS
# ============================================================================
ENABLE_LIVE_TRADING=false          # Start with false for testing
ENGINE_MODE=paper                  # paper, backtest, or live
TRADING_TIMEZONE=Asia/Kolkata

# ============================================================================
# RISK MANAGEMENT (CRITICAL)
# ============================================================================
MAX_POSITIONS=5
MAX_DAILY_LOSS=-5000              # Stop trading if loss exceeds this
MAX_DAILY_PROFIT=50000            # Close all if profit exceeds this
KILL_SWITCH_LOSS_THRESHOLD=-10000

# ============================================================================
# VRP & REGIME SETTINGS
# ============================================================================
MIN_VRP_SCORE=0.60
MIN_REGIME_CONFIDENCE=0.65

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL=INFO
ENABLE_JSON_TELEMETRY=true
```

### Step 3: Advanced Configuration

Customize trading behavior:

```bash
# Risk Management
STOP_LOSS_PERCENTAGE=-2.0          # 2% stop loss
TAKE_PROFIT_PERCENTAGE=3.0         # 3% take profit
POSITION_TIMEOUT_HOURS=4           # Close after 4 hours

# Signal Generation
SIGNAL_EXPIRY_MINUTES=5
PREFERRED_STRATEGIES=short_call,short_put,iron_condor
STRIKE_SELECTION_PREFERENCE=OTM

# Execution
ORDER_TIMEOUT_SECONDS=30
ORDER_RETRY_ATTEMPTS=3

# Monitoring
HEARTBEAT_INTERVAL_SECONDS=5
METRICS_UPDATE_INTERVAL_SECONDS=10
```

### Step 4: Database Setup (Optional)

```bash
# For PostgreSQL
POSTGRESQL_URL=postgresql://user:password@localhost:5432/beast_engine

# For MongoDB
MONGODB_URL=mongodb://localhost:27017/beast_engine

# For SQLite (default)
DATABASE_TYPE=sqlite
SQLITE_DB_PATH=data/beast_engine.db
```

---

## 🔐 Authentication

### Overview

Zerodha uses OAuth 2.0 for authentication. The flow requires:

1. **API Key** - Identifies your application
2. **Request Token** - Obtained after user login
3. **Access Token** - Used for API calls
4. **Public Token** - Long-lived token for session recovery

### Getting Your API Credentials

#### Step 1: Create API Application

1. Go to https://kite.zerodha.com/settings/api/applications
2. Click "Create new"
3. Fill in application details:
   - **App Name**: "NSE Beast System"
   - **Redirect URL**: `http://localhost:8080/` (or your callback URL)
   - **Purpose**: "Automated Trading"
4. Click "Create"
5. Copy and save:
   - **API Key** - Keep this safe
   - **API Secret** - Keep this very safe

#### Step 2: Generate Access Token

The system provides an authentication helper. Here's the OAuth flow:

```python
# Method 1: Automatic Flow (Recommended)
from auth.zerodha_auth import ZerodhaAuthenticator

authenticator = ZerodhaAuthenticator(
    api_key="your_api_key",
    api_secret="your_api_secret",
    access_token="your_access_token",  # Leave empty for first auth
    user_id="your_user_id"
)

# If access_token is empty, get login URL
login_url = authenticator.get_login_url()
print(f"Visit: {login_url}")
# User logs in, gets redirected with request_token in URL
# Extract request_token and exchange for access_token

request_token = "extracted_from_redirect"
access_token, public_token = authenticator.get_access_token(request_token)

print(f"Access Token: {access_token}")
print(f"Public Token: {public_token}")  # Save this for recovery
```

#### Step 3: Update .env File

```bash
# After obtaining tokens
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret
ZERODHA_ACCESS_TOKEN=your_access_token
ZERODHA_PUBLIC_TOKEN=your_public_token
ZERODHA_USER_ID=your_user_id
```

### Session Recovery

If you have a saved public token:

```bash
# Next time, use this to recover the session
python -m nse_beast_system.auth.recover_session \
    --api-key YOUR_API_KEY \
    --api-secret YOUR_API_SECRET \
    --public-token YOUR_PUBLIC_TOKEN \
    --user-id YOUR_USER_ID
```

### Manual Token Generation

For detailed manual flow:

```bash
# 1. Get login URL
python -c "
from kiteconnect import KiteConnect
kite = KiteConnect(api_key='YOUR_API_KEY')
print(kite.login_url())
"

# 2. User visits URL and logs in
# 3. Gets redirected with request_token
# 4. Extract request_token and run:

python -c "
from kiteconnect import KiteConnect
kite = KiteConnect(api_key='YOUR_API_KEY')
data = kite.generate_session(request_token='REQUEST_TOKEN', api_secret='YOUR_API_SECRET')
print(f\"Access Token: {data['access_token']}\")
print(f\"Public Token: {data['public_token']}\")
"
```

---

## 🚀 Quick Start

### Running in Paper Trading Mode (Recommended First)

```bash
# Step 1: Ensure .env is configured
cat nse_beast_system/.env | grep ZERODHA_

# Step 2: Run in paper trading mode (no real orders)
ENABLE_LIVE_TRADING=false python -m nse_beast_system.beast_engine

# Output should show:
# ✓ BeastEngine initialization complete
# ✓ All modules initialized successfully
# ✓ Trading loop started
# ✓ Monitoring trade signals...
```

### Running the Engine

#### Method 1: Direct Execution

```bash
# Run with default config
python -m nse_beast_system.beast_engine

# Run with custom config
python -m nse_beast_system.beast_engine --config config/custom_config.json

# Run in debug mode
DEBUG_MODE=true python -m nse_beast_system.beast_engine

# Dry run (no actual orders)
DRY_RUN_MODE=true python -m nse_beast_system.beast_engine
```

#### Method 2: Python Script

```python
import asyncio
from nse_beast_system.beast_engine import BeastEngine
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    engine = BeastEngine(
        api_key=os.getenv("ZERODHA_API_KEY"),
        api_secret=os.getenv("ZERODHA_API_SECRET"),
        access_token=os.getenv("ZERODHA_ACCESS_TOKEN"),
        user_id=os.getenv("ZERODHA_USER_ID"),
        config_path="config/beast_engine_config.json",
        enable_live_trading=False,  # Start with false
    )
    
    try:
        await engine.run()
    except KeyboardInterrupt:
        print("Shutdown signal received")
    finally:
        await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

#### Method 3: Docker (Recommended for Production)

```bash
# Build image
docker build -t nse-beast-system .

# Run container
docker run -d \
  --name beast-engine \
  -e ZERODHA_API_KEY=your_key \
  -e ZERODHA_API_SECRET=your_secret \
  -e ZERODHA_ACCESS_TOKEN=your_token \
  -e ENABLE_LIVE_TRADING=false \
  -v /path/to/logs:/app/logs \
  -v /path/to/data:/app/data \
  nse-beast-system
```

### Monitoring Output

Expected console output:

```
2024-06-05 09:15:23 - BeastEngine - INFO - Initializing NSE Beast System Engine
2024-06-05 09:15:24 - BeastEngine - INFO - Loaded configuration from config/beast_engine_config.json
2024-06-05 09:15:25 - BeastEngine - INFO - Initializing trading modules
2024-06-05 09:15:26 - BeastEngine - INFO - All modules initialized successfully
2024-06-05 09:15:26 - BeastEngine - INFO - Starting Beast Engine
2024-06-05 09:15:27 - BeastEngine - INFO - Trading loop started - Market hours: 09:15-15:30
2024-06-05 09:16:00 - BeastEngine - INFO - Signal generated: iron_condor (VRP: 0.72)
2024-06-05 09:16:01 - BeastEngine - INFO - Executing trade signal: iron_condor
2024-06-05 09:16:02 - BeastEngine - INFO - Order placed: 123456789
2024-06-05 09:16:03 - BeastEngine - INFO - Order placed: 123456790
2024-06-05 09:16:04 - BeastEngine - INFO - Trade TRADE_20240605_091604_123456 executed successfully
```

---

## 🛑 Kill-Switch Usage

### What is the Kill-Switch?

The kill-switch is an emergency mechanism to immediately close all positions and halt trading. It's triggered when:

1. Daily loss exceeds `KILL_SWITCH_LOSS_THRESHOLD`
2. Critical error occurs
3. User sends interrupt signal (Ctrl+C)
4. Health check fails

### Automatic Kill-Switch

The system monitors loss and automatically triggers:

```bash
# In .env
KILL_SWITCH_LOSS_THRESHOLD=-10000  # Closes all if loss > -10000
ENABLE_KILL_SWITCH=true
```

When triggered, the system:
1. Pauses new signal generation
2. Closes all open positions
3. Cancels pending orders
4. Logs detailed telemetry
5. Sends alerts (email/Telegram)

### Manual Kill-Switch

#### Method 1: Keyboard Interrupt (Graceful)

```bash
# Press Ctrl+C to gracefully shutdown
# The system will:
# - Close all positions
# - Cancel pending orders
# - Save telemetry
# - Clean up resources

^C
Shutdown signal received
BeastEngine shutdown complete
```

#### Method 2: Force Kill-Switch via API

```python
# From another terminal/script
import requests

# Trigger emergency exit
response = requests.post("http://localhost:8000/api/emergency-exit")
print(response.json())
# Output: {"status": "success", "message": "All positions closed"}
```

#### Method 3: File-Based Kill-Switch

```bash
# Create a kill-switch file
touch /tmp/beast_engine_kill_switch

# The engine checks this file every second
# When detected, immediately closes all positions and exits

# To disable
rm /tmp/beast_engine_kill_switch
```

### Kill-Switch Configuration

```bash
# .env configuration
ENABLE_KILL_SWITCH=true
KILL_SWITCH_LOSS_THRESHOLD=-10000        # Trigger at -10k loss
MAX_DAILY_LOSS=-5000                     # Secondary limit
MAX_DAILY_PROFIT=50000                   # Close on target profit

# Alert settings
ALERT_EMAIL_RECIPIENTS=your-email@example.com
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Kill-Switch Testing

```bash
# Dry run with kill-switch testing
DRY_RUN_MODE=true KILL_SWITCH_LOSS_THRESHOLD=-1 python -m nse_beast_system.beast_engine

# This will trigger kill-switch immediately for testing
```

---

## 📊 Logging & Monitoring

### Logging Structure

The system uses multi-level structured JSON logging:

```
logs/
├── beast_engine.json          # Main engine logs (JSON)
├── telemetry.json             # Trade telemetry (JSON)
└── console.log                # Console output
```

### Log Levels

```bash
# In .env
LOG_LEVEL=DEBUG    # Most verbose: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Structured JSON Logs

Each log entry includes:

```json
{
  "timestamp": "2024-06-05T09:15:26.123456Z",
  "level": "INFO",
  "logger": "BeastEngine",
  "message": "Trade executed successfully",
  "module": "beast_engine",
  "function": "_execute_signal",
  "line": 256,
  "extra_data": {
    "trade_id": "TRADE_20240605_091526_123456",
    "signal_type": "iron_condor",
    "vrp_score": 0.72,
    "entry_price": 125.50
  }
}
```

### Trade Telemetry

Each completed trade logs comprehensive data:

```json
{
  "timestamp": "2024-06-05T09:30:00Z",
  "trade_id": "TRADE_20240605_093000_123456",
  "signal_type": "iron_condor",
  "entry_price": 125.50,
  "entry_quantity": 100,
  "entry_leg_count": 4,
  "regime": "ranging",
  "vrp_score": 0.72,
  "market_condition": "normal",
  "entry_order_ids": ["123", "124", "125", "126"],
  "status": "closed",
  "pnl": 2500.0,
  "pnl_percentage": 2.5,
  "max_pnl": 3500.0,
  "exit_reason": "take_profit",
  "exit_timestamp": "2024-06-05T10:15:00Z",
  "execution_latency_ms": 45.2,
  "metadata": {
    "signal_id": "SIG_123456",
    "confidence": 0.85
  }
}
```

### Real-Time Metrics

Access current metrics:

```bash
# View metrics via API
curl http://localhost:8000/api/metrics

# Output:
{
  "total_trades": 45,
  "active_trades": 3,
  "closed_trades": 42,
  "total_pnl": 15250.0,
  "win_rate": 78.5,
  "avg_pnl_per_trade": 340.0,
  "max_drawdown": -12.5,
  "sharpe_ratio": 1.85,
  "consecutive_wins": 8,
  "consecutive_losses": 2,
  "signal_count": 147,
  "execution_success_rate": 96.2
}
```

### Monitoring Dashboard

Start the monitoring dashboard:

```bash
# Start Prometheus monitoring
python -m nse_beast_system.monitoring.prometheus_exporter

# Start Grafana (if installed)
docker run -d -p 3000:3000 grafana/grafana

# Access dashboard at http://localhost:3000
# Add Prometheus data source: http://localhost:9090
# Import Beast System dashboard
```

### Alert Configuration

Configure alerts via:

```bash
# Email alerts
ALERT_EMAIL_RECIPIENTS=your-email@example.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password

# Telegram alerts
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
TELEGRAM_CHAT_ID=1234567890

# Slack alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### Health Checks

View system health:

```bash
# API endpoint
curl http://localhost:8000/api/health

# Output:
{
  "broker_connected": true,
  "websocket_connected": true,
  "data_feed_active": true,
  "auth_valid": true,
  "last_heartbeat": "2024-06-05T10:15:26Z",
  "error_count": 0,
  "recovery_count": 0,
  "uptime_seconds": 3600.5,
  "api_call_count": 1250,
  "api_error_count": 3
}
```

---

## 📚 Module Reference

### Detailed Module Descriptions

#### 1. BeastEngine (beast_engine.py)

**Purpose**: Main orchestrator and event loop manager

**Key Methods**:
- `__init__()` - Initialize with credentials and config
- `run()` - Start the async event loop
- `_run_loop()` - Main trading cycle
- `_process_signals()` - Generate and execute trades
- `_manage_positions()` - Update P&L and risk
- `_check_exits()` - Evaluate exit conditions
- `shutdown()` - Graceful cleanup

**Config**:
```python
{
  "trading": {
    "max_positions": 5,
    "max_daily_loss": -5000,
    "position_timeout_hours": 4
  },
  "monitoring": {
    "heartbeat_interval_seconds": 5,
    "metrics_update_interval_seconds": 10
  }
}
```

#### 2. ZerodhaAuthenticator (auth/zerodha_auth.py)

**Purpose**: Handle Zerodha OAuth authentication

**Key Methods**:
- `validate_session()` - Check if access token is valid
- `get_login_url()` - Generate user login URL
- `get_access_token()` - Exchange request token for access token
- `refresh_session()` - Refresh expired tokens

**Usage**:
```python
from auth.zerodha_auth import ZerodhaAuthenticator

auth = ZerodhaAuthenticator(
    api_key="key",
    api_secret="secret",
    access_token="token",
    user_id="user"
)
is_valid = await auth.validate_session()
```

#### 3. KiteBrokerAdapter (adapters/kite_broker_adapter.py)

**Purpose**: Bridge to Zerodha KiteConnect API

**Key Methods**:
- `place_order()` - Execute a single order
- `cancel_order()` - Cancel an order
- `get_positions()` - Get all open positions
- `get_funds()` - Get account funds
- `get_quotes()` - Stream real-time quotes
- `cancel_all_orders()` - Emergency close all

**Supported Order Types**:
- MARKET
- LIMIT
- STOP_LOSS
- STOP_LOSS_MARKET

**Example**:
```python
from adapters.kite_broker_adapter import KiteBrokerAdapter

order_id = await adapter.place_order(
    symbol="NIFTY23JUN21000CE",
    side="BUY",
    quantity=1,
    order_type="MARKET"
)
```

#### 4. VRPCalculator (vrp/vrp_calculator.py)

**Purpose**: Calculate Volatility Risk Premium

**Key Methods**:
- `calculate_iv()` - Implied volatility from option prices
- `calculate_hv()` - Historical volatility from stock prices
- `calculate_vrp()` - Premium between IV and HV
- `score_vrp()` - Generate 0-1 score for trading

**Formula**:
```
VRP = (IV - HV) / IV
VRP Score = min(1.0, VRP * 2)  # Normalized to 0-1
```

**Configuration**:
```bash
VRP_CALCULATION_METHOD=blended    # historical, implied, or blended
VRP_HISTORICAL_LOOKBACK=30        # Days for HV calculation
VRP_IV_SOURCE=exchange            # exchange or model
MIN_VRP_SCORE=0.60                # Minimum for trading
```

#### 5. RegimeDetector (regime/regime_detector.py)

**Purpose**: Identify market regime (trending/ranging/volatile)

**Key Methods**:
- `detect_regime()` - Classify current market state
- `get_trend_strength()` - ADX-based trend strength
- `get_regime_confidence()` - Confidence level 0-1

**Regimes Detected**:
- `trending_up` - Strong uptrend (ADX > 25, +DI > -DI)
- `trending_down` - Strong downtrend (ADX > 25, -DI > +DI)
- `ranging` - No clear trend (ADX < 20)
- `volatile` - High volatility, unclear direction

**Configuration**:
```bash
REGIME_DETECTION_METHOD=adx       # sma, ema, adx, macd
REGIME_LOOKBACK_PERIOD=20         # Bars to analyze
ADX_STRONG_TREND_THRESHOLD=25
ADX_WEAK_TREND_THRESHOLD=20
```

#### 6. SignalGenerator (signals/signal_generator.py)

**Purpose**: Generate entry signals based on VRP and regime

**Key Methods**:
- `generate_signals()` - Create list of signals
- `validate_signal()` - Check signal quality
- `construct_legs()` - Build multi-leg strategy

**Signal Types**:
- `long_call` - Bullish call purchase
- `long_put` - Bearish put purchase
- `short_call` - Bullish call sale
- `short_put` - Bearish put sale
- `iron_condor` - 4-leg neutral strategy
- `strangle` - 2-leg directional bet
- `straddle` - 2-leg volatility bet

**Example Signal Object**:
```python
Signal(
    signal_type="iron_condor",
    entry_price=125.50,
    legs=[
        Leg(symbol="NIFTY23JUN21000CE", side="SELL", quantity=1),
        Leg(symbol="NIFTY23JUN21200CE", side="BUY", quantity=1),
        Leg(symbol="NIFTY23JUN20800PE", side="BUY", quantity=1),
        Leg(symbol="NIFTY23JUN21000PE", side="SELL", quantity=1),
    ],
    vrp_score=0.72,
    regime="ranging",
    confidence=0.85
)
```

#### 7. PositionManager (positions/position_manager.py)

**Purpose**: Track and manage open positions

**Key Methods**:
- `load_positions()` - Load from broker
- `get_positions()` - Get all positions
- `update_position_pnl()` - Calculate P&L
- `calculate_greeks()` - Compute Greeks
- `rebalance()` - Adjust for correlation

**Greeks Calculated**:
- Delta: Price sensitivity
- Gamma: Delta sensitivity
- Vega: Volatility sensitivity
- Theta: Time decay

**Configuration**:
```bash
REBALANCING_FREQUENCY_MINUTES=30
MAX_POSITION_CORRELATION=0.7      # Max correlation between positions
DELTA_HEDGE_TARGET=0.3
GAMMA_MANAGEMENT_THRESHOLD=0.5
```

#### 8. ExitEngine (exits/exit_engine.py)

**Purpose**: Determine when to close positions

**Key Methods**:
- `check_exit()` - Evaluate exit conditions
- `get_technical_exit()` - Based on technical indicators
- `get_risk_exit()` - Based on risk limits
- `get_time_exit()` - Based on elapsed time

**Exit Triggers**:
- Stop-loss: When loss exceeds threshold
- Take-profit: When profit target reached
- Time-based: Position held too long
- Technical: Regime change detected
- Correlation: Position correlation increases

**Configuration**:
```bash
STOP_LOSS_PERCENTAGE=-2.0
TAKE_PROFIT_PERCENTAGE=3.0
EXIT_AFTER_HOURS=4                # Hours to hold max
EXIT_VOLATILITY_THRESHOLD=30      # Exit if vol drops
```

#### 9. TradeMonitor (monitor/trade_monitor.py)

**Purpose**: Real-time monitoring and health checks

**Key Methods**:
- `monitor_trades()` - Track all open trades
- `is_feed_active()` - Check data feed health
- `calculate_metrics()` - Compute performance stats
- `check_health()` - System health validation
- `get_telemetry()` - Export telemetry data

**Metrics Tracked**:
- Total trades, wins, losses
- Consecutive wins/losses
- Win rate, Sharpe ratio, max drawdown
- Trade PnL distribution
- Execution latency

---

## ⚠️ Risk Warnings

### CRITICAL WARNINGS

⚠️ **LIVE TRADING RISK**

1. **Capital Loss Risk**
   - This system can lose money
   - Options are highly leveraged
   - You can lose more than your investment
   - Use only capital you can afford to lose

2. **Technology Risk**
   - Network failures can cause delayed exits
   - API failures may prevent order placement
   - System bugs could cause unintended trades
   - Always have manual kill-switch ready

3. **Market Risk**
   - Market gaps can bypass stop-loss
   - Liquidity may be insufficient to exit
   - Volatility can spike unexpectedly
   - Black swan events cannot be predicted

### Best Practices

✅ **Before Going Live**

- [ ] Test thoroughly in paper trading mode (minimum 2 weeks)
- [ ] Verify all .env variables are correct
- [ ] Test kill-switch mechanism manually
- [ ] Monitor for 1 hour before deploying
- [ ] Start with small position sizes
- [ ] Have backup exit plan

✅ **While Running**

- [ ] Monitor system health daily
- [ ] Check logs regularly for errors
- [ ] Verify positions match expectations
- [ ] Have manual kill-switch button ready
- [ ] Never leave unattended for long periods
- [ ] Keep API credentials secure

✅ **Risk Controls**

```bash
# Recommended risk parameters
MAX_POSITIONS=3                    # Not more than 3 concurrent
MAX_DAILY_LOSS=-3000              # Stop at -3000
MAX_DAILY_PROFIT=10000            # Take profits at +10000
STOP_LOSS_PERCENTAGE=-2.0         # 2% stop loss
TAKE_PROFIT_PERCENTAGE=2.5        # 2.5% take profit
POSITION_TIMEOUT_HOURS=2          # Close after 2 hours
```

### Liability Disclaimer

**THIS SOFTWARE IS PROVIDED "AS-IS" WITHOUT ANY WARRANTIES**

- The authors assume no responsibility for trading losses
- Past performance does not guarantee future results
- Use at your own risk with proper risk management
- Consult a financial advisor before trading
- Test extensively before deploying real capital
- Keep updated with latest market regulations

---

## 🔧 Troubleshooting

### Common Issues

#### Issue: "Authentication failed"

**Solution**:
```bash
# Check credentials in .env
grep ZERODHA_ nse_beast_system/.env

# Regenerate access token
python -m nse_beast_system.auth.get_token \
    --api-key YOUR_API_KEY \
    --api-secret YOUR_API_SECRET

# Verify token is valid
curl -H "X-Kite-Version: 3" \
    -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
    https://api.kite.trade/profile
```

#### Issue: "Connection timeout"

**Solution**:
```bash
# Check network connectivity
ping api.kite.trade

# Check firewall rules
sudo ufw allow 443

# Increase timeout in config
ORDER_TIMEOUT_SECONDS=60

# Check Zerodha API status
# https://status.zerodha.com/
```

#### Issue: "Order placement failed"

**Solution**:
```bash
# Verify account has day trading buying power
# Check margin availability via:
curl -H "Authorization: Bearer YOUR_TOKEN" \
    https://api.kite.trade/user/margins

# Verify symbol is active
curl -H "Authorization: Bearer YOUR_TOKEN" \
    "https://api.kite.trade/quote/NSE:NIFTY50"

# Check option expiry is valid
# Zerodha options expire on Thursday

# Enable OPTIONS trading in Zerodha settings
```

#### Issue: "No signals generated"

**Solution**:
```bash
# Check VRP score is high enough
DEBUG_MODE=true python -m nse_beast_system.beast_engine
# Look for: "VRP score too low: 0.45"

# Lower VRP threshold temporarily
MIN_VRP_SCORE=0.50

# Check regime is detected
# Look for: "Regime: ranging, confidence: 0.85"

# Check minimum regime confidence
MIN_REGIME_CONFIDENCE=0.60

# Verify signal generation is enabled
grep "ENABLE_OPTIONS_TRADING\|ENABLE_SHORT_SELLING" nse_beast_system/.env
```

#### Issue: "High latency (orders slow)"

**Solution**:
```bash
# Check system resources
top -p $(pgrep -f beast_engine)
# Look for: CPU < 50%, Memory < 1GB

# Enable performance optimization
USE_NUMBA=true
CACHE_STRATEGY_CALCULATIONS=true
PARALLEL_PROCESSING_ENABLED=true

# Reduce monitoring frequency
METRICS_UPDATE_INTERVAL_SECONDS=30  # Instead of 10

# Run on faster machine or VPS
```

#### Issue: "Database connection errors"

**Solution**:
```bash
# For SQLite
ls -la data/beast_engine.db
chmod 666 data/beast_engine.db

# For PostgreSQL
psql -h localhost -U user -d beast_engine -c "SELECT 1"

# Reset database
rm -f data/beast_engine.db
python -m nse_beast_system.db.init_db
```

### Debug Mode

```bash
# Enable verbose logging
DEBUG_MODE=true LOG_LEVEL=DEBUG python -m nse_beast_system.beast_engine

# Log all API requests
LOG_API_REQUESTS=true python -m nse_beast_system.beast_engine

# Simulation mode (no actual orders)
SIMULATION_MODE=true python -m nse_beast_system.beast_engine

# Dry run (connect but don't trade)
DRY_RUN_MODE=true python -m nse_beast_system.beast_engine
```

### Getting Help

1. **Check Logs**
   ```bash
   tail -f logs/beast_engine.json
   tail -f logs/telemetry.json
   ```

2. **Enable Debug Mode**
   ```bash
   DEBUG_MODE=true python -m nse_beast_system.beast_engine
   ```

3. **Check Zerodha Status**
   - https://status.zerodha.com/
   - Verify API is not under maintenance

4. **Review Configuration**
   ```bash
   python -c "
   import json
   with open('config/beast_engine_config.json') as f:
       config = json.load(f)
       print(json.dumps(config, indent=2))
   "
   ```

5. **Test Connectivity**
   ```bash
   python -m nse_beast_system.test_connectivity
   ```

---

## 👨‍💻 Development

### Project Structure

```
nse_beast_system/
├── beast_engine.py              # Main orchestrator
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
├── auth/
│   ├── __init__.py
│   └── zerodha_auth.py         # Authentication
├── adapters/
│   ├── __init__.py
│   └── kite_broker_adapter.py  # Broker integration
├── vrp/
│   ├── __init__.py
│   └── vrp_calculator.py       # Volatility analysis
├── regime/
│   ├── __init__.py
│   └── regime_detector.py      # Market state detection
├── signals/
│   ├── __init__.py
│   └── signal_generator.py     # Signal generation
├── positions/
│   ├── __init__.py
│   └── position_manager.py     # Position tracking
├── exits/
│   ├── __init__.py
│   └── exit_engine.py          # Exit logic
├── monitor/
│   ├── __init__.py
│   └── trade_monitor.py        # Monitoring
├── tests/
│   ├── test_beast_engine.py
│   ├── test_signals.py
│   └── test_execution.py
└── config/
    └── beast_engine_config.json # Configuration
```

### Running Tests

```bash
# Run all tests
pytest nse_beast_system/tests/ -v

# Run specific test
pytest nse_beast_system/tests/test_signals.py::test_signal_generation -v

# Run with coverage
pytest --cov=nse_beast_system nse_beast_system/tests/

# Run only unit tests (no integration)
pytest -m "not integration" -v

# Run only async tests
pytest -k "async" -v
```

### Code Quality

```bash
# Format code
black nse_beast_system/

# Lint
flake8 nse_beast_system/

# Type check
mypy nse_beast_system/

# Import sorting
isort nse_beast_system/

# Full quality check
black --check nse_beast_system/ && flake8 nse_beast_system/ && mypy nse_beast_system/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **Zerodha** for the excellent KiteConnect API
- **NSE** for providing derivatives trading
- The Python trading community for inspiration

---

## 📞 Support

- 📧 **Email**: support@example.com
- 💬 **Discord**: [Join Server](https://discord.gg/example)
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/nse-beast-system/issues)
- 📚 **Docs**: [Full Documentation](https://nse-beast-system.readthedocs.io)

---

## 📊 Recent Updates

### Version 1.0.0 (June 2024)
- ✅ Initial release
- ✅ Multi-leg strategy execution
- ✅ VRP-based signal generation
- ✅ Comprehensive risk management
- ✅ Real-time monitoring
- ✅ JSON telemetry logging

### Roadmap

- [ ] Machine learning predictions
- [ ] Advanced Greeks hedging
- [ ] Multi-broker support
- [ ] REST API for external integration
- [ ] Web dashboard
- [ ] Mobile alerts

---

## ⭐ Show Your Support

If this project helped you, consider:
- ⭐ Starring this repository
- 🐛 Reporting bugs
- 💡 Suggesting features
- 📢 Sharing with others
- 💰 Sponsoring development

---

**Made with ❤️ by the NSE Beast System Team**

*Last Updated: June 5, 2024*
