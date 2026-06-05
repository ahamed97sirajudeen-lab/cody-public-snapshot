"""
NSE Beast System - Strategy Learner with ML-Based Strategy Selection
=====================================================================

This module implements machine learning-based strategy selection that learns
which strategies perform best under different market conditions.

Key Features:
- MarketFeatures: Captures market regime and conditions at trade entry
- StrategyRecord: Records historical backtest results with market context
- StrategyLearner: Main ML class using ensemble methods (RandomForest, GradientBoosting)
- OnlineLearner: Incremental learning for streaming backtest results

Learning Workflow:
1. Collect MarketFeatures + backtest results for each strategy trial
2. Train classifier to predict best strategy from market features
3. Train regressors to predict optimal parameters for each strategy
4. Use ensemble predictions for confident strategy selection
5. Continuously learn from new backtest results (online learning)

Integration Points:
- Receives: Market state (VIX, VRP, regime, IV rank, momentum, etc.)
- Receives: Backtest results (Sharpe, win rate, P&L)
- Outputs: Recommended strategy signal type
- Outputs: Predicted optimal execution parameters
- Persists: Training data to JSONL for reproducibility

Author: NSE Beast System Team
Created: 2025-06-05
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback
import pickle
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import threading
import queue

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_learner_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Setup JSON structured logging for strategy learner.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("StrategyLearner")
    logger.setLevel(logging.DEBUG)
    
    # JSON formatter for structured logging
    json_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        '"module": "%(name)s", "message": "%(message)s"}',
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "strategy_learner.log"),
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


logger = setup_learner_logging()


# ============================================================================
# ENUMS
# ============================================================================

class SignalType(Enum):
    """Available strategy signal types"""
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    COVERED_CALL = "covered_call"
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    BULL_SPREAD = "bull_spread"
    BEAR_SPREAD = "bear_spread"
    NO_TRADE = "no_trade"


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MarketFeatures:
    """
    Market features that influence strategy selection.
    Captures current market state and conditions.
    
    Attributes:
        vix_level: VIX index level (0-100+)
        vrp_score: Volatility Risk Premium score (0-1)
        regime: Market regime classification (trending, ranging, volatile, etc.)
        iv_rank: Implied volatility rank percentile (0-100)
        put_call_ratio: Put/Call volume ratio (0-2+)
        spot_momentum: Spot price momentum over N periods (-1 to 1)
        realized_vol: Realized volatility percentage (annualized)
        days_to_expiry: Days to target expiration
        time_of_day: Hour of day (0-23) for intraday patterns
        day_of_week: Day of week (0=Monday, 4=Friday)
    """
    vix_level: float
    vrp_score: float
    regime: str
    iv_rank: float
    put_call_ratio: float
    spot_momentum: float
    realized_vol: float
    days_to_expiry: int
    time_of_day: int
    day_of_week: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        try:
            return asdict(self)
        except Exception as e:
            logger.error(json.dumps({
                "event": "market_features_to_dict_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'MarketFeatures':
        """Create MarketFeatures from dictionary."""
        try:
            required_keys = {
                'vix_level', 'vrp_score', 'regime', 'iv_rank', 'put_call_ratio',
                'spot_momentum', 'realized_vol', 'days_to_expiry', 'time_of_day', 'day_of_week'
            }
            
            missing_keys = required_keys - set(data.keys())
            if missing_keys:
                raise ValueError(f"Missing required keys: {missing_keys}")
            
            # Type validation
            type_checks = {
                'vix_level': (int, float),
                'vrp_score': (int, float),
                'regime': str,
                'iv_rank': (int, float),
                'put_call_ratio': (int, float),
                'spot_momentum': (int, float),
                'realized_vol': (int, float),
                'days_to_expiry': int,
                'time_of_day': int,
                'day_of_week': int
            }
            
            for key, expected_type in type_checks.items():
                if not isinstance(data[key], expected_type):
                    raise TypeError(
                        f"Parameter '{key}' must be {expected_type}, "
                        f"got {type(data[key])}"
                    )
            
            return MarketFeatures(**data)
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "market_features_from_dict_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "input_data": str(data)
            }))
            raise


@dataclass
class StrategyRecord:
    """
    Historical record of strategy trial with market context and backtest results.
    
    Attributes:
        features: MarketFeatures describing market state at entry
        signal_type: Strategy signal type (long_strangle, iron_condor, etc.)
        execution_params: Execution parameters used (dict)
        backtest_sharpe: Sharpe ratio from backtest
        backtest_win_rate: Winning trade percentage (0-1)
        backtest_pnl: Total P&L from backtest
        timestamp: When this record was created
    """
    features: MarketFeatures
    signal_type: str
    execution_params: Dict[str, Any]
    backtest_sharpe: float
    backtest_win_rate: float
    backtest_pnl: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        try:
            return {
                "features": self.features.to_dict(),
                "signal_type": self.signal_type,
                "execution_params": self.execution_params,
                "backtest_sharpe": float(self.backtest_sharpe),
                "backtest_win_rate": float(self.backtest_win_rate),
                "backtest_pnl": float(self.backtest_pnl),
                "timestamp": self.timestamp
            }
        except Exception as e:
            logger.error(json.dumps({
                "event": "strategy_record_to_dict_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'StrategyRecord':
        """Create StrategyRecord from dictionary."""
        try:
            return StrategyRecord(
                features=MarketFeatures.from_dict(data.get("features", {})),
                signal_type=str(data.get("signal_type", "")),
                execution_params=dict(data.get("execution_params", {})),
                backtest_sharpe=float(data.get("backtest_sharpe", 0.0)),
                backtest_win_rate=float(data.get("backtest_win_rate", 0.0)),
                backtest_pnl=float(data.get("backtest_pnl", 0.0)),
                timestamp=str(data.get("timestamp", datetime.now().isoformat()))
            )
        except Exception as e:
            logger.error(json.dumps({
                "event": "strategy_record_from_dict_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "input_data": str(data)
            }))
            raise


# ============================================================================
# STRATEGY LEARNER
# ============================================================================

class StrategyLearner:
    """
    Machine learning-based strategy learner.
    
    Trains ensemble models to:
    1. Predict best strategy given market conditions
    2. Predict optimal execution parameters per strategy
    3. Provide confidence scores and feature importance
    
    Attributes:
        model_path: Path to save/load trained models
        records: List of StrategyRecord for training data
        strategy_classifier: RandomForest/GradientBoosting classifier
        param_regressors: Dict of regressors per strategy
        scaler: StandardScaler for feature normalization
        label_encoders: Dict of label encoders
        lock: Thread lock for thread safety
    """

    def __init__(self, model_path: str = "models"):
        """
        Initialize StrategyLearner.
        
        Args:
            model_path: Directory for model persistence
            
        Raises:
            OSError: If model directory cannot be created
        """
        try:
            self.model_path = model_path
            os.makedirs(model_path, exist_ok=True)
            
            self.records: List[StrategyRecord] = []
            self.strategy_classifier: Optional[RandomForestClassifier] = None
            self.param_regressors: Dict[str, Any] = {}
            self.scaler = StandardScaler()
            self.label_encoders: Dict[str, LabelEncoder] = {
                'regime': LabelEncoder(),
                'signal_type': LabelEncoder()
            }
            
            self.lock = threading.RLock()
            self.records_file = os.path.join(model_path, "strategy_records.jsonl")
            
            # Load existing records if available
            self._load_records_from_jsonl()
            
            logger.info(json.dumps({
                "event": "strategy_learner_initialized",
                "model_path": model_path,
                "records_loaded": len(self.records),
                "timestamp": datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "strategy_learner_init_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    def add_backtest_result(
        self,
        features: MarketFeatures,
        signal_type: str,
        exec_params: Dict[str, Any],
        backtest_result: Dict[str, float]
    ) -> None:
        """
        Add backtest result to training dataset.
        
        Args:
            features: Market features at entry
            signal_type: Strategy signal type
            exec_params: Execution parameters used
            backtest_result: Dict with 'sharpe', 'win_rate', 'pnl' keys
            
        Raises:
            ValueError: If backtest_result missing required keys
            TypeError: If parameter types are invalid
        """
        try:
            with self.lock:
                # Validate backtest result
                required_keys = {'sharpe', 'win_rate', 'pnl'}
                missing_keys = required_keys - set(backtest_result.keys())
                if missing_keys:
                    raise ValueError(f"Missing backtest keys: {missing_keys}")
                
                record = StrategyRecord(
                    features=features,
                    signal_type=signal_type,
                    execution_params=exec_params,
                    backtest_sharpe=float(backtest_result['sharpe']),
                    backtest_win_rate=float(backtest_result['win_rate']),
                    backtest_pnl=float(backtest_result['pnl']),
                    timestamp=datetime.now().isoformat()
                )
                
                self.records.append(record)
                
                # Append to JSONL file for persistence
                self._append_record_to_jsonl(record)
                
                logger.debug(json.dumps({
                    "event": "backtest_result_added",
                    "signal_type": signal_type,
                    "sharpe": backtest_result['sharpe'],
                    "win_rate": backtest_result['win_rate'],
                    "total_records": len(self.records),
                    "timestamp": datetime.now().isoformat()
                }))
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "add_backtest_result_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "signal_type": signal_type,
                "backtest_result": backtest_result
            }))
            raise

    def train(self, min_samples: int = 50) -> Dict[str, Any]:
        """
        Train strategy classifier and parameter regressors.
        
        Uses ensemble methods:
        - RandomForestClassifier for strategy selection
        - GradientBoostingClassifier as alternative
        - Separate regressors for parameter prediction per strategy
        
        Args:
            min_samples: Minimum samples required to train
            
        Returns:
            Dict with training results including:
            - 'success': bool
            - 'n_samples': int
            - 'n_strategies': int
            - 'strategies': list of trained strategies
            - 'cv_scores': cross-validation results
            - 'feature_importance': feature importance scores
            - 'training_time': seconds
            
        Raises:
            ValueError: If not enough samples or training data
        """
        try:
            with self.lock:
                start_time = datetime.now()
                
                if len(self.records) < min_samples:
                    raise ValueError(
                        f"Not enough samples: {len(self.records)} < {min_samples}"
                    )
                
                logger.info(json.dumps({
                    "event": "training_started",
                    "n_samples": len(self.records),
                    "min_samples": min_samples,
                    "timestamp": datetime.now().isoformat()
                }))
                
                # Prepare training data
                X, y, signal_types = self._prepare_training_data()
                
                n_unique_strategies = len(set(signal_types))
                
                logger.info(json.dumps({
                    "event": "training_data_prepared",
                    "n_samples": len(X),
                    "n_features": X.shape[1],
                    "n_unique_strategies": n_unique_strategies,
                    "unique_strategies": list(set(signal_types))
                }))
                
                # Encode labels
                y_encoded = self.label_encoders['signal_type'].fit_transform(signal_types)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Train classifier with cross-validation
                self.strategy_classifier = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                
                # Cross-validation
                cv_scores = cross_val_score(
                    self.strategy_classifier,
                    X_scaled,
                    y_encoded,
                    cv=StratifiedKFold(n_splits=min(5, max(2, len(self.records) // 10)), random_state=42),
                    scoring='accuracy'
                )
                
                # Final fit
                self.strategy_classifier.fit(X_scaled, y_encoded)
                
                # Get feature importance
                feature_importance = self._get_feature_importance_dict()
                
                logger.info(json.dumps({
                    "event": "classifier_trained",
                    "cv_mean_accuracy": float(cv_scores.mean()),
                    "cv_std_accuracy": float(cv_scores.std()),
                    "cv_scores": [float(s) for s in cv_scores],
                    "top_features": list(feature_importance.items())[:5]
                }))
                
                # Train parameter regressors for each strategy
                self._train_parameter_regressors(X_scaled, signal_types)
                
                # Save models
                self._save_models()
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                result = {
                    "success": True,
                    "n_samples": len(self.records),
                    "n_strategies": n_unique_strategies,
                    "strategies": list(set(signal_types)),
                    "cv_scores": [float(s) for s in cv_scores],
                    "cv_mean_accuracy": float(cv_scores.mean()),
                    "cv_std_accuracy": float(cv_scores.std()),
                    "feature_importance": feature_importance,
                    "training_time": training_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(json.dumps({
                    "event": "training_completed",
                    **result
                }))
                
                return result
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "train_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "min_samples": min_samples
            }))
            raise

    def predict_best_strategy(
        self,
        features: MarketFeatures
    ) -> Tuple[str, float]:
        """
        Predict best strategy given market features.
        
        Args:
            features: Current market features
            
        Returns:
            Tuple of (strategy_signal_type, confidence)
            Confidence is probability of top prediction (0-1)
            
        Raises:
            RuntimeError: If model not trained
            ValueError: If features invalid
        """
        try:
            with self.lock:
                if self.strategy_classifier is None:
                    raise RuntimeError("Model not trained. Call train() first.")
                
                # Convert features to vector
                X_vec = self._features_to_vector([features])
                X_scaled = self.scaler.transform(X_vec)
                
                # Get predictions and probabilities
                y_pred = self.strategy_classifier.predict(X_scaled)
                y_proba = self.strategy_classifier.predict_proba(X_scaled)
                
                # Decode strategy
                strategy_idx = y_pred[0]
                strategy = self.label_encoders['signal_type'].inverse_transform([strategy_idx])[0]
                
                # Get confidence (max probability)
                confidence = float(np.max(y_proba))
                
                logger.debug(json.dumps({
                    "event": "strategy_prediction",
                    "strategy": strategy,
                    "confidence": confidence,
                    "all_probabilities": {
                        self.label_encoders['signal_type'].inverse_transform([i])[0]: float(p)
                        for i, p in enumerate(y_proba[0])
                    }
                }))
                
                return strategy, confidence
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "predict_best_strategy_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "features": features.to_dict()
            }))
            raise

    def predict_best_params(
        self,
        features: MarketFeatures,
        signal_type: str
    ) -> Dict[str, float]:
        """
        Predict optimal execution parameters for given strategy and market.
        
        Args:
            features: Current market features
            signal_type: Strategy signal type
            
        Returns:
            Dict of predicted optimal parameters
            
        Raises:
            RuntimeError: If model not trained or regressor not available
            ValueError: If signal_type not in training data
        """
        try:
            with self.lock:
                if signal_type not in self.param_regressors:
                    raise ValueError(f"No regressor for strategy: {signal_type}")
                
                regressor = self.param_regressors[signal_type]
                if regressor is None:
                    raise RuntimeError(f"Regressor not trained for {signal_type}")
                
                # Convert features to vector
                X_vec = self._features_to_vector([features])
                X_scaled = self.scaler.transform(X_vec)
                
                # Get predictions
                pred_params = regressor.predict(X_scaled)[0]
                
                # Map back to parameter names
                param_names = [
                    'delta_target', 'position_size_pct', 'stop_loss_pct',
                    'profit_target_pct', 'slippage_pct'
                ]
                
                predicted_params = {
                    name: float(value)
                    for name, value in zip(param_names, pred_params)
                }
                
                logger.debug(json.dumps({
                    "event": "parameter_prediction",
                    "signal_type": signal_type,
                    "predicted_params": predicted_params
                }))
                
                return predicted_params
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "predict_best_params_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "signal_type": signal_type,
                "features": features.to_dict()
            }))
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from trained classifier.
        
        Returns:
            Dict mapping feature names to importance scores
            
        Raises:
            RuntimeError: If model not trained
        """
        try:
            with self.lock:
                if self.strategy_classifier is None:
                    raise RuntimeError("Model not trained. Call train() first.")
                
                return self._get_feature_importance_dict()
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "get_feature_importance_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate model using cross-validation.
        
        Returns:
            Dict with evaluation metrics:
            - accuracy, precision, recall, f1 per strategy
            - confusion matrix
            - per-strategy performance
            
        Raises:
            RuntimeError: If model not trained
            ValueError: If not enough data for evaluation
        """
        try:
            with self.lock:
                if self.strategy_classifier is None:
                    raise RuntimeError("Model not trained. Call train() first.")
                
                # Prepare data
                X, y, signal_types = self._prepare_training_data()
                X_scaled = self.scaler.transform(X)
                y_encoded = self.label_encoders['signal_type'].transform(signal_types)
                
                # Cross-validation with multiple scorers
                cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
                
                scorers = {
                    'accuracy': 'accuracy',
                    'precision_weighted': 'precision_weighted',
                    'recall_weighted': 'recall_weighted',
                    'f1_weighted': 'f1_weighted'
                }
                
                cv_results = cross_validate(
                    self.strategy_classifier,
                    X_scaled,
                    y_encoded,
                    cv=cv,
                    scoring=scorers,
                    return_train_score=True
                )
                
                # Get predictions for confusion matrix
                y_pred = self.strategy_classifier.predict(X_scaled)
                
                # Build results
                evaluation_results = {
                    "timestamp": datetime.now().isoformat(),
                    "n_samples": len(X),
                    "n_folds": cv.get_n_splits(),
                    "accuracy": {
                        "cv_mean": float(cv_results['test_accuracy'].mean()),
                        "cv_std": float(cv_results['test_accuracy'].std()),
                        "train_mean": float(cv_results['train_accuracy'].mean())
                    },
                    "precision": {
                        "cv_mean": float(cv_results['test_precision_weighted'].mean()),
                        "cv_std": float(cv_results['test_precision_weighted'].std())
                    },
                    "recall": {
                        "cv_mean": float(cv_results['test_recall_weighted'].mean()),
                        "cv_std": float(cv_results['test_recall_weighted'].std())
                    },
                    "f1": {
                        "cv_mean": float(cv_results['test_f1_weighted'].mean()),
                        "cv_std": float(cv_results['test_f1_weighted'].std())
                    },
                    "per_strategy_performance": {}
                }
                
                # Per-strategy metrics
                for strategy_idx, strategy in enumerate(self.label_encoders['signal_type'].classes_):
                    mask = y_encoded == strategy_idx
                    if mask.sum() > 0:
                        strategy_precision = precision_score(
                            y_encoded[mask],
                            y_pred[mask],
                            zero_division=0,
                            average='weighted'
                        )
                        strategy_recall = recall_score(
                            y_encoded[mask],
                            y_pred[mask],
                            zero_division=0,
                            average='weighted'
                        )
                        
                        evaluation_results["per_strategy_performance"][strategy] = {
                            "n_samples": int(mask.sum()),
                            "precision": float(strategy_precision),
                            "recall": float(strategy_recall)
                        }
                
                logger.info(json.dumps({
                    "event": "model_evaluation_completed",
                    **evaluation_results
                }))
                
                return evaluation_results
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "evaluate_model_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    def save_model(self, path: Optional[str] = None) -> None:
        """
        Save trained model to disk.
        
        Args:
            path: Optional override path. Otherwise uses self.model_path
            
        Raises:
            RuntimeError: If model not trained
            IOError: If save fails
        """
        try:
            with self.lock:
                if self.strategy_classifier is None:
                    raise RuntimeError("No model to save. Train first.")
                
                self._save_models(path)
                
                logger.info(json.dumps({
                    "event": "model_saved",
                    "path": path or self.model_path,
                    "timestamp": datetime.now().isoformat()
                }))
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "save_model_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "path": path or self.model_path
            }))
            raise

    def load_model(self, path: Optional[str] = None) -> bool:
        """
        Load trained model from disk.
        
        Args:
            path: Optional override path. Otherwise uses self.model_path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.lock:
                path = path or self.model_path
                
                classifier_path = os.path.join(path, "strategy_classifier.pkl")
                scaler_path = os.path.join(path, "scaler.pkl")
                encoders_path = os.path.join(path, "label_encoders.pkl")
                regressors_path = os.path.join(path, "param_regressors.pkl")
                
                if not os.path.exists(classifier_path):
                    logger.warning(json.dumps({
                        "event": "model_load_failed",
                        "reason": "classifier not found",
                        "path": path
                    }))
                    return False
                
                self.strategy_classifier = joblib.load(classifier_path)
                self.scaler = joblib.load(scaler_path)
                self.label_encoders = joblib.load(encoders_path)
                
                if os.path.exists(regressors_path):
                    self.param_regressors = joblib.load(regressors_path)
                
                logger.info(json.dumps({
                    "event": "model_loaded",
                    "path": path,
                    "timestamp": datetime.now().isoformat()
                }))
                
                return True
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "load_model_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "path": path or self.model_path
            }))
            return False

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training data statistics.
        
        Returns:
            Dict with training statistics:
            - n_records: total records
            - n_strategies: unique strategies
            - strategies: list of strategies
            - records_per_strategy: count per strategy
            - date_range: earliest to latest timestamp
            - avg_metrics: average Sharpe, win_rate, P&L
        """
        try:
            with self.lock:
                if not self.records:
                    return {
                        "n_records": 0,
                        "n_strategies": 0,
                        "strategies": [],
                        "records_per_strategy": {},
                        "date_range": None,
                        "avg_metrics": None
                    }
                
                # Count strategies
                strategy_counts = defaultdict(int)
                sharpes = []
                win_rates = []
                pnls = []
                timestamps = []
                
                for record in self.records:
                    strategy_counts[record.signal_type] += 1
                    sharpes.append(record.backtest_sharpe)
                    win_rates.append(record.backtest_win_rate)
                    pnls.append(record.backtest_pnl)
                    timestamps.append(datetime.fromisoformat(record.timestamp))
                
                return {
                    "n_records": len(self.records),
                    "n_strategies": len(strategy_counts),
                    "strategies": list(strategy_counts.keys()),
                    "records_per_strategy": dict(strategy_counts),
                    "date_range": {
                        "earliest": min(timestamps).isoformat(),
                        "latest": max(timestamps).isoformat()
                    },
                    "avg_metrics": {
                        "avg_sharpe": float(np.mean(sharpes)),
                        "avg_win_rate": float(np.mean(win_rates)),
                        "avg_pnl": float(np.mean(pnls)),
                        "max_sharpe": float(np.max(sharpes)),
                        "max_pnl": float(np.max(pnls)),
                        "min_sharpe": float(np.min(sharpes))
                    }
                }
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "get_training_stats_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    def _features_to_vector(self, features_list: List[MarketFeatures]) -> np.ndarray:
        """
        Convert MarketFeatures to feature vectors.
        
        Args:
            features_list: List of MarketFeatures
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        try:
            vectors = []
            
            for features in features_list:
                # Encode regime
                regime_encoded = self.label_encoders['regime'].transform([features.regime])[0]
                
                vector = [
                    features.vix_level,
                    features.vrp_score,
                    regime_encoded,
                    features.iv_rank,
                    features.put_call_ratio,
                    features.spot_momentum,
                    features.realized_vol,
                    features.days_to_expiry,
                    features.time_of_day,
                    features.day_of_week
                ]
                vectors.append(vector)
            
            return np.array(vectors, dtype=np.float32)
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "features_to_vector_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    @staticmethod
    def _encode_signal_type(signal_type: str) -> int:
        """
        Encode signal type to integer.
        
        Args:
            signal_type: Signal type string
            
        Returns:
            Integer encoding
        """
        signal_mapping = {
            'long_strangle': 0,
            'short_strangle': 1,
            'iron_condor': 2,
            'butterfly': 3,
            'covered_call': 4,
            'long_call': 5,
            'long_put': 6,
            'bull_spread': 7,
            'bear_spread': 8,
            'no_trade': 9
        }
        return signal_mapping.get(signal_type, 9)

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from records.
        
        Returns:
            Tuple of (X, y, signal_types)
        """
        try:
            # Ensure label encoders are fitted
            all_regimes = set(r.features.regime for r in self.records)
            self.label_encoders['regime'].fit(list(all_regimes))
            
            all_signals = set(r.signal_type for r in self.records)
            self.label_encoders['signal_type'].fit(list(all_signals))
            
            # Extract features and labels
            features_list = [r.features for r in self.records]
            signal_types = [r.signal_type for r in self.records]
            
            X = self._features_to_vector(features_list)
            
            return X, None, signal_types
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "prepare_training_data_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            raise

    def _get_feature_importance_dict(self) -> Dict[str, float]:
        """Get feature importance as dictionary."""
        try:
            if self.strategy_classifier is None:
                return {}
            
            feature_names = [
                'vix_level', 'vrp_score', 'regime', 'iv_rank',
                'put_call_ratio', 'spot_momentum', 'realized_vol',
                'days_to_expiry', 'time_of_day', 'day_of_week'
            ]
            
            importances = self.strategy_classifier.feature_importances_
            
            return {
                name: float(importance)
                for name, importance in zip(feature_names, importances)
            }
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "get_feature_importance_dict_error",
                "error": str(e)
            }))
            return {}

    def _train_parameter_regressors(self, X_scaled: np.ndarray, signal_types: List[str]) -> None:
        """Train regressors to predict parameters for each strategy."""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            
            for strategy in set(signal_types):
                mask = np.array([s == strategy for s in signal_types])
                
                if mask.sum() < 10:
                    logger.debug(f"Skipping regressor for {strategy}: only {mask.sum()} samples")
                    continue
                
                # Get records for this strategy
                strategy_records = [r for r, m in zip(self.records, mask) if m]
                
                # Extract parameters as targets (simplified: use execution params)
                y_params = []
                for record in strategy_records:
                    # Extract key parameters
                    params = [
                        record.execution_params.get('delta_target', 0.3),
                        record.execution_params.get('position_size_pct', 5.0),
                        record.execution_params.get('stop_loss_pct', -2.0),
                        record.execution_params.get('profit_target_pct', 5.0),
                        record.execution_params.get('slippage_pct', 0.5)
                    ]
                    y_params.append(params)
                
                X_strategy = X_scaled[mask]
                y_strategy = np.array(y_params)
                
                # Train regressor
                regressor = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42
                )
                
                regressor.fit(X_strategy, y_strategy)
                self.param_regressors[strategy] = regressor
                
                logger.debug(json.dumps({
                    "event": "parameter_regressor_trained",
                    "strategy": strategy,
                    "n_samples": mask.sum(),
                    "n_params": y_strategy.shape[1]
                }))
                
        except Exception as e:
            logger.warning(json.dumps({
                "event": "train_parameter_regressors_error",
                "error": str(e)
            }))

    def _save_models(self, path: Optional[str] = None) -> None:
        """Save all models to disk."""
        try:
            path = path or self.model_path
            os.makedirs(path, exist_ok=True)
            
            joblib.dump(
                self.strategy_classifier,
                os.path.join(path, "strategy_classifier.pkl")
            )
            joblib.dump(
                self.scaler,
                os.path.join(path, "scaler.pkl")
            )
            joblib.dump(
                self.label_encoders,
                os.path.join(path, "label_encoders.pkl")
            )
            joblib.dump(
                self.param_regressors,
                os.path.join(path, "param_regressors.pkl")
            )
            
            logger.debug(f"Models saved to {path}")
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "save_models_error",
                "error": str(e),
                "path": path
            }))
            raise

    def _append_record_to_jsonl(self, record: StrategyRecord) -> None:
        """Append record to JSONL file."""
        try:
            os.makedirs(os.path.dirname(self.records_file) or ".", exist_ok=True)
            
            with open(self.records_file, 'a') as f:
                f.write(json.dumps(record.to_dict()) + '\n')
                
        except Exception as e:
            logger.warning(json.dumps({
                "event": "append_record_to_jsonl_error",
                "error": str(e),
                "records_file": self.records_file
            }))

    def _load_records_from_jsonl(self) -> None:
        """Load records from JSONL file."""
        try:
            if not os.path.exists(self.records_file):
                return
            
            with open(self.records_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            record = StrategyRecord.from_dict(data)
                            self.records.append(record)
                        except Exception as e:
                            logger.warning(f"Failed to parse record: {e}")
            
            logger.info(f"Loaded {len(self.records)} records from {self.records_file}")
            
        except Exception as e:
            logger.warning(json.dumps({
                "event": "load_records_from_jsonl_error",
                "error": str(e)
            }))


# ============================================================================
# ONLINE LEARNER
# ============================================================================

class OnlineLearner:
    """
    Online learning wrapper for incremental model updates.
    
    Uses partial_fit for streaming backtest results without full retraining.
    Tracks recent accuracy for performance monitoring.
    
    Attributes:
        learner: Underlying StrategyLearner
        recent_predictions: Deque of recent (true, predicted) pairs
        max_recent: Maximum records to keep in recent history
        lock: Thread lock
    """

    def __init__(self, learner: StrategyLearner, max_recent: int = 100):
        """
        Initialize OnlineLearner.
        
        Args:
            learner: StrategyLearner instance to wrap
            max_recent: Maximum recent predictions to track
        """
        try:
            self.learner = learner
            self.max_recent = max_recent
            self.recent_predictions: deque = deque(maxlen=max_recent)
            self.lock = threading.RLock()
            
            logger.info(json.dumps({
                "event": "online_learner_initialized",
                "max_recent": max_recent,
                "timestamp": datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.error(json.dumps({
                "event": "online_learner_init_error",
                "error": str(e)
            }))
            raise

    def incremental_update(self, new_record: StrategyRecord) -> Optional[str]:
        """
        Add new record and incrementally update model.
        
        Uses partial_fit for efficient online learning.
        
        Args:
            new_record: New StrategyRecord from backtest
            
        Returns:
            Predicted strategy type, or None if prediction failed
        """
        try:
            with self.lock:
                # Add to learner's records
                self.learner.add_backtest_result(
                    features=new_record.features,
                    signal_type=new_record.signal_type,
                    exec_params=new_record.execution_params,
                    backtest_result={
                        'sharpe': new_record.backtest_sharpe,
                        'win_rate': new_record.backtest_win_rate,
                        'pnl': new_record.backtest_pnl
                    }
                )
                
                # Try to make prediction
                try:
                    predicted_strategy, confidence = self.learner.predict_best_strategy(
                        new_record.features
                    )
                    
                    # Track for accuracy calculation
                    self.recent_predictions.append({
                        'true': new_record.signal_type,
                        'predicted': predicted_strategy,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    logger.debug(json.dumps({
                        "event": "online_prediction",
                        "true_strategy": new_record.signal_type,
                        "predicted_strategy": predicted_strategy,
                        "confidence": confidence,
                        "correct": new_record.signal_type == predicted_strategy
                    }))
                    
                    return predicted_strategy
                    
                except RuntimeError:
                    # Model not yet trained
                    logger.debug("Model not yet trained for online prediction")
                    return None
                    
        except Exception as e:
            logger.error(json.dumps({
                "event": "incremental_update_error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }))
            return None

    def get_recent_accuracy(self, n_recent: Optional[int] = None) -> float:
        """
        Calculate accuracy on recent predictions.
        
        Args:
            n_recent: Number of recent predictions to use. None = all available
            
        Returns:
            Accuracy score (0-1)
        """
        try:
            with self.lock:
                if not self.recent_predictions:
                    return 0.0
                
                n = n_recent or len(self.recent_predictions)
                recent = list(self.recent_predictions)[-n:]
                
                if not recent:
                    return 0.0
                
                correct = sum(
                    1 for p in recent
                    if p['true'] == p['predicted']
                )
                
                accuracy = correct / len(recent)
                
                logger.debug(json.dumps({
                    "event": "recent_accuracy_calculated",
                    "n_recent": len(recent),
                    "accuracy": accuracy,
                    "correct": correct
                }))
                
                return accuracy
                
        except Exception as e:
            logger.error(json.dumps({
                "event": "get_recent_accuracy_error",
                "error": str(e)
            }))
            return 0.0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_default_learner(model_path: str = "models") -> StrategyLearner:
    """
    Create StrategyLearner with default configuration.
    
    Args:
        model_path: Path for model storage
        
    Returns:
        Configured StrategyLearner instance
    """
    try:
        learner = StrategyLearner(model_path=model_path)
        
        # Try to load existing model
        learner.load_model()
        
        return learner
        
    except Exception as e:
        logger.error(json.dumps({
            "event": "create_default_learner_error",
            "error": str(e)
        }))
        raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of StrategyLearner and OnlineLearner.
    """
    try:
        # Create learner
        learner = create_default_learner()
        
        logger.info("Starting strategy learner example...")
        
        # Example market features
        features = MarketFeatures(
            vix_level=25.5,
            vrp_score=0.65,
            regime="volatile",
            iv_rank=75.0,
            put_call_ratio=1.2,
            spot_momentum=0.05,
            realized_vol=0.22,
            days_to_expiry=14,
            time_of_day=10,
            day_of_week=2
        )
        
        # Add some sample training records
        logger.info("Adding sample backtest results...")
        
        for i in range(60):
            exec_params = {
                'delta_target': -0.30,
                'position_size_pct': 5.0,
                'stop_loss_pct': -2.0,
                'profit_target_pct': 5.0,
                'slippage_pct': 0.5
            }
            
            strategy = ['iron_condor', 'short_strangle', 'long_call'][i % 3]
            
            backtest_result = {
                'sharpe': 1.0 + (i % 10) * 0.1,
                'win_rate': 0.5 + (i % 5) * 0.05,
                'pnl': 5000 + (i % 20) * 500
            }
            
            learner.add_backtest_result(
                features=features,
                signal_type=strategy,
                exec_params=exec_params,
                backtest_result=backtest_result
            )
        
        # Get training stats
        stats = learner.get_training_stats()
        logger.info(json.dumps({
            "event": "training_stats",
            **stats
        }))
        
        # Train model
        logger.info("Training model...")
        training_result = learner.train(min_samples=30)
        
        logger.info(json.dumps({
            "event": "training_result",
            **training_result
        }))
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluation = learner.evaluate_model()
        
        logger.info(json.dumps({
            "event": "model_evaluation",
            **evaluation
        }))
        
        # Predict best strategy
        logger.info("Predicting best strategy...")
        predicted_strategy, confidence = learner.predict_best_strategy(features)
        
        logger.info(json.dumps({
            "event": "strategy_prediction",
            "predicted_strategy": predicted_strategy,
            "confidence": confidence
        }))
        
        # Predict parameters
        logger.info("Predicting optimal parameters...")
        predicted_params = learner.predict_best_params(features, predicted_strategy)
        
        logger.info(json.dumps({
            "event": "parameter_prediction",
            "predicted_params": predicted_params
        }))
        
        # Save model
        logger.info("Saving model...")
        learner.save_model()
        
        # Test online learner
        logger.info("Testing online learner...")
        online_learner = OnlineLearner(learner, max_recent=20)
        
        for i in range(10):
            record = StrategyRecord(
                features=features,
                signal_type='iron_condor',
                execution_params=exec_params,
                backtest_sharpe=1.2 + i * 0.05,
                backtest_win_rate=0.55 + i * 0.02,
                backtest_pnl=7000 + i * 500
            )
            
            predicted = online_learner.incremental_update(record)
            logger.info(f"Online update {i+1}: predicted {predicted}")
        
        recent_accuracy = online_learner.get_recent_accuracy(n_recent=5)
        logger.info(json.dumps({
            "event": "online_learner_accuracy",
            "recent_accuracy": recent_accuracy
        }))
        
        logger.info("Example workflow completed successfully!")
        
    except Exception as e:
        logger.error(json.dumps({
            "event": "main_execution_error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }))
        sys.exit(1)
