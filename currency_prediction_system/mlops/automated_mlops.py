#!/usr/bin/env python3
"""
MLOps Pipeline with Automated Retraining
=======================================

This module implements:
- Automated model retraining
- Performance monitoring
- Backtesting against market movements
- Model versioning and deployment
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import logging
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import joblib

# ML Libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

class MLOpsPipeline:
    """MLOps pipeline for automated model management"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup directories
        self.models_dir = Path('models')
        self.logs_dir = Path('logs')
        self.backtest_dir = Path('backtest_results')
        
        for dir_path in [self.models_dir, self.logs_dir, self.backtest_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Model registry
        self.model_registry = self._load_model_registry()
        
        # Performance tracking
        self.performance_history = self._load_performance_history()
        
    def _load_model_registry(self):
        """Load model registry from file"""
        registry_path = self.models_dir / 'model_registry.json'
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                return json.load(f)
        return {
            'models': {},
            'current_model': None,
            'deployment_history': []
        }
    
    def _save_model_registry(self):
        """Save model registry to file"""
        registry_path = self.models_dir / 'model_registry.json'
        with open(registry_path, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def _load_performance_history(self):
        """Load performance history from file"""
        history_path = self.logs_dir / 'performance_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                return json.load(f)
        return {
            'daily_performance': [],
            'model_comparisons': [],
            'backtest_results': []
        }
    
    def _save_performance_history(self):
        """Save performance history to file"""
        history_path = self.logs_dir / 'performance_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
    
    def register_model(self, model_name, model_path, metrics, version=None):
        """Register a new model in the registry"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_entry = {
            'name': model_name,
            'version': version,
            'path': str(model_path),
            'metrics': metrics,
            'created_at': datetime.now().isoformat(),
            'status': 'trained'
        }
        
        self.model_registry['models'][f"{model_name}_{version}"] = model_entry
        
        self.logger.info(f"Registered model: {model_name}_{version}")
        self._save_model_registry()
        
        return f"{model_name}_{version}"
    
    def deploy_model(self, model_id):
        """Deploy a model as the current production model"""
        if model_id not in self.model_registry['models']:
            raise ValueError(f"Model {model_id} not found in registry")
        
        # Update current model
        self.model_registry['current_model'] = model_id
        
        # Add to deployment history
        deployment_entry = {
            'model_id': model_id,
            'deployed_at': datetime.now().isoformat(),
            'previous_model': self.model_registry.get('current_model')
        }
        self.model_registry['deployment_history'].append(deployment_entry)
        
        self.logger.info(f"Deployed model: {model_id}")
        self._save_model_registry()
    
    def monitor_performance(self, model_id, actual_values, predicted_values):
        """Monitor model performance in production"""
        # Calculate metrics
        mae = mean_absolute_error(actual_values, predicted_values)
        mse = mean_squared_error(actual_values, predicted_values)
        r2 = r2_score(actual_values, predicted_values)
        
        # Calculate directional accuracy
        actual_direction = np.diff(actual_values) > 0
        predicted_direction = np.diff(predicted_values) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(actual_values) / actual_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        performance_metrics = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store performance
        self.performance_history['daily_performance'].append({
            'model_id': model_id,
            'metrics': performance_metrics
        })
        
        # Check if retraining is needed
        self._check_retraining_needed(model_id, performance_metrics)
        
        self._save_performance_history()
        
        return performance_metrics
    
    def _check_retraining_needed(self, model_id, current_metrics):
        """Check if model retraining is needed based on performance"""
        # Get historical performance for this model
        model_performances = [
            entry for entry in self.performance_history['daily_performance']
            if entry['model_id'] == model_id
        ]
        
        if len(model_performances) < 7:  # Need at least a week of data
            return
        
        # Calculate average performance over last 7 days
        recent_performances = model_performances[-7:]
        avg_mae = np.mean([p['metrics']['mae'] for p in recent_performances])
        avg_directional_accuracy = np.mean([p['metrics']['directional_accuracy'] for p in recent_performances])
        
        # Check if performance is degrading
        current_mae = current_metrics['mae']
        current_directional_accuracy = current_metrics['directional_accuracy']
        
        # Retraining triggers
        retraining_needed = False
        reasons = []
        
        if current_mae > avg_mae * 1.2:  # 20% worse than average
            retraining_needed = True
            reasons.append("MAE degradation")
        
        if current_directional_accuracy < avg_directional_accuracy * 0.9:  # 10% worse
            retraining_needed = True
            reasons.append("Directional accuracy degradation")
        
        if current_directional_accuracy < 0.5:  # Below 50% accuracy
            retraining_needed = True
            reasons.append("Poor directional accuracy")
        
        if retraining_needed:
            self.logger.warning(f"Retraining needed for {model_id}: {', '.join(reasons)}")
            self._trigger_retraining(model_id, reasons)
    
    def _trigger_retraining(self, model_id, reasons):
        """Trigger automated model retraining"""
        retraining_event = {
            'model_id': model_id,
            'triggered_at': datetime.now().isoformat(),
            'reasons': reasons,
            'status': 'pending'
        }
        
        # Store retraining event
        retraining_log_path = self.logs_dir / 'retraining_events.json'
        retraining_events = []
        
        if retraining_log_path.exists():
            with open(retraining_log_path, 'r') as f:
                retraining_events = json.load(f)
        
        retraining_events.append(retraining_event)
        
        with open(retraining_log_path, 'w') as f:
            json.dump(retraining_events, f, indent=2)
        
        self.logger.info(f"Retraining triggered for {model_id}")
    
    def backtest_model(self, model_id, historical_data, test_period_days=30):
        """Backtest model against historical data"""
        self.logger.info(f"Backtesting model {model_id}")
        
        # Load model
        model_entry = self.model_registry['models'][model_id]
        model_path = Path(model_entry['path'])
        
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return None
        
        # Load model and make predictions
        try:
            model = joblib.load(model_path)
            
            # Prepare test data
            test_data = historical_data.tail(test_period_days)
            X_test = test_data.drop(['target'], axis=1, errors='ignore')
            y_test = test_data['target'] if 'target' in test_data.columns else test_data.iloc[:, -1]
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate backtest metrics
            backtest_metrics = self._calculate_backtest_metrics(y_test, predictions)
            
            # Store backtest results
            backtest_result = {
                'model_id': model_id,
                'test_period_days': test_period_days,
                'metrics': backtest_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            self.performance_history['backtest_results'].append(backtest_result)
            self._save_performance_history()
            
            # Save detailed backtest results
            backtest_file = self.backtest_dir / f"backtest_{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backtest_file, 'w') as f:
                json.dump(backtest_result, f, indent=2)
            
            self.logger.info(f"Backtest completed for {model_id}")
            return backtest_metrics
            
        except Exception as e:
            self.logger.error(f"Backtest failed for {model_id}: {e}")
            return None
    
    def _calculate_backtest_metrics(self, actual, predicted):
        """Calculate comprehensive backtest metrics"""
        # Basic regression metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        
        # Financial metrics
        returns_actual = np.diff(actual) / actual[:-1]
        returns_predicted = np.diff(predicted) / predicted[:-1]
        
        # Directional accuracy
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        # Sharpe ratio
        sharpe_ratio = np.mean(returns_actual) / np.std(returns_actual) if np.std(returns_actual) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + returns_actual)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        winning_trades = np.sum(returns_actual > 0)
        total_trades = len(returns_actual)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades
        }
    
    def compare_models(self, model_ids):
        """Compare multiple models"""
        self.logger.info(f"Comparing models: {model_ids}")
        
        comparison_results = {}
        
        for model_id in model_ids:
            if model_id in self.model_registry['models']:
                # Get model metrics
                model_metrics = self.model_registry['models'][model_id]['metrics']
                
                # Get recent performance
                recent_performance = [
                    entry for entry in self.performance_history['daily_performance']
                    if entry['model_id'] == model_id
                ][-7:]  # Last 7 days
                
                if recent_performance:
                    avg_metrics = {
                        'avg_mae': np.mean([p['metrics']['mae'] for p in recent_performance]),
                        'avg_directional_accuracy': np.mean([p['metrics']['directional_accuracy'] for p in recent_performance]),
                        'avg_sharpe_ratio': np.mean([p['metrics']['sharpe_ratio'] for p in recent_performance])
                    }
                else:
                    avg_metrics = model_metrics
                
                comparison_results[model_id] = {
                    'model_metrics': model_metrics,
                    'recent_performance': avg_metrics
                }
        
        # Store comparison
        self.performance_history['model_comparisons'].append({
            'timestamp': datetime.now().isoformat(),
            'models_compared': model_ids,
            'results': comparison_results
        })
        
        self._save_performance_history()
        
        return comparison_results
    
    def get_model_performance_summary(self, model_id=None):
        """Get performance summary for a model or all models"""
        if model_id:
            # Get specific model performance
            model_performances = [
                entry for entry in self.performance_history['daily_performance']
                if entry['model_id'] == model_id
            ]
        else:
            # Get all model performances
            model_performances = self.performance_history['daily_performance']
        
        if not model_performances:
            return {}
        
        # Calculate summary statistics
        all_mae = [p['metrics']['mae'] for p in model_performances]
        all_directional_accuracy = [p['metrics']['directional_accuracy'] for p in model_performances]
        all_sharpe_ratio = [p['metrics']['sharpe_ratio'] for p in model_performances]
        
        summary = {
            'total_predictions': len(model_performances),
            'avg_mae': np.mean(all_mae),
            'avg_directional_accuracy': np.mean(all_directional_accuracy),
            'avg_sharpe_ratio': np.mean(all_sharpe_ratio),
            'best_mae': np.min(all_mae),
            'best_directional_accuracy': np.max(all_directional_accuracy),
            'best_sharpe_ratio': np.max(all_sharpe_ratio),
            'worst_mae': np.max(all_mae),
            'worst_directional_accuracy': np.min(all_directional_accuracy),
            'worst_sharpe_ratio': np.min(all_sharpe_ratio)
        }
        
        return summary
    
    def schedule_retraining(self, model_id, schedule_type='weekly'):
        """Schedule automated retraining"""
        schedule_entry = {
            'model_id': model_id,
            'schedule_type': schedule_type,
            'created_at': datetime.now().isoformat(),
            'next_retraining': self._calculate_next_retraining(schedule_type),
            'status': 'scheduled'
        }
        
        # Store schedule
        schedule_path = self.logs_dir / 'retraining_schedule.json'
        schedule_data = []
        
        if schedule_path.exists():
            with open(schedule_path, 'r') as f:
                schedule_data = json.load(f)
        
        schedule_data.append(schedule_entry)
        
        with open(schedule_path, 'w') as f:
            json.dump(schedule_data, f, indent=2)
        
        self.logger.info(f"Scheduled {schedule_type} retraining for {model_id}")
    
    def _calculate_next_retraining(self, schedule_type):
        """Calculate next retraining date"""
        now = datetime.now()
        
        if schedule_type == 'daily':
            next_date = now + timedelta(days=1)
        elif schedule_type == 'weekly':
            next_date = now + timedelta(weeks=1)
        elif schedule_type == 'monthly':
            next_date = now + timedelta(days=30)
        else:
            next_date = now + timedelta(weeks=1)
        
        return next_date.isoformat() 