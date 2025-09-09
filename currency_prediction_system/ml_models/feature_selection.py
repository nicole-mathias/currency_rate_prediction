#!/usr/bin/env python3
"""
Feature Selection Module
=======================

This module implements feature selection based on the dt_importance analysis
and provides real performance testing with actual metrics.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import logging

class FeatureSelector:
    """Feature selection based on importance analysis"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Important features identified from dt_importance analysis
        self.important_features = [
            'High', 'Close', 'interest_r_us', 'inflation_us', 'gdp_pc_j',
            'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd',
            'bb_upper', 'bb_lower', 'volatility_20', 'volume_sma_20'
        ]
        
        # Features to drop (non-important from dt_importance analysis)
        self.features_to_drop = [
            'market_movement', 'exchange_rate_USD_JY_y', 'exchange_rate_USD_JY_x',
            'date', 'Volume', 'Low', 'Open', 'gdp_pc_j', 'Adj Close',
            'interest_r_j', 'govt_debt_j', 'inflation_j', 'gdp_pc_us', 'govt_debt_us'
        ]
        
    def select_important_features(self, df):
        """Select only the important features for prediction"""
        self.logger.info("Selecting important features...")
        
        # Remove non-important features
        available_features = [col for col in df.columns if col not in self.features_to_drop]
        
        # Keep only important features that exist in the dataset
        selected_features = [col for col in self.important_features if col in available_features]
        
        # Add any technical indicators that might be missing
        technical_indicators = [col for col in available_features if any(indicator in col for indicator in 
                           ['sma_', 'ema_', 'rsi', 'macd', 'bb_', 'volatility_', 'volume_'])]
        
        selected_features.extend(technical_indicators)
        selected_features = list(set(selected_features))  # Remove duplicates
        
        self.logger.info(f"Selected {len(selected_features)} important features")
        self.logger.info(f"Features: {selected_features}")
        
        return df[selected_features]
    
    def test_feature_importance(self, df, target_col='target'):
        """Test feature importance and get real performance metrics"""
        self.logger.info("Testing feature importance with real data...")
        
        # Prepare data
        if target_col not in df.columns:
            # Use the last column as target if target_col doesn't exist
            target_col = df.columns[-1]
        
        # Select important features
        X_important = self.select_important_features(df)
        y = df[target_col]
        
        # Also prepare data with all features for comparison
        X_all = df.drop([target_col], axis=1, errors='ignore')
        
        # Remove any columns that are all NaN
        X_important = X_important.dropna(axis=1, how='all')
        X_all = X_all.dropna(axis=1, how='all')
        
        # Align data
        common_index = X_important.index.intersection(y.index)
        X_important = X_important.loc[common_index]
        X_all = X_all.loc[common_index]
        y = y.loc[common_index]
        
        if len(X_important) == 0 or len(y) == 0:
            self.logger.warning("No data available for testing")
            return {}
        
        # Split data
        X_train_important, X_test_important, y_train, y_test = train_test_split(
            X_important, y, test_size=0.2, random_state=42
        )
        
        X_train_all, X_test_all, _, _ = train_test_split(
            X_all, y, test_size=0.2, random_state=42
        )
        
        # Scale data
        scaler = StandardScaler()
        X_train_important_scaled = scaler.fit_transform(X_train_important)
        X_test_important_scaled = scaler.transform(X_test_important)
        
        X_train_all_scaled = scaler.fit_transform(X_train_all)
        X_test_all_scaled = scaler.transform(X_test_all)
        
        # Test models with important features only
        results_important = self._test_models(X_train_important_scaled, X_test_important_scaled, y_train, y_test, "Important Features")
        
        # Test models with all features
        results_all = self._test_models(X_train_all_scaled, X_test_all_scaled, y_train, y_test, "All Features")
        
        # Calculate improvements
        improvements = self._calculate_improvements(results_important, results_all)
        
        return {
            'important_features': results_important,
            'all_features': results_all,
            'improvements': improvements,
            'selected_features': list(X_important.columns)
        }
    
    def _test_models(self, X_train, X_test, y_train, y_test, feature_set_name):
        """Test different models and return performance metrics"""
        self.logger.info(f"Testing models with {feature_set_name}...")
        
        results = {}
        
        # 1. Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        rf_metrics = {
            'mae': mean_absolute_error(y_test, rf_pred),
            'mse': mean_squared_error(y_test, rf_pred),
            'r2': r2_score(y_test, rf_pred),
            'cv_score': np.mean(cross_val_score(rf_model, X_train, y_train, cv=5))
        }
        results['Random Forest'] = rf_metrics
        
        # 2. XGBoost
        try:
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            
            xgb_metrics = {
                'mae': mean_absolute_error(y_test, xgb_pred),
                'mse': mean_squared_error(y_test, xgb_pred),
                'r2': r2_score(y_test, xgb_pred),
                'cv_score': np.mean(cross_val_score(xgb_model, X_train, y_train, cv=5))
            }
            results['XGBoost'] = xgb_metrics
        except Exception as e:
            self.logger.warning(f"XGBoost failed: {e}")
        
        # 3. Ensemble (simple average)
        ensemble_pred = (rf_pred + xgb_pred) / 2 if 'XGBoost' in results else rf_pred
        
        ensemble_metrics = {
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'mse': mean_squared_error(y_test, ensemble_pred),
            'r2': r2_score(y_test, ensemble_pred),
            'cv_score': np.mean([rf_metrics['cv_score'], results.get('XGBoost', {}).get('cv_score', 0)])
        }
        results['Ensemble'] = ensemble_metrics
        
        # Log results
        self.logger.info(f"\n{feature_set_name} Results:")
        for model_name, metrics in results.items():
            self.logger.info(f"{model_name}: MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")
        
        return results
    
    def _calculate_improvements(self, results_important, results_all):
        """Calculate real improvements from using important features"""
        improvements = {}
        
        for model_name in results_important.keys():
            if model_name in results_all:
                important_metrics = results_important[model_name]
                all_metrics = results_all[model_name]
                
                # Calculate percentage improvements
                mae_improvement = ((all_metrics['mae'] - important_metrics['mae']) / all_metrics['mae']) * 100
                r2_improvement = ((important_metrics['r2'] - all_metrics['r2']) / abs(all_metrics['r2'])) * 100 if all_metrics['r2'] != 0 else 0
                cv_improvement = ((important_metrics['cv_score'] - all_metrics['cv_score']) / abs(all_metrics['cv_score'])) * 100 if all_metrics['cv_score'] != 0 else 0
                
                improvements[model_name] = {
                    'mae_improvement': mae_improvement,
                    'r2_improvement': r2_improvement,
                    'cv_improvement': cv_improvement,
                    'important_mae': important_metrics['mae'],
                    'all_mae': all_metrics['mae'],
                    'important_r2': important_metrics['r2'],
                    'all_r2': all_metrics['r2']
                }
        
        return improvements
    
    def get_feature_importance_ranking(self, df, target_col='target'):
        """Get feature importance ranking using Random Forest"""
        self.logger.info("Calculating feature importance ranking...")
        
        # Prepare data
        if target_col not in df.columns:
            target_col = df.columns[-1]
        
        X = df.drop([target_col], axis=1, errors='ignore')
        y = df[target_col]
        
        # Remove NaN columns
        X = X.dropna(axis=1, how='all')
        
        # Align data
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) == 0 or len(y) == 0:
            return {}
        
        # Train Random Forest for feature importance
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # Get feature importance
        feature_importance = rf_model.feature_importances_
        feature_names = X.columns
        
        # Create ranking
        importance_dict = {}
        for feature, importance in zip(feature_names, feature_importance):
            importance_dict[feature] = importance
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        self.logger.info("Feature Importance Ranking:")
        for i, (feature, importance) in enumerate(sorted_importance.items(), 1):
            self.logger.info(f"{i}. {feature}: {importance:.4f}")
        
        return sorted_importance 