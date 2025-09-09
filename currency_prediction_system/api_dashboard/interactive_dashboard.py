#!/usr/bin/env python3
"""
Interactive Plotly Dashboard
===========================

This module creates an interactive dashboard with:
- Real-time currency predictions
- Technical indicators visualization
- Sentiment analysis trends
- Model performance metrics
- Cross-currency correlations
"""

import dash
from dash import dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class InteractiveDashboard:
    """Interactive Plotly dashboard for currency prediction system"""
    
    def __init__(self, config):
        self.config = config
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Currency Rate Prediction System", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Currency Selection
            dbc.Row([
                dbc.Col([
                    html.Label("Select Currency Pair:"),
                    dcc.Dropdown(
                        id='currency-dropdown',
                        options=[
                            {'label': 'USD/JPY', 'value': 'USDJPY'},
                            {'label': 'EUR/USD', 'value': 'EURUSD'},
                            {'label': 'GBP/USD', 'value': 'GBPUSD'},
                            {'label': 'USD/CHF', 'value': 'USDCHF'},
                            {'label': 'AUD/USD', 'value': 'AUDUSD'}
                        ],
                        value='USDJPY',
                        className="mb-3"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Time Range:"),
                    dcc.Dropdown(
                        id='time-range-dropdown',
                        options=[
                            {'label': '1 Day', 'value': '1D'},
                            {'label': '1 Week', 'value': '1W'},
                            {'label': '1 Month', 'value': '1M'},
                            {'label': '3 Months', 'value': '3M'},
                            {'label': '6 Months', 'value': '6M'},
                            {'label': '1 Year', 'value': '1Y'}
                        ],
                        value='1M',
                        className="mb-3"
                    )
                ], width=6)
            ]),
            
            # Main Charts
            dbc.Row([
                # Price and Predictions Chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Price and Predictions"),
                        dbc.CardBody([
                            dcc.Graph(id='price-predictions-chart')
                        ])
                    ])
                ], width=8),
                
                # Technical Indicators
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Technical Indicators"),
                        dbc.CardBody([
                            dcc.Graph(id='technical-indicators-chart')
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Sentiment and Performance
            dbc.Row([
                # Sentiment Analysis
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Sentiment Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id='sentiment-chart')
                        ])
                    ])
                ], width=6),
                
                # Model Performance
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Model Performance"),
                        dbc.CardBody([
                            dcc.Graph(id='performance-chart')
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Cross-Currency Correlations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Cross-Currency Correlations"),
                        dbc.CardBody([
                            dcc.Graph(id='correlation-heatmap')
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Model Metrics Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Model Metrics"),
                        dbc.CardBody([
                            html.Div(id='metrics-table')
                        ])
                    ])
                ])
            ]),
            
            # Update Interval
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # 30 seconds
                n_intervals=0
            )
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            Output('price-predictions-chart', 'figure'),
            [Input('currency-dropdown', 'value'),
             Input('time-range-dropdown', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_price_predictions_chart(currency_pair, time_range, n_intervals):
            """Update price and predictions chart"""
            # Simulate data (replace with actual data from your system)
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            
            # Simulate price data
            np.random.seed(42)
            base_price = 110 if currency_pair == 'USDJPY' else 1.1
            price_data = base_price + np.cumsum(np.random.randn(100) * 0.01)
            
            # Simulate predictions
            predictions = price_data + np.random.randn(100) * 0.005
            
            # Create figure
            fig = go.Figure()
            
            # Actual price
            fig.add_trace(go.Scatter(
                x=dates,
                y=price_data,
                mode='lines',
                name='Actual Price',
                line=dict(color='blue', width=2)
            ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=dates,
                y=predictions,
                mode='lines',
                name='Predictions',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Confidence interval
            confidence_upper = predictions + 0.01
            confidence_lower = predictions - 0.01
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=confidence_upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=confidence_lower,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                title=f'{currency_pair} Price and Predictions',
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('technical-indicators-chart', 'figure'),
            [Input('currency-dropdown', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_technical_indicators_chart(currency_pair, n_intervals):
            """Update technical indicators chart"""
            # Simulate technical indicators
            dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
            
            # RSI
            rsi = 50 + 20 * np.sin(np.linspace(0, 4*np.pi, 100))
            
            # MACD
            macd = np.random.randn(100) * 0.1
            macd_signal = macd + np.random.randn(100) * 0.05
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('RSI', 'MACD', 'Bollinger Bands'),
                vertical_spacing=0.1
            )
            
            # RSI
            fig.add_trace(
                go.Scatter(x=dates, y=rsi, name='RSI', line=dict(color='purple')),
                row=1, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
            
            # MACD
            fig.add_trace(
                go.Scatter(x=dates, y=macd, name='MACD', line=dict(color='blue')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=dates, y=macd_signal, name='Signal', line=dict(color='orange')),
                row=2, col=1
            )
            
            # Bollinger Bands
            base_price = 110
            sma = base_price + np.cumsum(np.random.randn(100) * 0.01)
            bb_upper = sma + 2 * np.std(sma)
            bb_lower = sma - 2 * np.std(sma)
            
            fig.add_trace(
                go.Scatter(x=dates, y=bb_upper, name='BB Upper', line=dict(color='gray')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=dates, y=sma, name='SMA', line=dict(color='blue')),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=dates, y=bb_lower, name='BB Lower', line=dict(color='gray')),
                row=3, col=1
            )
            
            fig.update_layout(height=600, showlegend=True)
            
            return fig
        
        @self.app.callback(
            Output('sentiment-chart', 'figure'),
            [Input('currency-dropdown', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_sentiment_chart(currency_pair, n_intervals):
            """Update sentiment analysis chart"""
            # Simulate sentiment data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            
            # Sentiment scores
            sentiment_scores = np.random.randn(30) * 0.3 + 0.1
            
            # Create figure
            fig = go.Figure()
            
            # Sentiment line
            fig.add_trace(go.Scatter(
                x=dates,
                y=sentiment_scores,
                mode='lines+markers',
                name='Sentiment Score',
                line=dict(color='green', width=2)
            ))
            
            # Zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            # Positive/Negative regions
            fig.add_hrect(y0=0, y1=1, fillcolor="green", opacity=0.1)
            fig.add_hrect(y0=-1, y1=0, fillcolor="red", opacity=0.1)
            
            fig.update_layout(
                title=f'{currency_pair} Sentiment Analysis',
                xaxis_title='Date',
                yaxis_title='Sentiment Score',
                yaxis=dict(range=[-1, 1])
            )
            
            return fig
        
        @self.app.callback(
            Output('performance-chart', 'figure'),
            [Input('currency-dropdown', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_performance_chart(currency_pair, n_intervals):
            """Update model performance chart"""
            # Simulate performance metrics
            models = ['Random Forest', 'XGBoost', 'LSTM', 'Ensemble']
            accuracy = [0.75, 0.78, 0.82, 0.85]
            mae = [0.045, 0.042, 0.038, 0.035]
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Accuracy', 'Mean Absolute Error'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Accuracy bars
            fig.add_trace(
                go.Bar(x=models, y=accuracy, name='Accuracy', marker_color='green'),
                row=1, col=1
            )
            
            # MAE bars
            fig.add_trace(
                go.Bar(x=models, y=mae, name='MAE', marker_color='red'),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f'{currency_pair} Model Performance',
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output('correlation-heatmap', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_correlation_heatmap(n_intervals):
            """Update correlation heatmap"""
            # Simulate correlation matrix
            currencies = ['USDJPY', 'EURUSD', 'GBPUSD', 'USDCHF', 'AUDUSD']
            
            # Generate correlation matrix
            np.random.seed(42)
            corr_matrix = np.random.rand(5, 5) * 0.8 + 0.2
            np.fill_diagonal(corr_matrix, 1.0)
            
            # Make it symmetric
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                x=currencies,
                y=currencies,
                color_continuous_scale='RdBu',
                aspect="auto"
            )
            
            fig.update_layout(
                title='Cross-Currency Correlations',
                xaxis_title='Currency Pairs',
                yaxis_title='Currency Pairs'
            )
            
            return fig
        
        @self.app.callback(
            Output('metrics-table', 'children'),
            [Input('currency-dropdown', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_metrics_table(currency_pair, n_intervals):
            """Update metrics table"""
            # Simulate metrics data
            metrics_data = {
                'Model': ['Random Forest', 'XGBoost', 'LSTM', 'Ensemble'],
                'Accuracy': [0.75, 0.78, 0.82, 0.85],
                'MAE': [0.045, 0.042, 0.038, 0.035],
                'Directional Accuracy': [0.68, 0.72, 0.75, 0.78],
                'Sharpe Ratio': [0.45, 0.52, 0.58, 0.62]
            }
            
            df = pd.DataFrame(metrics_data)
            
            # Create table
            table = dbc.Table.from_dataframe(
                df,
                striped=True,
                bordered=True,
                hover=True,
                className="mt-3"
            )
            
            return table
    
    def run(self, debug=True, host='0.0.0.0', port=8050):
        """Run the dashboard"""
        self.app.run_server(debug=debug, host=host, port=port) 