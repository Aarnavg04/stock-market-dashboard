"""
Plotly Dash dashboard for visualization
"""
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_fetcher import StockDataFetcher
from src.monte_carlo import MonteCarloSimulator
from src.risk_metrics import RiskAnalyzer
from src.garch_model import GARCHVolatilityModel

# Initialize components
data_fetcher = StockDataFetcher()
simulator = MonteCarloSimulator(n_simulations=5000)
risk_analyzer = RiskAnalyzer()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Stock Market Analytics Dashboard", className="text-center mb-4"),
            html.H4("Monte Carlo Simulation with GARCH Volatility", className="text-center text-muted mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Select Parameters"),
                    dcc.Dropdown(
                        id='stock-dropdown',
                        options=[{'label': t, 'value': t} for t in ['AAPL', 'MSFT', 'GOOGL', 'TSLA']],
                        value='AAPL',
                        className="mb-3"
                    ),
                    html.Label("Horizon (Days)"),
                    dcc.Slider(
                        id='horizon-slider',
                        min=5, max=60, step=5, value=30,
                        marks={i: str(i) for i in range(5, 61, 10)}
                    ),
                    dbc.Button("Run Simulation", id="run-button", color="primary", className="w-100 mt-3")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dcc.Loading(dcc.Graph(id='price-paths-graph'))
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Risk Metrics"),
                    html.Div(id='risk-metrics-display')
                ])
            ])
        ], width=3)
    ])
])

@app.callback(
    [Output('price-paths-graph', 'figure'),
     Output('risk-metrics-display', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('stock-dropdown', 'value'),
     State('horizon-slider', 'value')]
)
def run_simulation(n_clicks, ticker, horizon):
    if not n_clicks:
        return {}, html.Div("Click 'Run Simulation' to start")
    
    # Fetch data
    prices = data_fetcher.fetch_historical_data(ticker, period="2y")
    if prices.empty:
        return {}, html.Div("Error fetching data")
    
    # Fit GARCH and simulate
    garch = GARCHVolatilityModel()
    garch_params = garch.fit(prices['Close'])
    paths = simulator.simulate_gbm_garch(prices['Close'], horizon, garch_params)
    
    # Calculate metrics
    metrics = risk_analyzer.calculate_metrics(paths, prices['Close'].iloc[-1])
    
    # Create price paths figure
    fig = go.Figure()
    
    # Add sample paths
    for i in range(min(100, simulator.n_simulations)):
        fig.add_trace(go.Scatter(
            y=paths[i, :],
            mode='lines',
            line=dict(color='lightgray', width=0.5),
            opacity=0.3,
            showlegend=False
        ))
    
    # Add mean path
    fig.add_trace(go.Scatter(
        y=np.mean(paths, axis=0),
        mode='lines',
        line=dict(color='blue', width=2),
        name='Mean Path'
    ))
    
    fig.update_layout(
        title=f"{ticker} Price Simulation",
        xaxis_title="Days",
        yaxis_title="Price ($)"
    )
    
    # Create metrics display
    metrics_display = html.Div([
        html.P(f"Expected Return: {metrics['expected_return']:.2f}%"),
        html.P(f"Volatility: {metrics['volatility']:.2f}%"),
        html.P(f"Prob. of Profit: {metrics['probability_profit']:.1f}%"),
        html.P(f"VaR (95%): {metrics['var_95']:.2f}%"),
        html.P(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    ])
    
    return fig, metrics_display

def run_dashboard():
    """Run the dashboard"""
    app.run(debug=True, port=8050)

if __name__ == '__main__':
    run_dashboard()