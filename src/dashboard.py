"""
Enhanced Stock Market Analytics Dashboard with Event-Driven Analysis
"""
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yfinance as yf

from src.data_fetcher import StockDataFetcher
from src.monte_carlo import MonteCarloSimulator
from src.risk_metrics import RiskAnalyzer
from src.garch_model import GARCHVolatilityModel
from src.backtest import EarningsVolatilityBacktest

logger = logging.getLogger(__name__)


def _naive_day(ts) -> pd.Timestamp:
    """Timestamp -> tz-naive midnight of the same local calendar day."""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def next_event_date(earnings_dates, after):
    """Earliest earnings date strictly after `after` (tz-safe), or None if there is none."""
    after = _naive_day(after)
    upcoming = [_naive_day(d) for d in earnings_dates if _naive_day(d) > after]
    return min(upcoming) if upcoming else None


# Initialize components with enhanced settings
data_fetcher = StockDataFetcher()
simulator = MonteCarloSimulator(n_simulations=10000)  # Increased to 10,000 simulations
risk_analyzer = RiskAnalyzer()

# Initialize Dash app with dark theme
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
                suppress_callback_exceptions=True)

# Custom CSS for modern look
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
                font-family: 'Inter', sans-serif;
            }
            .card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 15px;
            }
            .metric-card {
                background: linear-gradient(135deg, rgba(130, 87, 229, 0.2), rgba(49, 119, 238, 0.2));
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 10px;
                transition: transform 0.3s;
            }
            .metric-card:hover {
                transform: translateY(-2px);
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #00d4ff;
            }
            .metric-label {
                font-size: 12px;
                color: #a0a0a0;
                text-transform: uppercase;
            }
            .positive {
                color: #00ff88 !important;
            }
            .negative {
                color: #ff4757 !important;
            }
            .btn-glow {
                box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
                transition: all 0.3s;
            }
            .btn-glow:hover {
                box-shadow: 0 0 30px rgba(0, 212, 255, 0.8);
                transform: translateY(-2px);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Get S&P 500 tickers
sp500_tickers = data_fetcher.get_sp500_tickers()

# Layout with enhanced UI
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    html.I(className="fas fa-chart-line me-3"),
                    "Stock Market Analytics Dashboard"
                ], className="text-center mb-2", style={'color': '#00d4ff'}),
                html.P("Monte Carlo Simulation with GARCH Volatility & Event-Driven Analysis", 
                       className="text-center text-muted mb-4",
                       style={'fontSize': '18px'})
            ])
        ])
    ], className="mb-4"),
    
    # Control Panel and Main Visualizations
    dbc.Row([
        # Left Panel - Controls
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([html.I(className="fas fa-cog me-2"), "Simulation Parameters"], 
                            className="mb-4", style={'color': '#00d4ff'}),
                    
                    # Stock Selection
                    html.Label("Select Stock (S&P 500)", className="mb-2"),
                    dcc.Dropdown(
                        id='stock-dropdown',
                        options=[{'label': t, 'value': t} for t in sp500_tickers],
                        value='AAPL',
                        className="mb-3",
                        style={'backgroundColor': 'rgba(0,0,0,0.3)'}
                    ),
                    
                    # Simulation Settings
                    html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)'}),
                    
                    html.Label("Forecast Horizon (Days)", className="mb-2"),
                    dcc.Slider(
                        id='horizon-slider',
                        min=5, max=90, step=5, value=30,
                        marks={i: f'{i}d' for i in [5, 30, 60, 90]},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    html.Label("Number of Simulations", className="mb-2 mt-3"),
                    dcc.RadioItems(
                        id='sim-count',
                        options=[
                            {'label': '1,000 (Fast)', 'value': 1000},
                            {'label': '5,000 (Balanced)', 'value': 5000},
                            {'label': '10,000 (Accurate)', 'value': 10000}
                        ],
                        value=10000,
                        inline=True,
                        className="mb-3"
                    ),
                    
                    # Event Analysis Toggle
                    dbc.Switch(
                        id='earnings-toggle',
                        label="Analyze Earnings Event",
                        value=True,
                        className="mb-3"
                    ),
                    
                    # Run Button
                    dbc.Button([
                        html.I(className="fas fa-play me-2"),
                        "Run Simulation"
                    ], id="run-button", 
                       color="primary", 
                       className="w-100 mt-3 btn-glow",
                       size="lg"),
                ])
            ], className="mb-3"),
            
            # Quick Stats Card
            dbc.Card([
                dbc.CardBody([
                    html.H5([html.I(className="fas fa-info-circle me-2"), "Quick Stats"], 
                            className="mb-3", style={'color': '#00d4ff'}),
                    html.Div(id='quick-stats')
                ])
            ])
        ], width=3),
        
        # Center - Main Visualizations
        dbc.Col([
            dcc.Loading(
                type="default",
                children=[
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(id='price-paths-graph', 
                                     config={'displayModeBar': False},
                                     style={'height': '500px'})
                        ])
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dcc.Graph(id='distribution-graph',
                                             config={'displayModeBar': False},
                                             style={'height': '300px'})
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dcc.Graph(id='volatility-cone',
                                             config={'displayModeBar': False},
                                             style={'height': '300px'})
                                ])
                            ])
                        ], width=6)
                    ])
                ]
            )
        ], width=6),
        
        # Right Panel - Risk Metrics
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([html.I(className="fas fa-shield-alt me-2"), "Risk Analytics"], 
                            className="mb-4", style={'color': '#00d4ff'}),
                    html.Div(id='risk-metrics-display')
                ])
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardBody([
                    html.H5([html.I(className="fas fa-chart-bar me-2"), "Probability Analysis"], 
                            className="mb-3", style={'color': '#00d4ff'}),
                    html.Div(id='probability-analysis')
                ])
            ])
        ], width=3)
    ]),
    
    # Bottom Row - Additional Analytics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([html.I(className="fas fa-history me-2"), "Historical Accuracy"], 
                            className="mb-3", style={'color': '#00d4ff'}),
                    dcc.Graph(id='backtest-accuracy',
                             config={'displayModeBar': False},
                             style={'height': '250px'})
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([html.I(className="fas fa-calendar me-2"), "Event Calendar"], 
                            className="mb-3", style={'color': '#00d4ff'}),
                    html.Div(id='event-calendar')
                ])
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([html.I(className="fas fa-database me-2"), "Data Pipeline Status"], 
                            className="mb-3", style={'color': '#00d4ff'}),
                    html.Div(id='pipeline-status')
                ])
            ])
        ], width=4)
    ], className="mt-3"),
    
    # Hidden div to store simulation results
    dcc.Store(id='simulation-results')
], fluid=True, className="p-4")
# Callback for running simulation
@app.callback(
    Output('simulation-results', 'data'),
    [Input('run-button', 'n_clicks')],
    [State('stock-dropdown', 'value'),
     State('horizon-slider', 'value'),
     State('sim-count', 'value'),
     State('earnings-toggle', 'value')]
)
def run_simulation(n_clicks, ticker, horizon, n_sims, analyze_earnings):
    if not n_clicks:
        return {}
    
    # Update simulator with selected number of simulations
    simulator.n_simulations = n_sims
    
    # Fetch 5 years of historical data
    prices = data_fetcher.fetch_historical_data(ticker, period="5y")
    if prices.empty:
        return {}
    
    # Fit GARCH model
    garch = GARCHVolatilityModel()
    garch_params = garch.fit(prices['Close'])
    
    # Simulations start from the last close, so count days to the event from that date
    last_close_date = _naive_day(prices.index[-1])

    # Check for an upcoming earnings event if enabled
    event_date = None
    if analyze_earnings:
        event_date = next_event_date(data_fetcher.fetch_earnings_dates(ticker), last_close_date)

    # Run simulation
    days_to_event = None
    if event_date is not None:
        # Simulate around earnings with volatility adjustment
        try:
            paths, days_to_event = simulator.simulate_around_event(
                prices['Close'], event_date, last_close_date, vol_multiplier=1.5
            )
        except ValueError as e:
            logger.warning(f"Event simulation skipped: {e}")
    if days_to_event is None:
        # Standard simulation (no upcoming earnings date, or event mode off)
        paths = simulator.simulate_gbm_garch(prices['Close'], horizon, garch_params)
    horizon = paths.shape[1]  # event runs cover days_to_event + 5 post-event days
    
    # Calculate comprehensive metrics
    metrics = risk_analyzer.calculate_metrics(paths, prices['Close'].iloc[-1])
    
    # Calculate path statistics
    path_stats = risk_analyzer.calculate_path_statistics(paths)
    
    # Historical volatility for comparison
    returns = np.log(prices['Close'] / prices['Close'].shift(1)).dropna()
    hist_vol = returns.std() * np.sqrt(252) * 100
    
    # Prepare results
    results = {
        'paths': paths.tolist(),
        'metrics': metrics,
        'path_stats': path_stats.to_dict(),
        'current_price': float(prices['Close'].iloc[-1]),
        'ticker': ticker,
        'horizon': horizon,
        'garch_params': garch_params,
        'historical_vol': hist_vol,
        'days_to_event': days_to_event,
        'n_simulations': n_sims
    }
    
    return results

# Callback for updating main graphs
@app.callback(
    [Output('price-paths-graph', 'figure'),
     Output('distribution-graph', 'figure'),
     Output('volatility-cone', 'figure')],
    [Input('simulation-results', 'data')]
)
def update_main_graphs(results):
    if not results:
        empty_fig = go.Figure()
        empty_fig.update_layout(template='plotly_dark')
        return empty_fig, empty_fig, empty_fig
    
    paths = np.array(results['paths'])
    current_price = results['current_price']
    ticker = results['ticker']
    
    # 1. Price Paths Figure with confidence bands
    fig_paths = go.Figure()
    
    # Add percentile bands
    percentiles = [5, 25, 50, 75, 95]
    colors = ['rgba(255,71,87,0.2)', 'rgba(255,200,87,0.3)', 
              'rgba(0,212,255,1)', 'rgba(255,200,87,0.3)', 'rgba(255,71,87,0.2)']
    
    for i, p in enumerate(percentiles):
        y_values = np.percentile(paths, p, axis=0)
        fig_paths.add_trace(go.Scatter(
            y=y_values,
            mode='lines',
            name=f'{p}th Percentile',
            line=dict(color=colors[i], width=2 if p == 50 else 1),
            fill='tonexty' if i > 0 else None
        ))
    
    # Add sample paths
    sample_indices = np.random.choice(paths.shape[0], min(100, paths.shape[0]), replace=False)
    for i in sample_indices:
        fig_paths.add_trace(go.Scatter(
            y=paths[i, :],
            mode='lines',
            line=dict(color='gray', width=0.3),
            opacity=0.1,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add current price line
    fig_paths.add_hline(y=current_price, line_dash="dash", 
                        line_color="white", opacity=0.5,
                        annotation_text=f"Current: ${current_price:.2f}")
    
    # Add earnings marker if applicable
    if results.get('days_to_event'):
        fig_paths.add_vline(x=results['days_to_event'], 
                           line_dash="dash", line_color="yellow",
                           annotation_text="Earnings")
    
    fig_paths.update_layout(
        title=f"{ticker} Monte Carlo Price Simulation ({results['n_simulations']:,} paths)",
        xaxis_title="Days Forward",
        yaxis_title="Price ($)",
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified'
    )
    
    # 2. Distribution of Final Prices
    final_prices = paths[:, -1]
    returns = (final_prices - current_price) / current_price * 100
    
    fig_dist = go.Figure()
    
    # Histogram
    fig_dist.add_trace(go.Histogram(
        x=returns,
        nbinsx=50,
        name='Distribution',
        marker_color='rgba(0,212,255,0.6)',
        showlegend=False
    ))
    
    # Add vertical lines for key metrics
    fig_dist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
    fig_dist.add_vline(x=np.mean(returns), line_color="green", 
                      annotation_text=f"Mean: {np.mean(returns):.1f}%")
    fig_dist.add_vline(x=np.percentile(returns, 5), line_color="red",
                      annotation_text=f"VaR 95%: {np.percentile(returns, 5):.1f}%")
    
    fig_dist.update_layout(
        title=f"Return Distribution at Day {results['horizon']}",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        template='plotly_dark',
        showlegend=False
    )
    
    # 3. Volatility Cone
    fig_cone = go.Figure()
    
    # Calculate realized volatility at different horizons
    horizons = [5, 10, 20, 30, 60, 90]
    realized_vols = []
    
    for h in horizons:
        if h <= paths.shape[1]:
            h_returns = (paths[:, h-1] - current_price) / current_price
            vol = np.std(h_returns) * np.sqrt(252/h) * 100
            realized_vols.append(vol)
        else:
            realized_vols.append(np.nan)
    
    # Plot volatility cone
    fig_cone.add_trace(go.Scatter(
        x=horizons[:len(realized_vols)],
        y=realized_vols,
        mode='lines+markers',
        name='Simulated Vol',
        line=dict(color='cyan', width=2),
        marker=dict(size=8)
    ))
    
    # Add historical volatility line
    fig_cone.add_hline(y=results['historical_vol'], 
                      line_dash="dash", line_color="yellow",
                      annotation_text=f"Historical: {results['historical_vol']:.1f}%")
    
    # Add GARCH forecast
    # long_run_variance is a daily variance of percent returns -> annualized vol in %
    garch_vol = np.sqrt(results['garch_params']['long_run_variance']) * np.sqrt(252)
    fig_cone.add_hline(y=garch_vol, 
                      line_dash="dot", line_color="magenta",
                      annotation_text=f"GARCH LR: {garch_vol:.1f}%")
    
    fig_cone.update_layout(
        title="Volatility Term Structure",
        xaxis_title="Horizon (Days)",
        yaxis_title="Annualized Volatility (%)",
        template='plotly_dark',
        showlegend=True
    )
    
    return fig_paths, fig_dist, fig_cone

# Callback for updating metrics panels
@app.callback(
    [Output('risk-metrics-display', 'children'),
     Output('probability-analysis', 'children'),
     Output('quick-stats', 'children')],
    [Input('simulation-results', 'data')]
)
def update_metrics_panels(results):
    if not results:
        return html.Div("Run simulation to see metrics"), html.Div(), html.Div()
    
    metrics = results['metrics']
    
    # Risk Metrics Display
    risk_display = html.Div([
        create_metric_card("Expected Return", f"{metrics['expected_return']:.2f}%", 
                          metrics['expected_return'] > 0),
        create_metric_card("Volatility", f"{metrics['volatility']:.2f}%"),
        create_metric_card("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}",
                          metrics['sharpe_ratio'] > 0.5),
        create_metric_card("VaR (95%)", f"{metrics['var_95']:.2f}%",
                          metrics['var_95'] > -10, inverse=True),
        create_metric_card("CVaR (95%)", f"{metrics['cvar_95']:.2f}%",
                          metrics['cvar_95'] > -15, inverse=True),
        create_metric_card("Max Drawdown", f"{metrics['max_drawdown']:.2f}%",
                          metrics['max_drawdown'] > -20, inverse=True),
        create_metric_card("Skewness", f"{metrics['skewness']:.3f}",
                          metrics['skewness'] > 0),
        create_metric_card("Kurtosis", f"{metrics['kurtosis']:.3f}")
    ])
    
    # Probability Analysis
    prob_display = html.Div([
        create_metric_card("Profit Probability", f"{metrics['probability_profit']:.1f}%",
                          metrics['probability_profit'] > 50),
        create_metric_card("Loss >10%", f"{metrics['probability_loss_10pct']:.1f}%",
                          metrics['probability_loss_10pct'] < 20, inverse=True),
        html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)'}),
        html.H6("Price Targets", className="mb-3"),
        create_metric_card("25th Percentile", f"${results['current_price'] * (1 + metrics['percentile_25']/100):.2f}"),
        create_metric_card("Median", f"${results['current_price'] * (1 + metrics['median_return']/100):.2f}"),
        create_metric_card("75th Percentile", f"${results['current_price'] * (1 + metrics['percentile_75']/100):.2f}")
    ])
    
    # Quick Stats
    quick_stats = html.Div([
        html.P([html.Strong("Ticker: "), results['ticker']], className="mb-2"),
        html.P([html.Strong("Current Price: "), f"${results['current_price']:.2f}"], className="mb-2"),
        html.P([html.Strong("Simulations: "), f"{results['n_simulations']:,}"], className="mb-2"),
        html.P([html.Strong("Horizon: "), f"{results['horizon']} days"], className="mb-2"),
        html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)'}),
        html.H6("GARCH Parameters", className="mb-2"),
        html.P([html.Strong("α: "), f"{results['garch_params']['alpha']:.4f}"], className="mb-1"),
        html.P([html.Strong("β: "), f"{results['garch_params']['beta']:.4f}"], className="mb-1"),
        html.P([html.Strong("ω: "), f"{results['garch_params']['omega']:.6f}"], className="mb-1")
    ])
    
    return risk_display, prob_display, quick_stats

# Callback for bottom panels
@app.callback(
    [Output('backtest-accuracy', 'figure'),
     Output('event-calendar', 'children'),
     Output('pipeline-status', 'children')],
    [Input('stock-dropdown', 'value')]
)
def update_bottom_panels(ticker):
    # Backtest Accuracy Chart -- now driven by a REAL backtest, not random data
    accuracy_fig = go.Figure()

    try:
        prices_df = data_fetcher.get_prices_from_db(ticker)
        earnings_dates = data_fetcher.fetch_earnings_dates(ticker)
        if not prices_df.empty and earnings_dates:
            bt = EarningsVolatilityBacktest(train_window=250, post_window=5, tol=1.5)
            bt_result = bt.run(prices_df["close"], earnings_dates)
            detail = bt_result["detail"]
            real_accuracy = bt_result["accuracy_pct"]

            if not detail.empty:
                # Plot forecast vs realized vol per event
                accuracy_fig.add_trace(go.Scatter(
                    x=detail["earnings_date"].astype(str), y=detail["forecast_vol"],
                    mode="lines+markers", name="Forecast Vol",
                    line=dict(color="cyan", width=2)))
                accuracy_fig.add_trace(go.Scatter(
                    x=detail["earnings_date"].astype(str), y=detail["realized_vol"],
                    mode="lines+markers", name="Realized Vol",
                    line=dict(color="magenta", width=2)))
                title = (f"Post-Earnings Vol: forecast vs realized "
                         f"({real_accuracy:.0f}% in-band, n={bt_result['events_tested']})")
            else:
                title = "Backtest: not enough history for this ticker"
        else:
            title = "Backtest: load data + earnings first (run initialize_data.py)"
    except Exception as e:
        title = f"Backtest unavailable: {e}"

    accuracy_fig.update_layout(
        title=title,
        xaxis_title="Earnings Event",
        yaxis_title="Annualized Vol (%)",
        template='plotly_dark',
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Event Calendar
    try:
        earnings_dates = data_fetcher.fetch_earnings_dates(ticker)
        if earnings_dates:
            upcoming = next_event_date(earnings_dates, pd.Timestamp.now())
            if upcoming is not None:
                headline = f"Next Earnings: {upcoming.strftime('%Y-%m-%d')}"
            else:
                latest = max(_naive_day(d) for d in earnings_dates)
                headline = f"No upcoming date listed (latest: {latest.strftime('%Y-%m-%d')})"
            event_display = html.Div([
                html.P([html.I(className="fas fa-bell me-2"), headline],
                      className="mb-2", style={'color': 'yellow'}),
                html.P(f"{len(earnings_dates)} earnings dates on record for {ticker}",
                       className="mb-2", style={'fontSize': '12px'}),
            ])
        else:
            event_display = html.P("No upcoming earnings found", className="text-muted")
    except:
        event_display = html.P("Earnings data unavailable", className="text-muted")
    
    # Pipeline Status -- report the REAL row count from SQLite
    try:
        real_count = data_fetcher.get_record_count()
        db_detail = f"{real_count:,} records"
    except Exception:
        db_detail = "not initialized"

    pipeline_status = html.Div([
        create_status_indicator("Simulation Data", "success", "Yahoo Finance (5y) on each run"),
        create_status_indicator("Backtest Data", "success", "SQLite price history"),
        create_status_indicator("Database", "success", db_detail),
        html.Hr(style={'borderColor': 'rgba(255,255,255,0.1)'}),
        html.Small(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                  className="text-muted")
    ])
    
    return accuracy_fig, event_display, pipeline_status

# Helper Functions
def create_metric_card(label, value, positive=None, inverse=False):
    """Create a styled metric card"""
    if positive is not None:
        if inverse:
            color_class = "negative" if positive else "positive"
        else:
            color_class = "positive" if positive else "negative"
    else:
        color_class = ""
    
    return html.Div([
        html.P(label, className="metric-label"),
        html.P(value, className=f"metric-value {color_class}")
    ], className="metric-card")

def create_status_indicator(label, status, detail):
    """Create a status indicator with icon"""
    icons = {
        'success': 'fa-check-circle',
        'warning': 'fa-exclamation-triangle',
        'error': 'fa-times-circle'
    }
    colors = {
        'success': '#00ff88',
        'warning': '#ffd700',
        'error': '#ff4757'
    }
    
    return html.Div([
        html.I(className=f"fas {icons[status]} me-2", 
               style={'color': colors[status]}),
        html.Span(f"{label}: ", style={'fontWeight': 'bold'}),
        html.Span(detail, className="text-muted")
    ], className="mb-2")

def run_dashboard():
    """Run the enhanced dashboard"""
    app.run(debug=True, port=8050, host='0.0.0.0')

if __name__ == '__main__':
    run_dashboard()