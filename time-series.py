import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

colors = {
    'background': '#1f2630',        # dark background
    'paper': '#1f2630',             # same as background for a sleek look
    'text': '#ffffff',              # white text
    'original': '#00adb5',          # teal for original data
    'smoothed': '#ff2e63',          # vibrant pink for smoothed data
    'rolling_mean': '#ffd369',      # golden yellow for the rolling mean
    'threshold_range': 'rgba(255, 141, 0, 0.2)',  # transparent orange for threshold fill
    'anomalies': '#eeeeee'          # light gray for anomalies
}


def generate_test_data():
    """
    Generate artificial time series data for four drone motors over 5 years.
    Each motor signal is a combination of:
      - Annual seasonal variation (sine wave)
      - Weekly operational cycles (sine wave)
      - A motor-specific frequency component
      - A slight linear trend
      - Random noise
    """
    np.random.seed(42)
    # Generate dates over 5 years (approximately 1825 days)
    dates = pd.date_range(start='2020-01-01', periods=1825, freq='D')
    t = np.arange(len(dates))
    signals = {}
    
    for motor in ['motor1', 'motor2', 'motor3', 'motor4']:
        seasonal = np.sin(2 * np.pi * t / 365)
        weekly = 0.5 * np.sin(2 * np.pi * t / 7)
        freq_factor = 1 + 0.2 * np.random.rand()  # motor-specific frequency variation
        motor_specific = 0.7 * np.sin(2 * np.pi * t / (365 / freq_factor))
        trend = 0.001 * t
        noise = np.random.normal(0, 0.3, len(dates))
        signal = seasonal + weekly + motor_specific + trend + noise
        signals[motor] = signal

    dfs = {}
    for motor, values in signals.items():
        df = pd.DataFrame({'date': dates, 'value': values})
        df.set_index('date', inplace=True)
        # Smooth data using a moving average with a window of 10
        df['smoothed_value'] = df['value'].rolling(window=10, center=True).mean()
        dfs[motor] = df
    return dfs


def detect_anomalies(df, window_size, z_threshold):
    """
    Calculate rolling statistics and detect anomalies using a Z-score method.
    """
    df['rolling_mean'] = df['smoothed_value'].rolling(window=window_size).mean()
    df['rolling_std'] = df['smoothed_value'].rolling(window=window_size).std()
    df['z_score'] = (df['smoothed_value'] - df['rolling_mean']) / df['rolling_std']
    df['anomaly'] = np.abs(df['z_score']) > z_threshold
    return df


def score_anomaly_detection(df, optimal_proportion=0.05):
    """
    Score anomaly detection by comparing the proportion of anomalies to an optimal value.
    """
    proportion = df['anomaly'].sum() / len(df)
    return abs(proportion - optimal_proportion)


def optimize_window_size(dfs, z_threshold, window_range=range(5, 101, 5)):
    """
    Grid-search for the optimal rolling window size across all motor signals.
    """
    best_window = None
    best_score = float('inf')
    for window_size in window_range:
        scores = []
        for df in dfs.values():
            temp_df = detect_anomalies(df.copy(), window_size, z_threshold)
            scores.append(score_anomaly_detection(temp_df))
        avg_score = np.mean(scores)
        if avg_score < best_score:
            best_score = avg_score
            best_window = window_size
    return best_window


def plot_motor_data(dfs, z_threshold):
    """
    Create an interactive Plotly figure with subplots for each motor's data
    and save the result as an HTML file.
    """
    num_plots = len(dfs)
    fig = make_subplots(
        rows=num_plots, cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{motor.capitalize()} Time Series with Anomalies" for motor in dfs.keys()]
    )

    for i, (motor, df) in enumerate(dfs.items(), start=1):
        # Original data trace
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['value'],
                mode='lines',
                name='Original',
                line=dict(color=colors['original'])
            ),
            row=i, col=1
        )
        # Smoothed data trace
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['smoothed_value'],
                mode='lines',
                name='Smoothed',
                line=dict(color=colors['smoothed'], dash='dash')
            ),
            row=i, col=1
        )
        # Rolling mean trace
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['rolling_mean'],
                mode='lines',
                name='Rolling Mean',
                line=dict(color=colors['rolling_mean'])
            ),
            row=i, col=1
        )
        # Threshold range (upper bound)
        upper_bound = df['rolling_mean'] + z_threshold * df['rolling_std']
        lower_bound = df['rolling_mean'] - z_threshold * df['rolling_std']
        fig.add_trace(
            go.Scatter(
                x=df.index, y=upper_bound,
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=i, col=1
        )
        # Threshold range (fill to lower bound)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=lower_bound,
                mode='lines',
                line=dict(color='rgba(0,0,0,0)'),
                fill='tonexty',
                fillcolor=colors['threshold_range'],
                showlegend=False,
                hoverinfo='skip'
            ),
            row=i, col=1
        )
        # Anomaly markers
        anomalies_df = df[df['anomaly']]
        fig.add_trace(
            go.Scatter(
                x=anomalies_df.index, y=anomalies_df['smoothed_value'],
                mode='markers',
                name='Anomalies',
                marker=dict(color=colors['anomalies'], size=6)
            ),
            row=i, col=1
        )

    fig.update_layout(
        title="Drone Motors Time Series with Anomaly Detection",
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['paper'],
        font=dict(color=colors['text']),
        height=300 * num_plots,
        showlegend=True
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='gray')

    fig.write_html("fig.html")
    # fig.write_image("fig.png")
    return fig


# ====================== MAIN ======================
if __name__ == '__main__':
    # Generate test data for four drone motors over 5 years
    dfs = generate_test_data()

    # Parameters for anomaly detection
    z_threshold = 2.8

    # Optimize the rolling window size for anomaly detection
    optimal_window = optimize_window_size(dfs, z_threshold)
    print(f"Optimal Window Size: {optimal_window}")

    # Detect anomalies for each motor using the optimal window size
    for motor, df in dfs.items():
        dfs[motor] = detect_anomalies(df, optimal_window, z_threshold)

    # Create the Plotly figure and save it as "fig.html"
    fig = plot_motor_data(dfs, z_threshold)
