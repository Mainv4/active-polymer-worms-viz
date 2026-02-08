"""
Tab 4: τ_trap Parameter Space
Visualize τ_mean and τ_fit as functions of simulation parameters.
"""

import sqlite3
from pathlib import Path

import numpy as np
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import optimize
import streamlit as st

# Data quality filter (consistent with compile_all_data.py)
MIN_TRAPPING_EVENTS = 5


def get_db_path():
    """Get path to trapping times database."""
    return Path(__file__).parent.parent / "trapping_times.db"


def exponential_decay(x, a, tau):
    """Exponential decay function: a * exp(-x/tau)"""
    return a * np.exp(-x / tau)


def fit_exponential(times, n_bins=30):
    """Fit exponential decay to trapping time distribution."""
    if len(times) < 10:
        return None, None

    counts, bin_edges = np.histogram(times, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mask = counts > 0
    if np.sum(mask) < 3:
        return None, None

    try:
        popt, pcov = optimize.curve_fit(
            exponential_decay,
            bin_centers[mask],
            counts[mask],
            p0=[max(counts), np.mean(times)],
            maxfev=5000
        )
        tau_error = np.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0
        return popt[1], tau_error
    except Exception:
        return None, None


@st.cache_data
def load_parameter_sets(N):
    """Load available parameter sets for a given N."""
    db_path = get_db_path()
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT Pe, T, kappa FROM parameter_sets
        WHERE N = ? AND n_events >= ?
        ORDER BY Pe, T, kappa
    """, (N, MIN_TRAPPING_EVENTS))
    rows = cursor.fetchall()
    conn.close()
    return rows


@st.cache_data
def load_trapping_times(N, Pe, T, kappa):
    """Load trapping times for specific parameters."""
    db_path = get_db_path()
    if not db_path.exists():
        return np.array([])

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT time_min FROM trapping_events
        WHERE N = ? AND ABS(Pe - ?) < 0.001 AND ABS(T - ?) < 0.001 AND ABS(kappa - ?) < 0.001
    """, (N, Pe, T, kappa))
    times = np.array([row[0] for row in cursor.fetchall()])
    conn.close()
    return times


@st.cache_data
def compute_all_tau_stats(N, n_bins, max_cutoff):
    """Compute τ_mean and τ_fit for all parameter combinations of a given N."""
    param_sets = load_parameter_sets(N)
    results = []

    for Pe, T, kappa in param_sets:
        times = load_trapping_times(N, Pe, T, kappa)
        if len(times) == 0:
            continue

        # Apply cutoff
        times_filtered = times[times <= max_cutoff]
        if len(times_filtered) < 5:
            continue

        tau_mean = np.mean(times_filtered)
        tau_median = np.median(times_filtered)
        tau_fit, tau_fit_error = fit_exponential(times_filtered, n_bins)

        results.append({
            'Pe': Pe,
            'T': T,
            'kappa': kappa,
            'tau_mean': tau_mean,
            'tau_median': tau_median,
            'tau_fit': tau_fit,
            'tau_fit_error': tau_fit_error,
            'n_events': len(times_filtered)
        })

    return results


@st.cache_data
def get_available_N():
    """Get list of available N values in database."""
    db_path = get_db_path()
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT N FROM parameter_sets WHERE n_events >= ? ORDER BY N", (MIN_TRAPPING_EVENTS,))
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]


@st.cache_data
def get_max_time(N):
    """Get maximum trapping time for a given N."""
    db_path = get_db_path()
    if not db_path.exists():
        return 100.0

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(time_min) FROM trapping_events WHERE N = ?", (N,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result[0] else 100.0


def render_tau_params_tab():
    """Render the τ_trap Parameter Space tab."""
    st.header("τ_trap Parameter Space")
    st.markdown("""
    Compare τ_mean and τ_fit across parameter space for different temperatures.
    """)

    # Check database exists
    db_path = get_db_path()
    if not db_path.exists():
        st.error(f"Database not found: {db_path}")
        st.info("Run `python collect_trapping_data.py` to create the database.")
        return

    # Get available N values
    available_N = get_available_N()
    if not available_N:
        st.warning("No data available in database.")
        return

    # Top controls row
    col_n, col_bins, col_cutoff = st.columns(3)

    with col_n:
        N = st.selectbox("Polymer length (N)", available_N, index=0, key="tau_N")

    max_time = get_max_time(N)

    with col_bins:
        n_bins = st.slider("Number of bins (for fit)", 5, 100, 30, key="tau_params_bins")

    with col_cutoff:
        max_cutoff = st.slider("Max τ cutoff (min)", 1.0, float(max_time), min(50.0, float(max_time)), step=1.0, key="tau_params_cutoff")

    # Compute stats
    stats = compute_all_tau_stats(N, n_bins, max_cutoff)

    if not stats:
        st.warning("No valid data after applying filters.")
        return

    # Observable and display controls
    col1, col2, col3 = st.columns(3)

    with col1:
        observable = st.radio(
            "Observable",
            ["τ_mean", "τ_fit"],
            horizontal=True,
            key="tau_observable"
        )
        obs_key = "tau_mean" if observable == "τ_mean" else "tau_fit"

    with col2:
        x_param = st.selectbox(
            "X-axis Parameter",
            ["Pe", "kappa"],
            index=0,
            key="tau_x_param"
        )

    with col3:
        group_by = st.selectbox(
            "Group lines by",
            ["kappa", "Pe"],
            index=0 if x_param == "Pe" else 1,
            key="tau_group_by"
        )

    # Display mode
    display_mode = st.radio(
        "Display mode",
        ["4 temperatures", "Single temperature"],
        horizontal=True,
        key="tau_display_mode"
    )

    selected_temp = 0.1
    if display_mode == "Single temperature":
        available_temps = sorted(set(s['T'] for s in stats))
        selected_temp = st.selectbox(
            "Select temperature",
            available_temps,
            index=available_temps.index(0.1) if 0.1 in available_temps else 0,
            key="tau_temp"
        )

    # Plot options
    opt_col1, opt_col2, opt_col3 = st.columns(3)

    with opt_col1:
        use_log_y = st.checkbox("Log scale (Y-axis)", value=False, key="tau_log_y")

    with opt_col2:
        color_scheme_type = st.selectbox(
            "Color scheme",
            ["Discrete", "Gradient"],
            index=0,
            key="tau_color_scheme"
        )
        if color_scheme_type == "Gradient":
            color_palette = st.selectbox(
                "Color scale",
                ["Viridis", "Plasma", "Turbo", "Inferno"],
                index=0,
                key="tau_gradient_palette"
            )
        else:
            color_palette = "Alphabet"

    with opt_col3:
        show_error_bars = st.checkbox("Show error bars (τ_fit only)", value=True, key="tau_error_bars")

    # Get unique group values
    group_values = sorted(set(s[group_by] for s in stats))

    # Color mapping
    if color_scheme_type == "Discrete":
        colors = px.colors.qualitative.Alphabet
        color_map = {val: colors[i % len(colors)] for i, val in enumerate(group_values)}
    else:
        if len(group_values) > 1:
            min_val = min(group_values)
            max_val = max(group_values)
            normalized_vals = [(val - min_val) / (max_val - min_val) for val in group_values]
        else:
            normalized_vals = [0.5]
        color_map = {}
        for i, val in enumerate(group_values):
            color_rgb = pc.sample_colorscale(color_palette.lower(), [normalized_vals[i]])[0]
            color_map[val] = color_rgb

    # Create plot
    if display_mode == "4 temperatures":
        _render_3_temps(stats, obs_key, observable, x_param, group_by, group_values,
                        color_map, use_log_y, show_error_bars)
    else:
        _render_single_temp(stats, selected_temp, obs_key, observable, x_param, group_by, group_values,
                            color_map, use_log_y, show_error_bars)


def _render_3_temps(stats, obs_key, obs_label, x_param, group_by, group_values,
                    color_map, use_log_y, show_error_bars):
    """Render 4-temperature mode with subplots."""
    temps = [0.05, 0.1, 0.2, 0.3]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[f"T = {T}" for T in temps],
        horizontal_spacing=0.06
    )

    for col_idx, temp in enumerate(temps, 1):
        stats_temp = [s for s in stats if abs(s['T'] - temp) < 0.01]

        if not stats_temp:
            continue

        for group_val in group_values:
            stats_group = [s for s in stats_temp if abs(s[group_by] - group_val) < 0.001]

            if not stats_group:
                continue

            # Filter out None values for tau_fit
            if obs_key == "tau_fit":
                stats_group = [s for s in stats_group if s['tau_fit'] is not None]

            if not stats_group:
                continue

            # Sort by x_param
            stats_group = sorted(stats_group, key=lambda s: s[x_param])

            x_vals = [s[x_param] for s in stats_group]
            y_vals = [s[obs_key] for s in stats_group]

            error_y_dict = None
            if show_error_bars and obs_key == "tau_fit":
                errors = [s['tau_fit_error'] if s['tau_fit_error'] else 0 for s in stats_group]
                if any(e > 0 for e in errors):
                    error_y_dict = dict(type='data', array=errors, visible=True)

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name=f"{group_by}={group_val}",
                    line=dict(color=color_map[group_val], width=2),
                    marker=dict(size=8, color=color_map[group_val]),
                    error_y=error_y_dict,
                    connectgaps=True,
                    showlegend=(col_idx == 1),
                    legendgroup=str(group_val),
                    customdata=np.array([[s['Pe'], s['T'], s['kappa'], s['n_events']] for s in stats_group]),
                    hovertemplate='<b>Pe</b>: %{customdata[0]:.2f}<br>' +
                                  '<b>T</b>: %{customdata[1]:.2f}<br>' +
                                  '<b>κ</b>: %{customdata[2]:.2f}<br>' +
                                  '<b>N events</b>: %{customdata[3]}<br>' +
                                  f'<b>{x_param}</b>: %{{x:.2f}}<br>' +
                                  f'<b>{obs_label}</b>: %{{y:.2f}} min<extra></extra>',
                ),
                row=1, col=col_idx
            )

    fig.update_layout(
        height=500,
        template="plotly_white",
        font=dict(size=12),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02)
    )

    for col_idx in range(1, 5):
        fig.update_xaxes(title_text=x_param, row=1, col=col_idx)
        if col_idx == 1:
            fig.update_yaxes(title_text=f"{obs_label} (min)", row=1, col=col_idx)
        if use_log_y:
            fig.update_yaxes(type="log", row=1, col=col_idx)

    st.plotly_chart(fig, use_container_width=True)


def _render_single_temp(stats, selected_temp, obs_key, obs_label, x_param, group_by, group_values,
                        color_map, use_log_y, show_error_bars):
    """Render single-temperature mode."""
    stats_temp = [s for s in stats if abs(s['T'] - selected_temp) < 0.01]

    if not stats_temp:
        st.warning(f"No data available for T={selected_temp}")
        return

    fig = go.Figure()

    for group_val in group_values:
        stats_group = [s for s in stats_temp if abs(s[group_by] - group_val) < 0.001]

        if not stats_group:
            continue

        # Filter out None values for tau_fit
        if obs_key == "tau_fit":
            stats_group = [s for s in stats_group if s['tau_fit'] is not None]

        if not stats_group:
            continue

        # Sort by x_param
        stats_group = sorted(stats_group, key=lambda s: s[x_param])

        x_vals = [s[x_param] for s in stats_group]
        y_vals = [s[obs_key] for s in stats_group]

        error_y_dict = None
        if show_error_bars and obs_key == "tau_fit":
            errors = [s['tau_fit_error'] if s['tau_fit_error'] else 0 for s in stats_group]
            if any(e > 0 for e in errors):
                error_y_dict = dict(type='data', array=errors, visible=True)

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines+markers",
                name=f"{group_by}={group_val}",
                line=dict(color=color_map[group_val], width=2),
                marker=dict(size=8, color=color_map[group_val]),
                error_y=error_y_dict,
                connectgaps=True,
                customdata=np.array([[s['Pe'], s['T'], s['kappa'], s['n_events']] for s in stats_group]),
                hovertemplate='<b>Pe</b>: %{customdata[0]:.2f}<br>' +
                              '<b>T</b>: %{customdata[1]:.2f}<br>' +
                              '<b>κ</b>: %{customdata[2]:.2f}<br>' +
                              '<b>N events</b>: %{customdata[3]}<br>' +
                              f'<b>{x_param}</b>: %{{x:.2f}}<br>' +
                              f'<b>{obs_label}</b>: %{{y:.2f}} min<extra></extra>',
            )
        )

    fig.update_layout(
        title=f"T = {selected_temp}",
        xaxis_title=x_param,
        yaxis_title=f"{obs_label} (min)",
        height=500,
        template="plotly_white",
        font=dict(size=12),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02)
    )

    if use_log_y:
        fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)
