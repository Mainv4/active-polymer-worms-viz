"""
Tab 3: Trapping Time Distributions
Visualize trapping time distributions from SQLite database with interactive controls.
"""

import sqlite3
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy import optimize
import streamlit as st


def get_db_path():
    """Get path to trapping times database."""
    return Path(__file__).parent.parent / "trapping_times.db"


@st.cache_data
def load_parameter_index():
    """Load available parameter combinations from database."""
    db_path = get_db_path()
    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT N, Pe, T, kappa, n_events
        FROM parameter_sets
        ORDER BY N, Pe, T, kappa
    """)
    rows = cursor.fetchall()
    conn.close()

    # Build index: {N: {Pe: {T: [kappa values]}}}
    index = {}
    for N, Pe, T, kappa, n_events in rows:
        if N not in index:
            index[N] = {}
        if Pe not in index[N]:
            index[N][Pe] = {}
        if T not in index[N][Pe]:
            index[N][Pe][T] = []
        index[N][Pe][T].append((kappa, n_events))

    return index


@st.cache_data
def load_trapping_times(N, Pe, T, kappa):
    """Load trapping times for specific parameters from database."""
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


def exponential_decay(x, a, tau):
    """Exponential decay function: a * exp(-x/tau)"""
    return a * np.exp(-x / tau)


def fit_exponential(times, n_bins=30, max_cutoff=None):
    """Fit exponential decay to trapping time distribution."""
    if max_cutoff is not None:
        times = times[times <= max_cutoff]

    if len(times) < 10:
        return None, None

    # Create histogram
    counts, bin_edges = np.histogram(times, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Filter zero counts
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


def render_trapping_tab():
    """Render the Trapping Time Distributions tab."""
    st.header("Trapping Time Distributions")
    st.markdown("""
    Explore trapping time distributions from confinement simulations.
    Data from N=40, 50, 60 polymer lengths.
    """)

    # Check database exists
    db_path = get_db_path()
    if not db_path.exists():
        st.error(f"Database not found: {db_path}")
        st.info("Run `python collect_trapping_data.py` to create the database.")
        return

    # Load parameter index
    param_index = load_parameter_index()
    if param_index is None or len(param_index) == 0:
        st.warning("No data available in database.")
        return

    # Cascading selectors
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        available_N = sorted(param_index.keys())
        N = st.selectbox("Polymer length (N)", available_N, index=0)

    with col2:
        available_Pe = sorted(param_index[N].keys()) if N in param_index else []
        Pe = st.selectbox("Pe", available_Pe, index=0 if available_Pe else None)

    with col3:
        available_T = sorted(param_index[N][Pe].keys()) if N in param_index and Pe in param_index[N] else []
        T = st.selectbox("T", available_T, index=0 if available_T else None)

    with col4:
        kappa_options = param_index[N][Pe][T] if N in param_index and Pe in param_index[N] and T in param_index[N][Pe] else []
        kappa_labels = [f"{k:.2f} ({n} events)" for k, n in kappa_options]
        kappa_idx = st.selectbox("κ", range(len(kappa_labels)),
                                  format_func=lambda i: kappa_labels[i] if i < len(kappa_labels) else "",
                                  index=0 if kappa_labels else None)
        kappa = kappa_options[kappa_idx][0] if kappa_options and kappa_idx is not None else None

    if kappa is None:
        st.warning("No valid parameter combination selected.")
        return

    # Load data
    times = load_trapping_times(N, Pe, T, kappa)

    if len(times) == 0:
        st.warning("No trapping events for this parameter combination.")
        return

    # Histogram controls
    st.markdown("---")
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)

    with col_ctrl1:
        n_bins = st.slider("Number of bins", 5, 100, 30)

    with col_ctrl2:
        max_time = float(np.max(times))
        max_cutoff = st.slider("Max τ cutoff (min)", 0.0, max_time, max_time, step=0.5)

    with col_ctrl3:
        normalize = st.checkbox("Normalize (density)", value=True)

    with col_ctrl4:
        show_fit = st.checkbox("Show exponential fit", value=True)

    # Additional options
    col_ctrl5, col_ctrl6, _, _ = st.columns(4)
    with col_ctrl5:
        use_log_y = st.checkbox("Log scale (Y-axis)", value=False)

    # Filter times
    times_filtered = times[times <= max_cutoff]

    if len(times_filtered) == 0:
        st.warning("No events after cutoff filter.")
        return

    # Create histogram
    fig = go.Figure()

    histnorm = "probability density" if normalize else None
    fig.add_trace(go.Histogram(
        x=times_filtered,
        nbinsx=n_bins,
        histnorm=histnorm,
        name="Distribution",
        marker_color="steelblue",
        opacity=0.7
    ))

    # Fit and overlay
    tau_fit = None
    tau_error = None
    if show_fit:
        tau_fit, tau_error = fit_exponential(times_filtered, n_bins, max_cutoff)
        if tau_fit is not None:
            # Generate fit curve
            x_fit = np.linspace(0, max_cutoff, 200)
            # Get histogram normalization factor
            counts, bin_edges = np.histogram(times_filtered, bins=n_bins, density=normalize)
            a_fit = max(counts) if len(counts) > 0 else 1
            y_fit = exponential_decay(x_fit, a_fit, tau_fit)

            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                name=f"Fit: τ = {tau_fit:.2f} ± {tau_error:.2f} min",
                line=dict(color="red", width=2, dash="dash")
            ))

    # Update layout
    fig.update_layout(
        title=f"N={N}, Pe={Pe}, T={T}, κ={kappa:.2f}",
        xaxis_title="τ (min)",
        yaxis_title="P(τ)" if normalize else "Count",
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    if use_log_y:
        fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    st.markdown("### Statistics")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        st.metric("N events", len(times_filtered))
    with col_stat2:
        st.metric("Mean", f"{np.mean(times_filtered):.2f} min")
    with col_stat3:
        st.metric("Median", f"{np.median(times_filtered):.2f} min")
    with col_stat4:
        if tau_fit is not None:
            st.metric("τ (fit)", f"{tau_fit:.2f} ± {tau_error:.2f} min")
        else:
            st.metric("τ (fit)", "N/A")
