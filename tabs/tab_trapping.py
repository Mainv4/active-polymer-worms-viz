"""
Tab 3: Trapping Time Distributions
Visualize trapping time distributions from SQLite database with interactive controls.
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy import optimize
import streamlit as st

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

# Experimental data paths and constants
DB_EXP = Path(__file__).parent.parent / "exp_trapping_times.db"
TEMPERATURES_EXP = [10, 20, 30]
MAX_TRAP_TIME_EXP = 10.0  # minutes
TEMP_COLORS_MPL = {10: "#3498db", 20: "#2ecc71", 30: "#e74c3c"}


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


def exponential_pdf(x, tau):
    """Proper exponential PDF: (1/tau) * exp(-x/tau). Amplitude is constrained."""
    return (1.0 / tau) * np.exp(-x / tau)


def fit_exponential_log_space(times, bins):
    """
    Fit exponential decay using linear regression in log space.
    Original method from analyse_trapping_time.py

    This fits log(P) = a*τ + b, then extracts τ_c = -1/slope and A = exp(b).

    Args:
        times: Array of trapping times
        bins: Bin edges for histogram (e.g., np.linspace(0, 6, 13))

    Returns:
        (tau, tau_error, A) or (None, None, None) if fit fails
    """
    counts, bin_edges = np.histogram(times, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Only fit non-zero counts and positive bin centers
    mask = (counts > 0) & (bin_centers > 0)
    if np.sum(mask) < 3:
        return None, None, None

    log_counts = np.log(counts[mask])
    fit_centers = bin_centers[mask]

    try:
        # Linear fit: log(P) = slope*t + intercept
        def linear_func(x, a, b):
            return a * x + b

        popt, pcov = optimize.curve_fit(
            linear_func,
            fit_centers,
            log_counts,
            p0=[-1, 0]  # Initial guess: slope=-1, intercept=0
        )

        slope, intercept = popt
        tau = -1.0 / slope  # τ_c from slope
        A = np.exp(intercept)  # Amplitude

        # Error propagation: tau = -1/slope, so d(tau)/d(slope) = 1/slope^2
        if pcov[0, 0] > 0:
            tau_error = np.sqrt(pcov[0, 0]) / (slope ** 2)
        else:
            tau_error = 0

        return tau, tau_error, A
    except Exception:
        return None, None, None


def fit_exponential(times, n_bins=30, max_cutoff=None):
    """Fit proper exponential PDF to trapping time distribution.

    Uses constrained form (1/τ) * exp(-t/τ) where amplitude = 1/τ.
    This is the correct PDF for an exponential distribution.
    """
    if max_cutoff is not None:
        times = times[times <= max_cutoff]

    if len(times) < 10:
        return None, None

    # Create histogram with density=True for proper PDF
    counts, bin_edges = np.histogram(times, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Filter zero counts
    mask = counts > 0
    if np.sum(mask) < 3:
        return None, None

    try:
        # Fit the CONSTRAINED exponential PDF: (1/tau) * exp(-x/tau)
        # Only ONE free parameter: tau
        popt, pcov = optimize.curve_fit(
            exponential_pdf,
            bin_centers[mask],
            counts[mask],
            p0=[np.mean(times)],  # Initial guess = mean (MLE)
            bounds=(0.01, 100.0),  # tau must be positive
            maxfev=5000
        )
        tau = popt[0]
        tau_error = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 0
        return tau, tau_error
    except Exception:
        return None, None


@st.cache_data
def load_exp_trapping_data():
    """Load experimental trapping data from local database."""
    if not DB_EXP.exists():
        return None

    conn = sqlite3.connect(DB_EXP)
    import pandas as pd
    df = pd.read_sql("SELECT T_exp, length_mm, time_min FROM exp_trapping_events", conn)
    conn.close()
    return df


def _render_publication_style_figure(exp_data_filtered, n_bins, max_cutoff, normalize, fit_options, use_log_y):
    """Render publication-style figure with overlay distributions and τ vs T plot.

    Uses the same parameters as Histograms mode for consistency.
    Layout: overlaid distributions + τ vs T panel.
    """
    from plotly.subplots import make_subplots

    # Symbols for each temperature (matching the reference figure style)
    TEMP_SYMBOLS = {10: "square", 20: "circle", 30: "triangle-up"}

    # Use user-controlled bins (same as Histograms mode)
    bins = np.linspace(0, max_cutoff, n_bins + 1)

    # Compute statistics for each temperature using both fit methods
    stats = {}
    for temp in TEMPERATURES_EXP:
        df_temp = exp_data_filtered[exp_data_filtered["T_exp"] == temp]
        times = np.array(df_temp["time_min"].tolist())
        if len(times) > 0:
            mean_val = float(np.mean(times))
            se_mean = mean_val / np.sqrt(len(times))

            # Log-space fit
            tau_log, tau_log_error, A_log = fit_exponential_log_space(times, bins)
            # Direct fit
            tau_pdf, tau_pdf_error = fit_exponential(times, n_bins, max_cutoff)

            stats[temp] = {
                "times": times,
                "mean": mean_val,
                "se_mean": se_mean,
                "tau_log": tau_log if tau_log else mean_val,
                "tau_log_error": tau_log_error if tau_log_error else se_mean,
                "A_log": A_log if A_log else 1.0,
                "tau_pdf": tau_pdf,
                "tau_pdf_error": tau_pdf_error,
                "n": len(times)
            }

    # Create 2-panel figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["(c) P(τ) - Trapping time distributions", "(d) ⟨τ⟩ and τ_c vs T"],
        horizontal_spacing=0.12
    )

    # Panel 1: Overlaid distributions (scatter points from histogram)
    for temp in TEMPERATURES_EXP:
        if temp not in stats:
            continue

        times = stats[temp]["times"]

        # Create histogram with user-controlled bins and extract bin centers/heights
        counts, bin_edges = np.histogram(times, bins=bins, density=normalize)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Filter out zero counts for cleaner plot
        mask = counts > 0
        x_points = bin_centers[mask]
        y_points = counts[mask]

        # Scatter points for histogram (like original: marker="o", markersize=12)
        fig.add_trace(
            go.Scatter(
                x=x_points,
                y=y_points,
                mode="markers",
                name=f"T = {temp}°C (N={stats[temp]['n']})",
                marker=dict(
                    symbol=TEMP_SYMBOLS[temp],
                    size=12,
                    color=TEMP_COLORS_MPL[temp],
                    opacity=0.7,
                    line=dict(width=1, color="black")
                ),
                legendgroup=f"temp_{temp}",
                showlegend=True,
            ),
            row=1, col=1
        )

        # Exponential fit lines
        x_fit = np.linspace(0.01, max_cutoff, 200)

        # Log-space fit (dashed line)
        if "Log-space fit" in fit_options and stats[temp]["tau_log"] is not None:
            y_fit_log = stats[temp]["A_log"] * np.exp(-x_fit / stats[temp]["tau_log"])
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit_log,
                    mode="lines",
                    name=f"τ_log = {stats[temp]['tau_log']:.2f}",
                    line=dict(color=TEMP_COLORS_MPL[temp], width=2, dash="dash"),
                    legendgroup=f"temp_{temp}",
                    showlegend=False,
                ),
                row=1, col=1
            )

        # Direct fit (dotted line)
        if "Direct fit" in fit_options and stats[temp]["tau_pdf"] is not None:
            # Estimate amplitude from histogram
            counts, _ = np.histogram(times, bins=bins, density=normalize)
            a_fit = max(counts) if len(counts) > 0 else 1
            y_fit_pdf = exponential_decay(x_fit, a_fit, stats[temp]["tau_pdf"])
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit_pdf,
                    mode="lines",
                    name=f"τ_pdf = {stats[temp]['tau_pdf']:.2f}",
                    line=dict(color=TEMP_COLORS_MPL[temp], width=2, dash="dot"),
                    legendgroup=f"temp_{temp}",
                    showlegend=False,
                ),
                row=1, col=1
            )

    # Panel 2: τ_mean and τ_fit vs Temperature
    temps = [t for t in TEMPERATURES_EXP if t in stats]
    means = [stats[t]["mean"] for t in temps]
    se_means = [stats[t]["se_mean"] for t in temps]

    # Mean values (filled symbols) - ⟨τ⟩
    for i, temp in enumerate(temps):
        fig.add_trace(
            go.Scatter(
                x=[temp],
                y=[means[i]],
                mode="markers",
                name=f"⟨τ⟩ = {means[i]:.2f}",
                marker=dict(
                    symbol=TEMP_SYMBOLS[temp],
                    size=14,
                    color=TEMP_COLORS_MPL[temp],
                    line=dict(width=1, color="black")
                ),
                error_y=dict(type="data", array=[se_means[i]], visible=True, color=TEMP_COLORS_MPL[temp]),
                legendgroup=f"mean_{temp}",
                showlegend=True,
            ),
            row=1, col=2
        )

    # τ_log values (open symbols) - from log-space fit
    if "Log-space fit" in fit_options:
        for i, temp in enumerate(temps):
            tau_log = stats[temp]["tau_log"]
            tau_log_error = stats[temp]["tau_log_error"]
            if tau_log is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[temp],
                        y=[tau_log],
                        mode="markers",
                        name=f"τ_log = {tau_log:.2f}",
                        marker=dict(
                            symbol=f"{TEMP_SYMBOLS[temp]}-open",
                            size=14,
                            color=TEMP_COLORS_MPL[temp],
                            line=dict(width=2, color=TEMP_COLORS_MPL[temp])
                        ),
                        error_y=dict(type="data", array=[tau_log_error], visible=True, color=TEMP_COLORS_MPL[temp]),
                        legendgroup=f"log_{temp}",
                        showlegend=True,
                    ),
                    row=1, col=2
                )

    # τ_pdf values (cross symbols) - from Direct fit
    if "Direct fit" in fit_options:
        for i, temp in enumerate(temps):
            tau_pdf = stats[temp]["tau_pdf"]
            tau_pdf_error = stats[temp]["tau_pdf_error"]
            if tau_pdf is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[temp],
                        y=[tau_pdf],
                        mode="markers",
                        name=f"τ_pdf = {tau_pdf:.2f}",
                        marker=dict(
                            symbol="x",
                            size=12,
                            color=TEMP_COLORS_MPL[temp],
                            line=dict(width=2, color=TEMP_COLORS_MPL[temp])
                        ),
                        error_y=dict(type="data", array=[tau_pdf_error], visible=True, color=TEMP_COLORS_MPL[temp]),
                        legendgroup=f"pdf_{temp}",
                        showlegend=True,
                    ),
                    row=1, col=2
                )

    # Update layout
    fig.update_layout(
        height=450,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        ),
    )

    # Panel 1 axes - respect user settings
    fig.update_xaxes(title_text="τ (min)", range=[0, max_cutoff], row=1, col=1)
    y_title = "P(τ)" if normalize else "Count"
    if use_log_y:
        fig.update_yaxes(title_text=y_title, type="log", row=1, col=1)
    else:
        fig.update_yaxes(title_text=y_title, row=1, col=1)

    # Panel 2 axes
    fig.update_xaxes(title_text="T (°C)", row=1, col=2, tickvals=[10, 20, 30], range=[5, 35])
    fig.update_yaxes(title_text="⟨τ⟩, τ_c (min)", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

    # Statistics table
    st.markdown("### Statistics")
    import pandas as pd
    stats_data = []
    for temp in temps:
        s = stats[temp]
        row = {
            "Temperature": f"{temp}°C",
            "N events": s["n"],
            "⟨τ⟩ (min)": f"{s['mean']:.2f} ± {s['se_mean']:.2f}",
        }
        if "Log-space fit" in fit_options and s["tau_log"] is not None:
            row["τ (log-space)"] = f"{s['tau_log']:.2f} ± {s['tau_log_error']:.2f}"
        if "Direct fit" in fit_options and s["tau_pdf"] is not None:
            row["τ (direct)"] = f"{s['tau_pdf']:.2f} ± {s['tau_pdf_error']:.2f}"
        stats_data.append(row)
    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)


def _render_exp_trapping_subtab():
    """Render experimental trapping time distributions (all lengths combined)."""
    st.subheader("Experimental trapping time distributions")
    st.markdown("Distribution of trapping times combining all worm lengths, one panel per temperature.")

    # Load experimental data
    exp_data = load_exp_trapping_data()
    if exp_data is None or len(exp_data) == 0:
        st.error(f"Experimental database not found: {DB_EXP}")
        st.info("Run `python collect_exp_trapping_data.py` to create the database.")
        return

    # Display mode selector
    display_mode = st.radio(
        "Display mode",
        ["Histograms", "Summary"],
        horizontal=True,
        key="exp_trap_display_mode"
    )

    # Controls
    st.markdown("---")
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)

    with col_ctrl1:
        n_bins = st.slider("Number of bins", 5, 100, 30, key="exp_trap_bins")

    with col_ctrl2:
        max_cutoff = st.slider("Max τ cutoff (min)", 1.0, 15.0, MAX_TRAP_TIME_EXP, step=0.5, key="exp_trap_cutoff")

    with col_ctrl3:
        normalize = st.checkbox("Normalize (density)", value=True, key="exp_trap_normalize")

    with col_ctrl4:
        fit_options = st.multiselect(
            "Fit method(s)",
            ["Log-space fit", "Direct fit"],
            default=["Log-space fit"],
            key="exp_trap_fit_options",
            help="Log-space: régression linéaire sur log(P), robuste. Direct: fit sur P(τ) en échelle linéaire."
        )

    col_ctrl5, col_ctrl6, col_ctrl7, _ = st.columns(4)
    with col_ctrl5:
        use_log_y = st.checkbox("Log scale (Y-axis)", value=False, key="exp_trap_log_y")

    with col_ctrl6:
        show_mean = st.checkbox("Show mean line", value=True, key="exp_trap_mean")

    with col_ctrl7:
        temp_selected = st.selectbox(
            "Temperature",
            options=["All"] + TEMPERATURES_EXP,
            index=0,
            format_func=lambda x: "All temperatures" if x == "All" else f"{x}°C",
            key="exp_trap_temp",
            disabled=(display_mode == "Summary")
        )

    # Filter by cutoff
    exp_data_filtered = exp_data[exp_data["time_min"] <= max_cutoff]

    # Publication style view
    if display_mode == "Summary":
        _render_publication_style_figure(exp_data_filtered, n_bins, max_cutoff, normalize, fit_options, use_log_y)
        return

    if temp_selected == "All":
        # Create 3-panel figure with subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[f"T = {t}°C" for t in TEMPERATURES_EXP],
            horizontal_spacing=0.08
        )

        histnorm = "probability density" if normalize else None

        for idx, temp in enumerate(TEMPERATURES_EXP):
            col_idx = idx + 1
            df_temp = exp_data_filtered[exp_data_filtered["T_exp"] == temp]
            times = np.array(df_temp["time_min"].tolist())

            if len(times) == 0:
                continue

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=times,
                    nbinsx=n_bins,
                    histnorm=histnorm,
                    name=f"T={temp}°C (N={len(times)})",
                    marker_color=TEMP_COLORS_MPL[temp],
                    opacity=0.7,
                    showlegend=True,
                ),
                row=1, col=col_idx
            )

            # Fit estimates
            mean_val = float(np.mean(times))
            x_fit = np.linspace(0.01, max_cutoff, 200)
            bins = np.linspace(0, max_cutoff, n_bins + 1)

            # 1. Log-space fit
            if "Log-space fit" in fit_options and len(times) >= 10:
                tau_log, tau_log_error, A_log = fit_exponential_log_space(times, bins)
                if tau_log is not None:
                    y_fit_log = A_log * np.exp(-x_fit / tau_log)
                    fig.add_trace(
                        go.Scatter(
                            x=x_fit,
                            y=y_fit_log,
                            mode="lines",
                            name=f"τ_log={tau_log:.2f}±{tau_log_error:.2f}",
                            line=dict(color="darkred", width=2, dash="dash"),
                            showlegend=True,
                        ),
                        row=1, col=col_idx
                    )

            # 2. Direct fit
            if "Direct fit" in fit_options and len(times) >= 10:
                tau_pdf, tau_pdf_error = fit_exponential(times, n_bins, max_cutoff)
                if tau_pdf is not None:
                    counts, _ = np.histogram(times, bins=n_bins, density=normalize)
                    a_fit = max(counts) if len(counts) > 0 else 1
                    y_fit_pdf = exponential_decay(x_fit, a_fit, tau_pdf)
                    fig.add_trace(
                        go.Scatter(
                            x=x_fit,
                            y=y_fit_pdf,
                            mode="lines",
                            name=f"τ_pdf={tau_pdf:.2f}±{tau_pdf_error:.2f}",
                            line=dict(color="green", width=2, dash="dot"),
                            showlegend=True,
                        ),
                        row=1, col=col_idx
                    )

            # 3. Mean line (MLE estimate)
            if show_mean:
                # Also show exponential curve based on mean
                if normalize:
                    y_mean = (1.0 / mean_val) * np.exp(-x_fit / mean_val)
                else:
                    bin_width = max_cutoff / n_bins
                    y_mean = len(times) * bin_width * (1.0 / mean_val) * np.exp(-x_fit / mean_val)

                fig.add_trace(
                    go.Scatter(
                        x=x_fit,
                        y=y_mean,
                        mode="lines",
                        name=f"MLE: τ=μ={mean_val:.2f}",
                        line=dict(color="black", width=2, dash="dot"),
                        showlegend=True,
                    ),
                    row=1, col=col_idx
                )

        fig.update_layout(
            height=450,
            template="plotly_white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        )

        for col_idx in range(1, 4):
            fig.update_xaxes(title_text="τ (min)", row=1, col=col_idx)
            if col_idx == 1:
                fig.update_yaxes(title_text="P(τ)" if normalize else "Count", row=1, col=col_idx)
            if use_log_y:
                fig.update_yaxes(type="log", row=1, col=col_idx)

    else:
        # Single temperature view
        df_temp = exp_data_filtered[exp_data_filtered["T_exp"] == temp_selected]
        times = np.array(df_temp["time_min"].tolist())

        if len(times) == 0:
            st.warning(f"No data for T={temp_selected}°C after cutoff filter.")
            return

        fig = go.Figure()
        histnorm = "probability density" if normalize else None

        fig.add_trace(go.Histogram(
            x=times,
            nbinsx=n_bins,
            histnorm=histnorm,
            name=f"T={temp_selected}°C (N={len(times)})",
            marker_color=TEMP_COLORS_MPL[temp_selected],
            opacity=0.7
        ))

        # Fit estimates
        mean_val = float(np.mean(times))
        x_fit = np.linspace(0.01, max_cutoff, 200)
        bins = np.linspace(0, max_cutoff, n_bins + 1)

        # Track fit results for statistics
        tau_log, tau_log_error = None, None
        tau_pdf, tau_pdf_error = None, None

        # 1. Log-space fit
        if "Log-space fit" in fit_options and len(times) >= 10:
            tau_log, tau_log_error, A_log = fit_exponential_log_space(times, bins)
            if tau_log is not None:
                y_fit_log = A_log * np.exp(-x_fit / tau_log)
                fig.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit_log,
                    mode="lines",
                    name=f"τ_log = {tau_log:.2f} ± {tau_log_error:.2f} min",
                    line=dict(color="red", width=2, dash="dash")
                ))

        # 2. Direct fit
        if "Direct fit" in fit_options and len(times) >= 10:
            tau_pdf, tau_pdf_error = fit_exponential(times, n_bins, max_cutoff)
            if tau_pdf is not None:
                counts, _ = np.histogram(times, bins=n_bins, density=normalize)
                a_fit = max(counts) if len(counts) > 0 else 1
                y_fit_pdf = exponential_decay(x_fit, a_fit, tau_pdf)
                fig.add_trace(go.Scatter(
                    x=x_fit,
                    y=y_fit_pdf,
                    mode="lines",
                    name=f"τ_pdf = {tau_pdf:.2f} ± {tau_pdf_error:.2f} min",
                    line=dict(color="green", width=2, dash="dot")
                ))

        # 3. Mean (MLE estimate)
        if show_mean:
            if normalize:
                y_mean = (1.0 / mean_val) * np.exp(-x_fit / mean_val)
            else:
                bin_width = max_cutoff / n_bins
                y_mean = len(times) * bin_width * (1.0 / mean_val) * np.exp(-x_fit / mean_val)

            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_mean,
                mode="lines",
                name=f"MLE: τ = μ = {mean_val:.2f} min",
                line=dict(color="black", width=2, dash="dot")
            ))

        fig.update_layout(
            title=f"T = {temp_selected}°C",
            xaxis_title="τ (min)",
            yaxis_title="P(τ)" if normalize else "Count",
            height=500,
            template="plotly_white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        if use_log_y:
            fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    st.markdown("### Statistics")
    import pandas as pd
    stats_data = []
    for temp in TEMPERATURES_EXP:
        df_temp = exp_data_filtered[exp_data_filtered["T_exp"] == temp]
        if len(df_temp) > 0:
            times = np.array(df_temp["time_min"].tolist())
            mean_val = float(np.mean(times))
            se_mean = mean_val / np.sqrt(len(times))
            bins = np.linspace(0, max_cutoff, n_bins + 1)

            # Compute both fits
            tau_log, tau_log_error, _ = fit_exponential_log_space(times, bins) if "Log-space fit" in fit_options else (None, None, None)
            tau_pdf, tau_pdf_error = fit_exponential(times, n_bins, max_cutoff) if "Direct fit" in fit_options else (None, None)

            stats_data.append({
                "Temperature": f"{temp}°C",
                "N events": len(times),
                "Mean (μ)": f"{mean_val:.2f} ± {se_mean:.2f}",
                "Median": f"{float(np.median(times)):.2f}",
                "Std": f"{float(np.std(times)):.2f}",
                "τ (log-space)": f"{tau_log:.2f} ± {tau_log_error:.2f}" if tau_log else "N/A",
                "τ (direct)": f"{tau_pdf:.2f} ± {tau_pdf_error:.2f}" if tau_pdf else "N/A",
            })
    if stats_data:
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)


def _render_sim_trapping_subtab():
    """Render simulation trapping time distributions."""
    st.subheader("Simulation trapping time distributions")
    st.markdown("Explore trapping time distributions from confinement simulations (N=40, 50, 60).")

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
        fit_options = st.multiselect(
            "Fit method(s)",
            ["Log-space fit", "Direct fit"],
            default=["Log-space fit"],
            help="Log-space: régression linéaire sur log(P), robuste. Direct: fit sur P(τ) en échelle linéaire."
        )

    # Additional options
    col_ctrl5, _, _, _ = st.columns(4)
    with col_ctrl5:
        use_log_y = st.checkbox("Log scale (Y-axis)", value=False, key="trapping_log_y")

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
    tau_log = None
    tau_log_error = None
    tau_pdf = None
    tau_pdf_error = None
    x_fit = np.linspace(0, max_cutoff, 200)
    bins = np.linspace(0, max_cutoff, n_bins + 1)

    # Log-space fit
    if "Log-space fit" in fit_options:
        tau_log, tau_log_error, A_log = fit_exponential_log_space(times_filtered, bins)
        if tau_log is not None:
            y_fit_log = A_log * np.exp(-x_fit / tau_log)
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit_log,
                mode="lines",
                name=f"Log-space: τ = {tau_log:.2f} ± {tau_log_error:.2f} min",
                line=dict(color="red", width=2, dash="dash")
            ))

    # Direct fit
    if "Direct fit" in fit_options:
        tau_pdf, tau_pdf_error = fit_exponential(times_filtered, n_bins, max_cutoff)
        if tau_pdf is not None:
            counts, _ = np.histogram(times_filtered, bins=n_bins, density=normalize)
            a_fit = max(counts) if len(counts) > 0 else 1
            y_fit_pdf = exponential_decay(x_fit, a_fit, tau_pdf)
            fig.add_trace(go.Scatter(
                x=x_fit,
                y=y_fit_pdf,
                mode="lines",
                name=f"Direct fit: τ = {tau_pdf:.2f} ± {tau_pdf_error:.2f} min",
                line=dict(color="green", width=2, dash="dot")
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
    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)

    with col_stat1:
        st.metric("N events", len(times_filtered))
    with col_stat2:
        st.metric("Mean", f"{np.mean(times_filtered):.2f} min")
    with col_stat3:
        st.metric("Median", f"{np.median(times_filtered):.2f} min")
    with col_stat4:
        if tau_log is not None:
            st.metric("τ (log-space)", f"{tau_log:.2f} ± {tau_log_error:.2f} min")
        else:
            st.metric("τ (log-space)", "N/A")
    with col_stat5:
        if tau_pdf is not None:
            st.metric("τ (direct)", f"{tau_pdf:.2f} ± {tau_pdf_error:.2f} min")
        else:
            st.metric("τ (direct)", "N/A")


def render_trapping_tab():
    """Render the Trapping Time Distributions tab with sub-tabs."""
    st.header("Trapping Time Distributions")

    # Sub-tabs for simulation and experimental data
    tab_sim, tab_exp = st.tabs(["Simulation", "Experimental (all lengths)"])

    with tab_sim:
        _render_sim_trapping_subtab()

    with tab_exp:
        _render_exp_trapping_subtab()
