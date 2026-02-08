"""
Tab 5: Violin plots for trapping times.
Compare experimental and simulation trapping time distributions.
Experimental data grouped by worm length, simulation overlay for N=40,50,60.
"""

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
import streamlit as st
from matplotlib.lines import Line2D

# =============================================================================
# PATHS
# =============================================================================

DB_EXP = Path(__file__).parent.parent / "exp_trapping_times.db"
DB_SIM = Path(__file__).parent.parent / "trapping_times.db"

# Constants
MAX_TRAP_TIME = 10.0  # minutes
MAX_TRANS_TIME = 25.0  # minutes (for translocation)
TEMPERATURES_EXP = [10, 20, 30]  # Experimental temperatures in Celsius
LENGTH_THRESHOLD = 2.0  # mm - for grouping worms with similar lengths

# Colors matching reference figure (Plotly rgba format)
TEMP_COLORS_PLOTLY = {
    10: "rgba(52, 152, 219, 0.5)",  # Blue (#3498db)
    20: "rgba(46, 204, 113, 0.5)",  # Green (#2ecc71)
    30: "rgba(231, 76, 60, 0.5)",  # Red (#e74c3c)
}
SIM_COLOR_PLOTLY = "rgba(255, 165, 0, 0.7)"  # Orange for simulation

# Matplotlib colors (matching reference figure)
TEMP_COLORS_MPL = {10: "#3498db", 20: "#2ecc71", 30: "#e74c3c"}  # Blue  # Green  # Red
SIM_COLOR_MPL = "#FFA500"  # Orange


# =============================================================================
# DATA LOADING
# =============================================================================


@st.cache_data
def load_exp_trapping_data():
    """Load experimental trapping data from local database."""
    if not DB_EXP.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_EXP)
    df = pd.read_sql("SELECT T_exp, length_mm, time_min FROM exp_trapping_events", conn)
    conn.close()
    return df


@st.cache_data
def load_exp_translocation_data():
    """Load experimental translocation data from local database (success only)."""
    if not DB_EXP.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_EXP)
    try:
        df = pd.read_sql(
            "SELECT T_exp, length_mm, time_min FROM exp_translocation_events WHERE event_type = 'success'",
            conn,
        )
    except Exception:
        # Table might not exist
        df = pd.DataFrame()
    conn.close()
    return df


@st.cache_data
def load_worm_metrics():
    """Load per-worm metrics from database (for rate and success rate)."""
    if not DB_EXP.exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_EXP)
    try:
        df = pd.read_sql(
            "SELECT T_exp, length_mm, obs_time_min, n_success, n_attempt FROM exp_worm_metrics",
            conn,
        )
        # Compute derived metrics
        df["trans_rate"] = (df["n_success"] / df["obs_time_min"]) * 60  # events/hour
        df["success_rate"] = 100 * df["n_success"] / (df["n_success"] + df["n_attempt"])
        # Handle divisions by zero
        df["trans_rate"] = df["trans_rate"].fillna(0)
        df["success_rate"] = df["success_rate"].fillna(0)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


@st.cache_data
def get_common_sim_params():
    """
    Get parameter values that are common to all three chain lengths (N=40, 50, 60).
    Returns only (Pe, T, kappa) combinations available for ALL three N values.
    """
    if not DB_SIM.exists():
        return []

    conn = sqlite3.connect(DB_SIM)
    cursor = conn.cursor()

    # Find parameter sets common to N=40, 50, and 60
    cursor.execute(
        """
        SELECT Pe, T, kappa FROM trapping_events WHERE N = 40
        INTERSECT
        SELECT Pe, T, kappa FROM trapping_events WHERE N = 50
        INTERSECT
        SELECT Pe, T, kappa FROM trapping_events WHERE N = 60
        ORDER BY Pe, T, kappa
    """
    )
    common_params = cursor.fetchall()
    conn.close()

    return common_params


@st.cache_data
def get_best_params_for_temp(temp_exp, common_params):
    """
    Find the simulation parameters with mean τ_trap closest to experimental mean.
    Returns the index in common_params list.
    """
    if not DB_EXP.exists() or not DB_SIM.exists() or not common_params:
        return 0

    conn_exp = sqlite3.connect(DB_EXP)
    df_exp = pd.read_sql(
        f"SELECT time_min FROM exp_trapping_events WHERE T_exp = {temp_exp} AND time_min <= {MAX_TRAP_TIME}",
        conn_exp,
    )
    conn_exp.close()

    if len(df_exp) == 0:
        return 0

    mean_exp = df_exp["time_min"].mean()

    conn_sim = sqlite3.connect(DB_SIM)
    best_idx = 0
    best_diff = float("inf")

    for idx, (Pe, T, kappa) in enumerate(common_params):
        df_sim = pd.read_sql(
            f"""SELECT time_min FROM trapping_events
                WHERE N = 40 AND ABS(Pe - {Pe}) < 0.001
                AND ABS(T - {T}) < 0.001 AND ABS(kappa - {kappa}) < 0.001
                AND time_min <= {MAX_TRAP_TIME}""",
            conn_sim,
        )
        if len(df_sim) > 0:
            mean_sim = df_sim["time_min"].mean()
            diff = abs(mean_sim - mean_exp)
            if diff < best_diff:
                best_diff = diff
                best_idx = idx

    conn_sim.close()
    return best_idx


@st.cache_data
def load_sim_trapping_data(N, Pe, T, kappa):
    """Load simulation trapping times from database."""
    if not DB_SIM.exists():
        return np.array([])

    conn = sqlite3.connect(DB_SIM)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT time_min FROM trapping_events
        WHERE N = ? AND ABS(Pe - ?) < 0.001 AND ABS(T - ?) < 0.001 AND ABS(kappa - ?) < 0.001
    """,
        (N, Pe, T, kappa),
    )
    times = np.array([row[0] for row in cursor.fetchall()])
    conn.close()

    # Filter outliers
    times = times[times <= MAX_TRAP_TIME]
    return times


def group_worms_by_length(df, threshold=2.0, min_events=1):
    """
    Group trapping events by worm length (within threshold mm).
    Returns list of (mean_length, times_list) tuples.

    Args:
        df: DataFrame with 'length_mm' and 'time_min' columns
        threshold: Max length difference to group together (0 = no grouping)
        min_events: Minimum events per group to include in output
    """
    if len(df) == 0:
        return []

    # Sort by length
    df_sorted = df.sort_values("length_mm").reset_index(drop=True)

    grouped_data = []
    i = 0

    while i < len(df_sorted):
        current_length = df_sorted.iloc[i]["length_mm"]
        current_group_lengths = [current_length]
        current_group_times = [df_sorted.iloc[i]["time_min"]]

        j = i + 1
        while j < len(df_sorted):
            next_length = df_sorted.iloc[j]["length_mm"]
            if abs(next_length - current_length) <= threshold:
                current_group_lengths.append(next_length)
                current_group_times.append(df_sorted.iloc[j]["time_min"])
                j += 1
            else:
                break

        # Keep groups with >= min_events
        if len(current_group_times) >= min_events:
            mean_length = float(np.mean(current_group_lengths))
            grouped_data.append((mean_length, current_group_times))

        i = j

    return grouped_data


# =============================================================================
# RENDERING
# =============================================================================


def render_violin_comparison_tab():
    """Render violin plots tab with sub-tabs for all metrics."""
    st.header("Time distributions vs length")

    # Sub-tabs for all metrics
    tab_trap, tab_trans, tab_rate, tab_success = st.tabs(["τ_trap", "τ_trans", "Rate", "Success %"])

    with tab_trap:
        _render_trapping_subtab()

    with tab_trans:
        _render_translocation_subtab()

    with tab_rate:
        _render_rate_subtab()

    with tab_success:
        _render_success_rate_subtab()


def _render_trapping_subtab():
    """Render trapping time violin plots (experimental + simulation)."""
    st.subheader("Trapping time distributions")
    st.markdown(
        """
    Violin plots of experimental trapping times grouped by worm length,
    with optional simulation data overlay.
    """
    )

    # Check databases
    if not DB_EXP.exists():
        st.error(f"Experimental database not found: {DB_EXP}")
        st.info("Run `python collect_exp_trapping_data.py` to create the database.")
        return

    # Load experimental data
    exp_data = load_exp_trapping_data()
    if len(exp_data) == 0:
        st.error("No experimental data found.")
        return

    # Filter outliers
    exp_data = exp_data[exp_data["time_min"] <= MAX_TRAP_TIME]

    # Simulation controls - one selector per temperature
    selected_params_dict = {temp: None for temp in TEMPERATURES_EXP}

    if DB_SIM.exists():
        common_params = get_common_sim_params()

        if common_params:
            st.subheader("Simulation parameters (one per temperature)")

            # Format options as "Pe=X, T=Y, κ=Z"
            param_options = [
                f"Pe={p[0]:.2f}, T={p[1]:.2f}, κ={p[2]:.2f}" for p in common_params
            ]

            # Get best default index for each temperature
            default_indices = {
                temp: get_best_params_for_temp(temp, tuple(common_params))
                for temp in TEMPERATURES_EXP
            }

            cols = st.columns(3)
            for i, temp in enumerate(TEMPERATURES_EXP):
                with cols[i]:
                    show = st.checkbox(
                        f"T={temp}°C", value=True, key=f"violin_show_{temp}"
                    )
                    if show:
                        idx = st.selectbox(
                            "Parameters",
                            options=range(len(param_options)),
                            index=default_indices[temp],
                            format_func=lambda j, opts=param_options: opts[j],
                            key=f"violin_params_{temp}",
                        )
                        selected_params_dict[temp] = common_params[idx]

            st.caption(
                f"{len(common_params)} parameter sets available (common to N=40, 50, 60)"
            )
        else:
            st.warning("No common parameter sets found for N=40, 50, 60.")

    # Plot mode selection
    st.subheader("Plot options")
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        use_matplotlib = st.checkbox(
            "Use matplotlib (static)", value=False, key="violin_matplotlib"
        )
    with col_opt2:
        use_length_tolerance = st.checkbox(
            "Group similar lengths (±2mm)",
            value=True,
            key="violin_length_tolerance",
            help="When enabled, worms within ±2mm are grouped. When disabled, each unique length is separate."
        )

    # Mapping N to approximate length (mm)
    n_to_length = {40: 20, 50: 25, 60: 30}
    length_threshold = LENGTH_THRESHOLD if use_length_tolerance else 0.0

    if use_matplotlib:
        _render_matplotlib_figure(exp_data, selected_params_dict, n_to_length, length_threshold)
    else:
        _render_plotly_figure(exp_data, selected_params_dict, n_to_length, length_threshold)

    # Statistics summary
    _render_statistics(exp_data, selected_params_dict)

    # Export section
    _render_export_section(exp_data, selected_params_dict, length_threshold)


def _render_plotly_figure(exp_data, selected_params_dict, n_to_length, length_threshold=LENGTH_THRESHOLD):
    """Render interactive Plotly violin plots."""
    min_events = 3 if length_threshold > 0 else 1  # Fewer required when no grouping
    # Create Plotly figure with 3 stacked subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=["", "", ""],
    )

    for idx, temp in enumerate(TEMPERATURES_EXP):
        row = idx + 1

        # Filter experimental data for this temperature
        df_temp = exp_data[exp_data["T_exp"] == temp]

        # Group by length
        grouped = group_worms_by_length(df_temp, length_threshold, min_events)

        # Plot experimental violins
        for g_idx, (mean_length, times_list) in enumerate(grouped):

            fig.add_trace(
                go.Violin(
                    y=times_list,
                    x0=mean_length,
                    name=f"Exp T={temp}°C" if g_idx == 0 and idx == 0 else None,
                    legendgroup=f"exp_{temp}",
                    showlegend=(g_idx == 0 and idx == 0),
                    line_color=TEMP_COLORS_PLOTLY[temp].replace("0.6)", "1)"),
                    fillcolor=TEMP_COLORS_PLOTLY[temp],
                    opacity=0.8,
                    meanline_visible=True,
                    width=2.0,
                    side="both",
                    scalemode="width",
                    points=False,
                ),
                row=row,
                col=1,
            )

        # Overlay simulation violins if enabled for this temperature
        selected_params = selected_params_dict.get(temp)
        if selected_params is not None:
            Pe, T_sim, kappa = selected_params

            for N, x_pos in n_to_length.items():
                sim_times = load_sim_trapping_data(N, Pe, T_sim, kappa)

                if len(sim_times) >= 3:
                    # Offset simulation violin slightly to the right
                    fig.add_trace(
                        go.Violin(
                            y=sim_times.tolist(),
                            x0=x_pos + 2.5,
                            name=f"Sim N={N}" if idx == 0 else None,
                            legendgroup=f"sim_{N}",
                            showlegend=(idx == 0),
                            line_color="rgba(255, 140, 0, 1)",
                            fillcolor=SIM_COLOR_PLOTLY,
                            opacity=0.8,
                            meanline_visible=True,
                            width=2.0,
                            side="both",
                            scalemode="width",
                            points=False,
                        ),
                        row=row,
                        col=1,
                    )

        # Build annotation text
        annotation_text = f"T = {temp}°C"
        if selected_params is not None:
            Pe, T_sim, kappa = selected_params
            annotation_text += f"<br>Pe={Pe:.2f}, T={T_sim:.2f}, κ={kappa:.2f}"

        # Add temperature annotation
        fig.add_annotation(
            x=38,
            y=9,
            text=annotation_text,
            showarrow=False,
            font=dict(size=14),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            row=row,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=600,
        template="plotly_white",
        font=dict(size=16),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        violinmode="overlay",
        violingap=0,
        violingroupgap=0,
    )

    # Update axes
    for row in range(1, 4):
        fig.update_yaxes(
            title_text="τ_trap (min)" if row == 2 else "",
            range=[0, MAX_TRAP_TIME],
            row=row,
            col=1,
        )
        fig.update_xaxes(range=[12, 40], row=row, col=1)

    fig.update_xaxes(title_text="l_c (mm)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _render_matplotlib_figure(exp_data, selected_params_dict, n_to_length, length_threshold=LENGTH_THRESHOLD):
    """Render static matplotlib violin plots."""
    min_events = 3 if length_threshold > 0 else 1
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.subplots_adjust(hspace=0)

    # Check if any simulation is enabled
    any_sim_enabled = any(p is not None for p in selected_params_dict.values())

    for idx, temp in enumerate(TEMPERATURES_EXP):
        ax = axes[idx]

        # Filter experimental data for this temperature
        df_temp = exp_data[exp_data["T_exp"] == temp]

        # Group by length
        grouped = group_worms_by_length(df_temp, length_threshold, min_events)

        if len(grouped) > 0:
            positions_exp = [g[0] for g in grouped]
            data_exp = [g[1] for g in grouped]

            # Plot experimental violins
            parts = ax.violinplot(
                data_exp,
                positions=positions_exp,
                widths=1.5,
                showmeans=True,
                showmedians=True,
                showextrema=False,
            )

            # Style experimental violins
            for pc in parts["bodies"]:
                pc.set_facecolor(TEMP_COLORS_MPL[temp])
                pc.set_alpha(0.5)
                pc.set_edgecolor("black")
                pc.set_linewidth(1)

            parts["cmeans"].set_edgecolor("darkred")
            parts["cmeans"].set_linewidth(2)
            parts["cmedians"].set_edgecolor("darkblue")
            parts["cmedians"].set_linewidth(2)

        # Overlay simulation violins if enabled for this temperature
        selected_params = selected_params_dict.get(temp)
        if selected_params is not None:
            Pe, T_sim, kappa = selected_params
            sim_data = []
            sim_positions = []

            for N, x_pos in n_to_length.items():
                sim_times = load_sim_trapping_data(N, Pe, T_sim, kappa)
                if len(sim_times) >= 3:
                    sim_positions.append(x_pos + 2.5)
                    sim_data.append(sim_times.tolist())

            if len(sim_data) > 0:
                parts_sim = ax.violinplot(
                    sim_data,
                    positions=sim_positions,
                    widths=1.5,
                    showmeans=True,
                    showmedians=True,
                    showextrema=False,
                )

                for pc in parts_sim["bodies"]:
                    pc.set_facecolor("orange")
                    pc.set_alpha(0.6)
                    pc.set_edgecolor("darkorange")
                    pc.set_linewidth(1)

                parts_sim["cmeans"].set_edgecolor("darkred")
                parts_sim["cmeans"].set_linewidth(2)
                parts_sim["cmedians"].set_edgecolor("darkblue")
                parts_sim["cmedians"].set_linewidth(2)

        # Only bottom panel gets x-label
        if idx == 2:
            ax.set_xlabel(r"$\ell_c$ (mm)", fontsize=16)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        ax.set_ylabel(r"$\tau_{\rm trap}$ (min)", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(-1, None)

        # Legend only on top panel
        if idx == 0:
            legend_elements = [
                Line2D([0], [0], color="darkred", linewidth=2, label="Mean"),
                Line2D([0], [0], color="darkblue", linewidth=2, label="Median"),
            ]
            if any_sim_enabled:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        marker="s",
                        color="w",
                        markerfacecolor="orange",
                        markersize=10,
                        label="Simulation",
                    )
                )
            ax.legend(handles=legend_elements, loc="upper left", fontsize=12)

        # Build annotation text with parameters
        annotation_text = f"$T={temp}^\\circ$C"
        if selected_params is not None:
            Pe, T_sim, kappa = selected_params
            annotation_text += f"\nPe={Pe:.2f}, T={T_sim:.2f}, $\\kappa$={kappa:.2f}"

        # Temperature annotation box
        ax.text(
            0.95,
            0.95,
            annotation_text,
            transform=ax.transAxes,
            fontsize=12,
            va="top",
            ha="right",
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="black", linewidth=1.5
            ),
        )

    axes[2].set_xlim(12, 40)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_statistics(exp_data, selected_params_dict):
    """Render statistics summary table."""
    with st.expander("Statistics"):
        stats_data = []

        for temp in TEMPERATURES_EXP:
            df_temp = exp_data[exp_data["T_exp"] == temp]
            if len(df_temp) > 0:
                times = np.array(df_temp["time_min"].tolist())
                stats_data.append(
                    {
                        "Source": f"Exp T={temp}C",
                        "N events": len(df_temp),
                        "Mean (min)": f"{float(np.mean(times)):.2f}",
                        "Median (min)": f"{float(np.median(times)):.2f}",
                        "Std (min)": f"{float(np.std(times)):.2f}",
                    }
                )

            # Add simulation stats for this temperature if enabled
            selected_params = selected_params_dict.get(temp)
            if selected_params is not None:
                Pe, T_sim, kappa = selected_params
                all_sim_times = []
                for N in [40, 50, 60]:
                    sim_times = load_sim_trapping_data(N, Pe, T_sim, kappa)
                    all_sim_times.extend(sim_times)
                    if len(sim_times) > 0:
                        stats_data.append(
                            {
                                "Source": f"Sim T={temp}C N={N} (Pe={Pe:.2f})",
                                "N events": len(sim_times),
                                "Mean (min)": f"{float(np.mean(sim_times)):.2f}",
                                "Median (min)": f"{float(np.median(sim_times)):.2f}",
                                "Std (min)": f"{float(np.std(sim_times)):.2f}",
                            }
                        )
                # Add combined stats (mean over all N) - this is what's used to find best params
                if len(all_sim_times) > 0:
                    stats_data.append(
                        {
                            "Source": f"**Sim T={temp}C (all N)** Pe={Pe:.2f}",
                            "N events": len(all_sim_times),
                            "Mean (min)": f"{float(np.mean(all_sim_times)):.2f}",
                            "Median (min)": f"{float(np.median(all_sim_times)):.2f}",
                            "Std (min)": f"{float(np.std(all_sim_times)):.2f}",
                        }
                    )

        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)


def _render_export_section(exp_data, selected_params_dict, length_threshold):
    """Render export expander for violin plot with matplotlib figure download."""
    from utils.export_matplotlib import (
        setup_latex_style,
        create_violin_figure_mpl,
        create_violin_export_zip,
    )

    with st.expander("📥 Export matplotlib figure"):
        col1, col2, col3 = st.columns(3)

        with col1:
            layout_choice = st.radio(
                "Layout",
                ["Three panels (stacked)", "Single temperature"],
                key="violin_export_layout"
            )

        with col2:
            if layout_choice == "Single temperature":
                selected_temp = st.selectbox(
                    "Temperature",
                    TEMPERATURES_EXP,
                    format_func=lambda t: f"{t}°C",
                    key="violin_export_temp"
                )
            else:
                selected_temp = None

        with col3:
            fig_format = st.selectbox(
                "Format",
                ["pdf", "png", "svg"],
                key="violin_export_format"
            )
            use_latex = st.checkbox("Use LaTeX", value=True, key="violin_export_latex")

        include_simulation = st.checkbox(
            "Include simulation data",
            value=any(p is not None for p in selected_params_dict.values()),
            key="violin_export_include_sim",
            help="Include simulation violin overlays in the export"
        )

        if st.button("Generate export", key="violin_export_btn"):
            with st.spinner("Generating figure..."):
                try:
                    # Setup LaTeX style
                    setup_latex_style(use_latex=use_latex)

                    # Prepare grouped data for each temperature
                    min_events = 3 if length_threshold > 0 else 1
                    grouped_data_by_temp = {}
                    exp_data_dict = {}

                    for temp in TEMPERATURES_EXP:
                        df_temp = exp_data[exp_data["T_exp"] == temp]
                        grouped = group_worms_by_length(df_temp, length_threshold, min_events)
                        grouped_data_by_temp[temp] = grouped

                        # Store raw data for export
                        exp_data_dict[temp] = {
                            'times': np.array(df_temp["time_min"].tolist()),
                            'lengths': np.array(df_temp["length_mm"].tolist()),
                        }

                    # Prepare simulation data if enabled
                    sim_data = None
                    sim_params = None

                    if include_simulation:
                        sim_data = {}
                        sim_params = {}
                        n_to_length = {40: 20, 50: 25, 60: 30}

                        for temp, params in selected_params_dict.items():
                            if params is not None:
                                Pe, T_sim, kappa = params
                                sim_params[temp] = params
                                sim_data[temp] = {}

                                for N in n_to_length.keys():
                                    sim_times = load_sim_trapping_data(N, Pe, T_sim, kappa)
                                    if len(sim_times) >= 3:
                                        sim_data[temp][N] = sim_times

                    # Determine layout
                    layout = "single" if layout_choice == "Single temperature" else "stacked"

                    # Create figure
                    fig, _ = create_violin_figure_mpl(
                        grouped_data_by_temp,
                        layout=layout,
                        selected_temp=selected_temp,
                        sim_data=sim_data if include_simulation else None,
                        sim_params=sim_params if include_simulation else None,
                        length_threshold=length_threshold,
                    )

                    # Show preview
                    st.pyplot(fig)

                    # Prepare config for export
                    if layout == "single" and selected_temp is not None:
                        temps_for_export = [selected_temp]
                    else:
                        temps_for_export = TEMPERATURES_EXP

                    config = {
                        'layout': layout,
                        'temperatures': temps_for_export,
                        'length_threshold': length_threshold,
                        'use_latex': use_latex,
                    }

                    # Filter exp_data_dict to only include exported temperatures
                    exp_data_export = {t: exp_data_dict[t] for t in temps_for_export}

                    # Create ZIP
                    zip_bytes = create_violin_export_zip(
                        fig,
                        exp_data_export,
                        config,
                        sim_data_dict=sim_data if include_simulation else None,
                        sim_params=sim_params if include_simulation else None,
                        fig_format=fig_format,
                    )

                    # Provide download button
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📦 Download ZIP archive",
                        data=zip_bytes,
                        file_name=f"violin_trapping_export_{timestamp}.zip",
                        mime="application/zip",
                        key="violin_download_btn",
                    )

                    plt.close(fig)

                except Exception as e:
                    st.error(f"Error generating figure: {e}")
                    import traceback
                    st.code(traceback.format_exc())


# =============================================================================
# TRANSLOCATION SUBTAB
# =============================================================================


def _render_translocation_subtab():
    """Render translocation time violin plots (experimental only)."""
    st.subheader("Translocation time distributions")
    st.markdown(
        """
    Violin plots of successful translocation times grouped by worm length.
    Only successful translocations are shown.
    """
    )

    # Check database
    if not DB_EXP.exists():
        st.error(f"Experimental database not found: {DB_EXP}")
        st.info("Run `python collect_exp_translocation_data.py` to create the database.")
        return

    # Load translocation data
    trans_data = load_exp_translocation_data()
    if len(trans_data) == 0:
        st.warning("No translocation data found.")
        st.info("Run `python collect_exp_translocation_data.py` to populate the database.")
        return

    # Filter outliers
    trans_data = trans_data[trans_data["time_min"] <= MAX_TRANS_TIME]

    # Plot mode selection
    use_matplotlib = st.checkbox(
        "Use matplotlib (static)", value=False, key="trans_violin_matplotlib"
    )

    if use_matplotlib:
        _render_translocation_matplotlib(trans_data)
    else:
        _render_translocation_plotly(trans_data)

    # Statistics
    _render_translocation_statistics(trans_data)


def _render_translocation_plotly(trans_data):
    """Render interactive Plotly violin plots for translocation."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=["", "", ""],
    )

    for idx, temp in enumerate(TEMPERATURES_EXP):
        row = idx + 1

        # Filter data for this temperature
        df_temp = trans_data[trans_data["T_exp"] == temp]

        # Group by length
        grouped = group_worms_by_length(df_temp, LENGTH_THRESHOLD)

        # Plot violins
        for g_idx, (mean_length, times_list) in enumerate(grouped):
            if len(times_list) < 3:
                continue

            fig.add_trace(
                go.Violin(
                    y=times_list,
                    x0=mean_length,
                    name=f"Exp T={temp}°C" if g_idx == 0 and idx == 0 else None,
                    legendgroup=f"exp_{temp}",
                    showlegend=(g_idx == 0 and idx == 0),
                    line_color=TEMP_COLORS_PLOTLY[temp].replace("0.5)", "1)"),
                    fillcolor=TEMP_COLORS_PLOTLY[temp],
                    opacity=0.8,
                    meanline_visible=True,
                    width=2.0,
                    side="both",
                    scalemode="width",
                    points=False,
                ),
                row=row,
                col=1,
            )

        # Add temperature annotation
        fig.add_annotation(
            x=38,
            y=MAX_TRANS_TIME * 0.9,
            text=f"T = {temp}°C",
            showarrow=False,
            font=dict(size=14),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            row=row,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=600,
        template="plotly_white",
        font=dict(size=16),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        violinmode="overlay",
        violingap=0,
        violingroupgap=0,
    )

    # Update axes
    for row in range(1, 4):
        fig.update_yaxes(
            title_text="τ_trans (min)" if row == 2 else "",
            range=[0, MAX_TRANS_TIME],
            row=row,
            col=1,
        )
        fig.update_xaxes(range=[12, 40], row=row, col=1)

    fig.update_xaxes(title_text="l_c (mm)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _render_translocation_matplotlib(trans_data):
    """Render static matplotlib violin plots for translocation."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.subplots_adjust(hspace=0)

    for idx, temp in enumerate(TEMPERATURES_EXP):
        ax = axes[idx]

        # Filter data for this temperature
        df_temp = trans_data[trans_data["T_exp"] == temp]

        # Group by length
        grouped = group_worms_by_length(df_temp, LENGTH_THRESHOLD)

        if len(grouped) > 0:
            positions_exp = [g[0] for g in grouped]
            data_exp = [g[1] for g in grouped]

            # Plot violins
            parts = ax.violinplot(
                data_exp,
                positions=positions_exp,
                widths=1.5,
                showmeans=True,
                showmedians=True,
                showextrema=False,
            )

            # Style violins
            for pc in parts["bodies"]:
                pc.set_facecolor(TEMP_COLORS_MPL[temp])
                pc.set_alpha(0.5)
                pc.set_edgecolor("black")
                pc.set_linewidth(1)

            parts["cmeans"].set_edgecolor("darkred")
            parts["cmeans"].set_linewidth(2)
            parts["cmedians"].set_edgecolor("darkblue")
            parts["cmedians"].set_linewidth(2)

        # Labels
        if idx == 2:
            ax.set_xlabel(r"$\ell_c$ (mm)", fontsize=16)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        ax.set_ylabel(r"$\tau_{\rm trans}$ (min)", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(-1, None)

        # Legend on top panel
        if idx == 0:
            legend_elements = [
                Line2D([0], [0], color="darkred", linewidth=2, label="Mean"),
                Line2D([0], [0], color="darkblue", linewidth=2, label="Median"),
            ]
            ax.legend(handles=legend_elements, loc="upper left", fontsize=12)

        # Temperature annotation
        ax.text(
            0.95,
            0.95,
            f"$T={temp}^\\circ$C",
            transform=ax.transAxes,
            fontsize=12,
            va="top",
            ha="right",
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="black", linewidth=1.5
            ),
        )

    axes[2].set_xlim(12, 40)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_translocation_statistics(trans_data):
    """Render translocation statistics summary table."""
    with st.expander("Statistics"):
        stats_data = []

        for temp in TEMPERATURES_EXP:
            df_temp = trans_data[trans_data["T_exp"] == temp]
            if len(df_temp) > 0:
                times = np.array(df_temp["time_min"].tolist())
                stats_data.append(
                    {
                        "Source": f"Exp T={temp}C",
                        "N events": len(df_temp),
                        "Mean (min)": f"{float(np.mean(times)):.2f}",
                        "Median (min)": f"{float(np.median(times)):.2f}",
                        "Std (min)": f"{float(np.std(times)):.2f}",
                    }
                )

        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)


# =============================================================================
# RATE SUBTAB
# =============================================================================


def _render_rate_subtab():
    """Render translocation rate violin plots (per-worm)."""
    st.subheader("Translocation rate distributions")
    st.markdown(
        """
    Violin plots of translocation rates (events/hour) per worm, grouped by worm length.
    Rate = (successful translocations / observation time) × 60
    """
    )

    # Load worm metrics
    metrics_data = load_worm_metrics()
    if len(metrics_data) == 0:
        st.warning("No worm metrics found.")
        st.info("Run `python collect_exp_translocation_data.py` to populate the database.")
        return

    # Plot mode selection
    use_matplotlib = st.checkbox(
        "Use matplotlib (static)", value=False, key="rate_violin_matplotlib"
    )

    if use_matplotlib:
        _render_rate_matplotlib(metrics_data)
    else:
        _render_rate_plotly(metrics_data)

    # Statistics
    _render_rate_statistics(metrics_data)


def _render_rate_plotly(metrics_data):
    """Render interactive Plotly violin plots for translocation rate."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=["", "", ""],
    )

    for idx, temp in enumerate(TEMPERATURES_EXP):
        row = idx + 1

        # Filter data for this temperature
        df_temp = metrics_data[metrics_data["T_exp"] == temp]

        # Group by length (using trans_rate instead of time_min)
        grouped = _group_by_length_generic(df_temp, "trans_rate", LENGTH_THRESHOLD)

        # Plot violins
        for g_idx, (mean_length, values_list) in enumerate(grouped):
            if len(values_list) < 2:
                continue

            fig.add_trace(
                go.Violin(
                    y=values_list,
                    x0=mean_length,
                    name=f"T={temp}°C" if g_idx == 0 and idx == 0 else None,
                    legendgroup=f"exp_{temp}",
                    showlegend=(g_idx == 0 and idx == 0),
                    line_color=TEMP_COLORS_PLOTLY[temp].replace("0.5)", "1)"),
                    fillcolor=TEMP_COLORS_PLOTLY[temp],
                    opacity=0.8,
                    meanline_visible=True,
                    width=2.0,
                    side="both",
                    scalemode="width",
                    points=False,
                ),
                row=row,
                col=1,
            )

        # Add temperature annotation
        fig.add_annotation(
            x=38,
            y=12,
            text=f"T = {temp}°C",
            showarrow=False,
            font=dict(size=14),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            row=row,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=600,
        template="plotly_white",
        font=dict(size=16),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        violinmode="overlay",
        violingap=0,
        violingroupgap=0,
    )

    # Update axes
    for row in range(1, 4):
        fig.update_yaxes(
            title_text="Rate (ev/h)" if row == 2 else "",
            range=[0, None],
            row=row,
            col=1,
        )
        fig.update_xaxes(range=[12, 40], row=row, col=1)

    fig.update_xaxes(title_text="l_c (mm)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _render_rate_matplotlib(metrics_data):
    """Render static matplotlib violin plots for translocation rate."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.subplots_adjust(hspace=0)

    for idx, temp in enumerate(TEMPERATURES_EXP):
        ax = axes[idx]

        # Filter data for this temperature
        df_temp = metrics_data[metrics_data["T_exp"] == temp]

        # Group by length
        grouped = _group_by_length_generic(df_temp, "trans_rate", LENGTH_THRESHOLD)

        if len(grouped) > 0:
            positions = [g[0] for g in grouped]
            data_list = [g[1] for g in grouped]

            # Filter groups with at least 2 values
            valid = [(p, d) for p, d in zip(positions, data_list) if len(d) >= 2]
            if valid:
                positions = [v[0] for v in valid]
                data_list = [v[1] for v in valid]

                parts = ax.violinplot(
                    data_list,
                    positions=positions,
                    widths=1.5,
                    showmeans=True,
                    showmedians=True,
                    showextrema=False,
                )

                for pc in parts["bodies"]:
                    pc.set_facecolor(TEMP_COLORS_MPL[temp])
                    pc.set_alpha(0.5)
                    pc.set_edgecolor("black")
                    pc.set_linewidth(1)

                parts["cmeans"].set_edgecolor("darkred")
                parts["cmeans"].set_linewidth(2)
                parts["cmedians"].set_edgecolor("darkblue")
                parts["cmedians"].set_linewidth(2)

        if idx == 2:
            ax.set_xlabel(r"$\ell_c$ (mm)", fontsize=16)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        ax.set_ylabel("Rate (ev/h)", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(-1, None)

        if idx == 0:
            legend_elements = [
                Line2D([0], [0], color="darkred", linewidth=2, label="Mean"),
                Line2D([0], [0], color="darkblue", linewidth=2, label="Median"),
            ]
            ax.legend(handles=legend_elements, loc="upper left", fontsize=12)

        ax.text(
            0.95,
            0.95,
            f"$T={temp}^\\circ$C",
            transform=ax.transAxes,
            fontsize=12,
            va="top",
            ha="right",
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="black", linewidth=1.5
            ),
        )

    axes[2].set_xlim(12, 40)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_rate_statistics(metrics_data):
    """Render rate statistics summary table."""
    with st.expander("Statistics"):
        stats_data = []

        for temp in TEMPERATURES_EXP:
            df_temp = metrics_data[metrics_data["T_exp"] == temp]
            if len(df_temp) > 0:
                rates = np.array(df_temp["trans_rate"].tolist())
                stats_data.append(
                    {
                        "Source": f"T={temp}C",
                        "N worms": len(df_temp),
                        "Mean (ev/h)": f"{float(np.mean(rates)):.2f}",
                        "Median (ev/h)": f"{float(np.median(rates)):.2f}",
                        "Std (ev/h)": f"{float(np.std(rates)):.2f}",
                    }
                )

        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)


# =============================================================================
# SUCCESS RATE SUBTAB
# =============================================================================


def _render_success_rate_subtab():
    """Render success rate violin plots (per-worm)."""
    st.subheader("Success rate distributions")
    st.markdown(
        """
    Violin plots of translocation success rates (%) per worm, grouped by worm length.
    Success rate = 100 × (successful / total events)
    """
    )

    # Load worm metrics
    metrics_data = load_worm_metrics()
    if len(metrics_data) == 0:
        st.warning("No worm metrics found.")
        st.info("Run `python collect_exp_translocation_data.py` to populate the database.")
        return

    # Plot mode selection
    use_matplotlib = st.checkbox(
        "Use matplotlib (static)", value=False, key="success_violin_matplotlib"
    )

    if use_matplotlib:
        _render_success_matplotlib(metrics_data)
    else:
        _render_success_plotly(metrics_data)

    # Statistics
    _render_success_statistics(metrics_data)


def _render_success_plotly(metrics_data):
    """Render interactive Plotly violin plots for success rate."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=["", "", ""],
    )

    for idx, temp in enumerate(TEMPERATURES_EXP):
        row = idx + 1

        # Filter data for this temperature
        df_temp = metrics_data[metrics_data["T_exp"] == temp]

        # Group by length
        grouped = _group_by_length_generic(df_temp, "success_rate", LENGTH_THRESHOLD)

        # Plot violins
        for g_idx, (mean_length, values_list) in enumerate(grouped):
            if len(values_list) < 2:
                continue

            fig.add_trace(
                go.Violin(
                    y=values_list,
                    x0=mean_length,
                    name=f"T={temp}°C" if g_idx == 0 and idx == 0 else None,
                    legendgroup=f"exp_{temp}",
                    showlegend=(g_idx == 0 and idx == 0),
                    line_color=TEMP_COLORS_PLOTLY[temp].replace("0.5)", "1)"),
                    fillcolor=TEMP_COLORS_PLOTLY[temp],
                    opacity=0.8,
                    meanline_visible=True,
                    width=2.0,
                    side="both",
                    scalemode="width",
                    points=False,
                ),
                row=row,
                col=1,
            )

        # Add temperature annotation
        fig.add_annotation(
            x=38,
            y=90,
            text=f"T = {temp}°C",
            showarrow=False,
            font=dict(size=14),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            row=row,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=600,
        template="plotly_white",
        font=dict(size=16),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        violinmode="overlay",
        violingap=0,
        violingroupgap=0,
    )

    # Update axes
    for row in range(1, 4):
        fig.update_yaxes(
            title_text="Success (%)" if row == 2 else "",
            range=[0, 100],
            row=row,
            col=1,
        )
        fig.update_xaxes(range=[12, 40], row=row, col=1)

    fig.update_xaxes(title_text="l_c (mm)", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _render_success_matplotlib(metrics_data):
    """Render static matplotlib violin plots for success rate."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.subplots_adjust(hspace=0)

    for idx, temp in enumerate(TEMPERATURES_EXP):
        ax = axes[idx]

        # Filter data for this temperature
        df_temp = metrics_data[metrics_data["T_exp"] == temp]

        # Group by length
        grouped = _group_by_length_generic(df_temp, "success_rate", LENGTH_THRESHOLD)

        if len(grouped) > 0:
            positions = [g[0] for g in grouped]
            data_list = [g[1] for g in grouped]

            # Filter groups with at least 2 values
            valid = [(p, d) for p, d in zip(positions, data_list) if len(d) >= 2]
            if valid:
                positions = [v[0] for v in valid]
                data_list = [v[1] for v in valid]

                parts = ax.violinplot(
                    data_list,
                    positions=positions,
                    widths=1.5,
                    showmeans=True,
                    showmedians=True,
                    showextrema=False,
                )

                for pc in parts["bodies"]:
                    pc.set_facecolor(TEMP_COLORS_MPL[temp])
                    pc.set_alpha(0.5)
                    pc.set_edgecolor("black")
                    pc.set_linewidth(1)

                parts["cmeans"].set_edgecolor("darkred")
                parts["cmeans"].set_linewidth(2)
                parts["cmedians"].set_edgecolor("darkblue")
                parts["cmedians"].set_linewidth(2)

        if idx == 2:
            ax.set_xlabel(r"$\ell_c$ (mm)", fontsize=16)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        ax.set_ylabel("Success (%)", fontsize=16)
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(-1, 100)

        if idx == 0:
            legend_elements = [
                Line2D([0], [0], color="darkred", linewidth=2, label="Mean"),
                Line2D([0], [0], color="darkblue", linewidth=2, label="Median"),
            ]
            ax.legend(handles=legend_elements, loc="upper left", fontsize=12)

        ax.text(
            0.95,
            0.95,
            f"$T={temp}^\\circ$C",
            transform=ax.transAxes,
            fontsize=12,
            va="top",
            ha="right",
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="black", linewidth=1.5
            ),
        )

    axes[2].set_xlim(12, 40)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_success_statistics(metrics_data):
    """Render success rate statistics summary table."""
    with st.expander("Statistics"):
        stats_data = []

        for temp in TEMPERATURES_EXP:
            df_temp = metrics_data[metrics_data["T_exp"] == temp]
            if len(df_temp) > 0:
                rates = np.array(df_temp["success_rate"].tolist())
                stats_data.append(
                    {
                        "Source": f"T={temp}C",
                        "N worms": len(df_temp),
                        "Mean (%)": f"{float(np.mean(rates)):.2f}",
                        "Median (%)": f"{float(np.median(rates)):.2f}",
                        "Std (%)": f"{float(np.std(rates)):.2f}",
                    }
                )

        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _group_by_length_generic(df, value_column, threshold=2.0):
    """
    Group worms by length and return list of (mean_length, values_list) tuples.
    Generic version that works for any value column.
    """
    if len(df) == 0:
        return []

    # Sort by length
    df_sorted = df.sort_values("length_mm").reset_index(drop=True)

    grouped_data = []
    i = 0

    while i < len(df_sorted):
        current_length = df_sorted.iloc[i]["length_mm"]
        current_group_lengths = [current_length]
        current_group_values = [df_sorted.iloc[i][value_column]]

        j = i + 1
        while j < len(df_sorted):
            next_length = df_sorted.iloc[j]["length_mm"]
            if abs(next_length - current_length) <= threshold:
                current_group_lengths.append(next_length)
                current_group_values.append(df_sorted.iloc[j][value_column])
                j += 1
            else:
                break

        mean_length = float(np.mean(current_group_lengths))
        grouped_data.append((mean_length, current_group_values))

        i = j

    return grouped_data
