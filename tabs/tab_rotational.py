"""
Tab: Rotational Number
Visualize N_rot and various τ timescales as functions of simulation parameters.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.constants import LATEX_LABELS, EXP_TEMP_COLORS

# Plain text labels for dropdowns (Streamlit doesn't render HTML in selectbox)
# Simplified: N_rot, τ_trans (= min(τ_15, τ_0)), R²_trans, and τ_rot
DROPDOWN_LABELS = {
    "N_rot": "N_rot",
    "tau_trans": "τ_trans (min)",
    "R2_trans": "R² (visited surface)",
    "tau_rot": "τ_rot (min)",
}


@st.cache_data
def load_rotational_data():
    """Load confinement data with rotational columns.

    Now uses simplified format: single N_rot with τ_trans = min(τ_15, τ_0)
    """
    csv_path = Path(__file__).parent.parent / "data_confinement.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, comment="#")
    df["Pe"] = df["Pe"].round(2)
    df["T"] = df["T"].round(2)
    df["kappa"] = df["kappa"].round(2)

    # Keep only rows with rotational data
    # Simplified: N_rot, τ_trans (= min(τ_15, τ_0)), R²_trans, τ_rot
    rot_cols = ["N_rot", "tau_trans", "R2_trans", "tau_rot"]
    df_rot = df.dropna(subset=rot_cols, how="all")

    return df_rot


def render_rotational_tab(df_exp=None):
    """Render the Rotational Number tab."""
    st.header("Rotational Number Analysis")
    st.markdown("""
    Rotational number **N_rot = τ_rot / τ_trans** characterizes the ratio
    of rotational to translational time scales.

    - **N_rot > 1**: Slow rotation (τ_rot > τ_trans) — polymer explores cavity before completing a full rotation
    - **N_rot < 1**: Fast rotation (τ_rot < τ_trans) — polymer makes multiple rotations while exploring

    **Robust saturation time estimation:**
    - **τ_trans = min(τ_15, τ_0)**: the actual saturation time used
    - τ_15: time for MSD to reach R² = 15 (surface = 15 mm²)
    - τ_0: time at first MSD maximum (dMSD/dt = 0)

    Using the minimum ensures we capture the actual confinement time,
    even when MSD saturates before reaching the R² = 15 threshold.
    """)

    # Load data
    df = load_rotational_data()

    if df is None or len(df) == 0:
        st.error("No rotational data available. Run `compile_all_data.py` first.")
        return

    # Check which columns have data
    # Simplified: N_rot, τ_trans (= min(τ_15, τ_0)), R²_trans, τ_rot
    rot_cols = ["N_rot", "tau_trans", "R2_trans", "tau_rot"]
    available_cols = [col for col in rot_cols if col in df.columns and df[col].notna().any()]

    if not available_cols:
        st.warning("No rotational data columns found in the dataset.")
        return

    st.success(f"Loaded {len(df)} data points with rotational data")

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        observable = st.selectbox(
            "Observable (Y-axis)",
            available_cols,
            index=0,
            format_func=lambda x: DROPDOWN_LABELS.get(x, x),
            key="rot_observable"
        )

    with col2:
        x_param = st.selectbox(
            "X-axis Parameter",
            ["Pe", "kappa"],
            index=0,
            key="rot_x_param"
        )

    with col3:
        group_by = st.selectbox(
            "Group lines by",
            ["kappa", "Pe"],
            index=0 if x_param == "Pe" else 1,
            key="rot_group_by"
        )

    # Display mode
    display_mode = st.radio(
        "Display mode",
        ["4 temperatures", "All temperatures", "Single temperature"],
        horizontal=True,
        key="rot_display_mode"
    )

    selected_temp = None
    if display_mode == "Single temperature":
        available_temps = sorted(df["T"].unique())
        default_idx = available_temps.index(0.1) if 0.1 in available_temps else 0
        selected_temp = st.selectbox(
            "Select temperature",
            available_temps,
            index=default_idx,
            format_func=lambda x: f"T = {x}",
            key="rot_temp"
        )

    # Plot options
    opt_col1, opt_col2 = st.columns(2)

    with opt_col1:
        use_log_y = st.checkbox("Log scale (Y-axis)", value=False, key="rot_log_y")

    with opt_col2:
        color_scheme_type = st.selectbox(
            "Color scheme",
            ["Discrete", "Gradient"],
            index=0,
            key="rot_color_scheme"
        )
        if color_scheme_type == "Gradient":
            color_palette = st.selectbox(
                "Color scale",
                ["Viridis", "Plasma", "Turbo", "Inferno"],
                index=0,
                key="rot_gradient_palette"
            )
        else:
            color_palette = "Alphabet"

    # Get unique group values
    group_values = sorted(df[group_by].dropna().unique())

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

    # Create plot based on display mode
    if display_mode == "4 temperatures":
        _render_4_temps(df, observable, x_param, group_by, group_values, color_map, use_log_y, df_exp)
    elif display_mode == "All temperatures":
        _render_all_temps(df, observable, x_param, group_by, group_values, color_map, use_log_y, df_exp)
    else:
        _render_single_temp(df, selected_temp, observable, x_param, group_by, group_values, color_map, use_log_y, df_exp)


def _render_4_temps(df, observable, x_param, group_by, group_values, color_map, use_log_y, df_exp=None):
    """Render 4-temperature mode with subplots."""
    temps = [0.05, 0.1, 0.2, 0.3]
    # Mapping from simulation T (LJ units) to experimental T_celsius
    temp_to_celsius = {0.05: 5, 0.1: 10, 0.2: 20, 0.3: 30}

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[f"T = {T}" for T in temps],
        horizontal_spacing=0.06
    )

    y_label = LATEX_LABELS.get(observable, observable)

    # Track x-axis range for experimental markers
    x_max_global = 0

    for col_idx, temp in enumerate(temps, 1):
        df_temp = df[abs(df["T"] - temp) < 0.01].copy()

        if len(df_temp) == 0:
            continue

        for group_val in group_values:
            df_group = df_temp[abs(df_temp[group_by] - group_val) < 0.001].copy()

            if len(df_group) == 0:
                continue

            # Filter out NaN values for the observable
            df_group = df_group.dropna(subset=[observable])

            if len(df_group) == 0:
                continue

            # Sort by x_param
            df_group = df_group.sort_values(x_param)

            # Track x-axis range
            if len(df_group) > 0:
                x_max_global = max(x_max_global, df_group[x_param].max())

            fig.add_trace(
                go.Scatter(
                    x=df_group[x_param],
                    y=df_group[observable],
                    mode="lines+markers",
                    name=f"{group_by}={group_val}",
                    line=dict(color=color_map[group_val], width=2),
                    marker=dict(size=8, color=color_map[group_val]),
                    connectgaps=True,
                    showlegend=(col_idx == 1),
                    legendgroup=str(group_val),
                    customdata=np.column_stack([
                        df_group["Pe"].values,
                        df_group["T"].values,
                        df_group["kappa"].values
                    ]),
                    hovertemplate='<b>Pe</b>: %{customdata[0]:.2f}<br>' +
                                  '<b>T</b>: %{customdata[1]:.2f}<br>' +
                                  '<b>κ</b>: %{customdata[2]:.2f}<br>' +
                                  f'<b>{x_param}</b>: %{{x:.2f}}<br>' +
                                  f'<b>{observable}</b>: %{{y:.3g}}<extra></extra>',
                ),
                row=1, col=col_idx
            )

    # Add experimental data as horizontal dashed lines with star markers
    if df_exp is not None and observable in df_exp.columns:
        exp_legend_added = set()
        for col_idx, temp in enumerate(temps, 1):
            exp_temp_celsius = temp_to_celsius.get(temp)
            if exp_temp_celsius is None:
                continue

            exp_data = df_exp[df_exp["T_celsius"] == exp_temp_celsius]
            if len(exp_data) == 0 or exp_data[observable].isna().all():
                continue

            exp_val = exp_data[observable].values[0]
            color = EXP_TEMP_COLORS.get(exp_temp_celsius, "black")

            # Add horizontal dashed line
            fig.add_hline(
                y=exp_val,
                line_dash="dash",
                line_color=color,
                line_width=2,
                row=1, col=col_idx
            )

            # Add star marker at the right edge
            show_legend = exp_temp_celsius not in exp_legend_added
            fig.add_trace(
                go.Scatter(
                    x=[x_max_global * 0.95],
                    y=[exp_val],
                    mode="markers",
                    marker=dict(symbol="star", size=16, color=color,
                               line=dict(color="black", width=1)),
                    name=f"Exp T={exp_temp_celsius}°C",
                    showlegend=show_legend,
                    legendgroup=f"exp_{exp_temp_celsius}",
                    hovertemplate=f'<b>Exp T={exp_temp_celsius}°C</b><br>' +
                                  f'<b>{observable}</b>: {exp_val:.3g}<extra></extra>',
                ),
                row=1, col=col_idx
            )
            exp_legend_added.add(exp_temp_celsius)

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
            fig.update_yaxes(title_text=y_label, row=1, col=col_idx)
        if use_log_y:
            fig.update_yaxes(type="log", row=1, col=col_idx)

    st.plotly_chart(fig, use_container_width=True)


def _render_all_temps(df, observable, x_param, group_by, group_values, color_map, use_log_y, df_exp=None):
    """Render all temperatures in a single plot with temperature as symbol."""
    fig = go.Figure()

    y_label = LATEX_LABELS.get(observable, observable)
    temps = sorted(df["T"].unique())

    # Symbols for different temperatures
    symbols = ["circle", "square", "diamond", "triangle-up", "star", "cross"]
    temp_symbols = {t: symbols[i % len(symbols)] for i, t in enumerate(temps)}

    # Track x-axis range for experimental markers
    x_max_global = 0

    for group_val in group_values:
        df_group = df[abs(df[group_by] - group_val) < 0.001].copy()

        if len(df_group) == 0:
            continue

        for temp in temps:
            df_temp = df_group[abs(df_group["T"] - temp) < 0.01].copy()
            df_temp = df_temp.dropna(subset=[observable])

            if len(df_temp) == 0:
                continue

            df_temp = df_temp.sort_values(x_param)

            # Track x-axis range
            if len(df_temp) > 0:
                x_max_global = max(x_max_global, df_temp[x_param].max())

            fig.add_trace(
                go.Scatter(
                    x=df_temp[x_param],
                    y=df_temp[observable],
                    mode="lines+markers",
                    name=f"{group_by}={group_val}, T={temp}",
                    line=dict(color=color_map[group_val], width=2),
                    marker=dict(
                        size=10,
                        color=color_map[group_val],
                        symbol=temp_symbols[temp]
                    ),
                    connectgaps=True,
                    customdata=np.column_stack([
                        df_temp["Pe"].values,
                        df_temp["T"].values,
                        df_temp["kappa"].values
                    ]),
                    hovertemplate='<b>Pe</b>: %{customdata[0]:.2f}<br>' +
                                  '<b>T</b>: %{customdata[1]:.2f}<br>' +
                                  '<b>κ</b>: %{customdata[2]:.2f}<br>' +
                                  f'<b>{x_param}</b>: %{{x:.2f}}<br>' +
                                  f'<b>{observable}</b>: %{{y:.3g}}<extra></extra>',
                )
            )

    # Add experimental data as horizontal dashed lines with star markers
    if df_exp is not None and observable in df_exp.columns:
        for temp_celsius in df_exp["T_celsius"].unique():
            exp_data = df_exp[df_exp["T_celsius"] == temp_celsius]
            if exp_data[observable].isna().all():
                continue

            exp_val = exp_data[observable].values[0]
            color = EXP_TEMP_COLORS.get(int(temp_celsius), "black")

            # Add horizontal dashed line
            fig.add_hline(
                y=exp_val,
                line_dash="dash",
                line_color=color,
                line_width=2
            )

            # Add star marker at the right edge
            fig.add_trace(
                go.Scatter(
                    x=[x_max_global * 0.95],
                    y=[exp_val],
                    mode="markers",
                    marker=dict(symbol="star", size=20, color=color,
                               line=dict(color="black", width=1)),
                    name=f"Exp T={int(temp_celsius)}°C",
                    showlegend=True,
                    hovertemplate=f'<b>Exp T={int(temp_celsius)}°C</b><br>' +
                                  f'<b>{observable}</b>: {exp_val:.3g}<extra></extra>',
                )
            )

    fig.update_layout(
        title="All Temperatures",
        xaxis_title=x_param,
        yaxis_title=y_label,
        height=600,
        template="plotly_white",
        font=dict(size=12),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02)
    )

    if use_log_y:
        fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)


def _render_single_temp(df, selected_temp, observable, x_param, group_by, group_values, color_map, use_log_y, df_exp=None):
    """Render single-temperature mode."""
    # Mapping from simulation T (LJ units) to experimental T_celsius
    temp_to_celsius = {0.05: 5, 0.1: 10, 0.2: 20, 0.3: 30}

    df_temp = df[abs(df["T"] - selected_temp) < 0.01].copy()

    if len(df_temp) == 0:
        st.warning(f"No data available for T={selected_temp}")
        return

    y_label = LATEX_LABELS.get(observable, observable)

    fig = go.Figure()

    # Track x-axis range for experimental markers
    x_max_global = 0

    for group_val in group_values:
        df_group = df_temp[abs(df_temp[group_by] - group_val) < 0.001].copy()

        if len(df_group) == 0:
            continue

        df_group = df_group.dropna(subset=[observable])

        if len(df_group) == 0:
            continue

        df_group = df_group.sort_values(x_param)

        # Track x-axis range
        if len(df_group) > 0:
            x_max_global = max(x_max_global, df_group[x_param].max())

        fig.add_trace(
            go.Scatter(
                x=df_group[x_param],
                y=df_group[observable],
                mode="lines+markers",
                name=f"{group_by}={group_val}",
                line=dict(color=color_map[group_val], width=2),
                marker=dict(size=8, color=color_map[group_val]),
                connectgaps=True,
                customdata=np.column_stack([
                    df_group["Pe"].values,
                    df_group["T"].values,
                    df_group["kappa"].values
                ]),
                hovertemplate='<b>Pe</b>: %{customdata[0]:.2f}<br>' +
                              '<b>T</b>: %{customdata[1]:.2f}<br>' +
                              '<b>κ</b>: %{customdata[2]:.2f}<br>' +
                              f'<b>{x_param}</b>: %{{x:.2f}}<br>' +
                              f'<b>{observable}</b>: %{{y:.3g}}<extra></extra>',
            )
        )

    # Add experimental data for the corresponding temperature
    if df_exp is not None and observable in df_exp.columns:
        exp_temp_celsius = temp_to_celsius.get(selected_temp)
        if exp_temp_celsius is not None:
            exp_data = df_exp[df_exp["T_celsius"] == exp_temp_celsius]
            if len(exp_data) > 0 and not exp_data[observable].isna().all():
                exp_val = exp_data[observable].values[0]
                color = EXP_TEMP_COLORS.get(exp_temp_celsius, "black")

                # Add horizontal dashed line
                fig.add_hline(
                    y=exp_val,
                    line_dash="dash",
                    line_color=color,
                    line_width=2
                )

                # Add star marker at the right edge
                fig.add_trace(
                    go.Scatter(
                        x=[x_max_global * 0.95],
                        y=[exp_val],
                        mode="markers",
                        marker=dict(symbol="star", size=20, color=color,
                                   line=dict(color="black", width=1)),
                        name=f"Exp T={exp_temp_celsius}°C",
                        showlegend=True,
                        hovertemplate=f'<b>Exp T={exp_temp_celsius}°C</b><br>' +
                                      f'<b>{observable}</b>: {exp_val:.3g}<extra></extra>',
                    )
                )

    fig.update_layout(
        title=f"T = {selected_temp}",
        xaxis_title=x_param,
        yaxis_title=y_label,
        height=500,
        template="plotly_white",
        font=dict(size=12),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02)
    )

    if use_log_y:
        fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)
