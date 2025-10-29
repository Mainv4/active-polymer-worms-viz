#!/usr/bin/env python3
"""
Interactive Data Visualization for Active Polymer Simulations
=============================================================

Streamlit application for exploring complete_simulation_data.csv interactively.

Usage:
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Active Polymer Data Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title
st.title("Active Polymer Simulation Data Explorer")
st.markdown(
    """
Web application for exploring active polymer simulation observables (N=40 beads).
Visualize data from free space and confinement experiments using fuzzy-matched parameter sets
across Péclet number (Pe), temperature (T), and bending rigidity (κ).

**Experimental data:** :blue[⭐ 10°C], :green[⭐ 20°C], :red[⭐ 30°C]

**See `README.md` on the GitHub repo for details:**  
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Mainv4/active-polymer-worms-viz/)
"""
)


# Load data
@st.cache_data
def load_data(data_source="freespace"):
    """Load the compiled simulation data.

    Args:
        data_source: 'freespace' or 'confinement' reference base
    """
    if data_source == "freespace":
        csv_path = Path(__file__).parent / "data_freespace.csv"
    else:
        csv_path = Path(__file__).parent / "data_confinement.csv"

    # Read CSV, skipping comment lines
    df = pd.read_csv(csv_path, comment="#")

    # Round parameter values to avoid duplicates from fuzzy matching
    df["Pe"] = df["Pe"].round(2)
    df["T"] = df["T"].round(2)
    df["kappa"] = df["kappa"].round(2)

    return df


@st.cache_data
def load_exp_data():
    """Load experimental data if available."""
    csv_path = Path(__file__).parent / "data_exp.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# Sidebar - Data Source Selection
st.sidebar.header("🗂️ Data Source")

data_source = st.sidebar.radio(
    "Reference base:",
    options=["freespace", "confinement"],
    format_func=lambda x: (
        "Free Space (H_free)" if x == "freespace" else "Confinement (H_conf)"
    ),
    help="Choose which observable is used as the reference base for fuzzy matching. "
    "Free space maximizes data points for free space correlations, "
    "confinement maximizes data points for confinement correlations.",
)

try:
    df = load_data(data_source)
    df_exp = load_exp_data()

    # Display data source info
    if data_source == "freespace":
        st.sidebar.success(f"✓ Loaded {len(df)} points (H_free base)")
    else:
        st.sidebar.success(f"✓ Loaded {len(df)} points (H_conf base)")

    if df_exp is not None:
        st.sidebar.info(f"✓ Loaded {len(df_exp)} experimental points")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar - Filters
st.sidebar.header("📊 Filters")

# Parameter filters
st.sidebar.subheader("Parameters")

# Pe filter (range + discrete selection)
pe_min, pe_max = float(df["Pe"].min()), float(df["Pe"].max())
pe_range = st.sidebar.slider("Pe range", pe_min, pe_max, (pe_min, pe_max), step=0.1)
pe_values = sorted(df["Pe"].unique())
pe_selected = st.sidebar.multiselect(
    "Pick specific Pe (optional)",
    options=pe_values,
    default=[],
    format_func=lambda x: f"{x:.2f}",
)

# T filter (range + discrete selection)
t_min, t_max = float(df["T"].min()), float(df["T"].max())
t_range = st.sidebar.slider("T range", t_min, t_max, (t_min, t_max), step=0.01)
t_values = sorted(df["T"].unique())
t_selected = st.sidebar.multiselect(
    "Pick specific T (optional)",
    options=t_values,
    default=[],
    format_func=lambda x: f"{x:.2f}",
)

# κ filter (range + discrete selection)
kappa_min, kappa_max = float(df["kappa"].min()), float(df["kappa"].max())
kappa_range = st.sidebar.slider(
    "κ range", kappa_min, kappa_max, (kappa_min, kappa_max), step=0.1
)
kappa_values = sorted(df["kappa"].unique())
kappa_selected = st.sidebar.multiselect(
    "Pick specific κ (optional)",
    options=kappa_values,
    default=[],
    format_func=lambda x: f"{x:.2f}",
)

# NaN handling
st.sidebar.subheader("Data Quality")
exclude_nan = st.sidebar.checkbox("Exclude rows with any NaN", value=False)

# Documentation
st.sidebar.markdown("---")
with st.sidebar.expander("📖 Documentation"):
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text()
        st.markdown(readme_content)
    else:
        st.warning("README.md not found")

# Apply filters with hybrid logic (range AND discrete selection)

# Pe filter logic
if len(pe_selected) > 0:
    # If discrete selection is made, use intersection of range and selected values
    pe_mask = (
        df["Pe"].isin(pe_selected)
        & (df["Pe"] >= pe_range[0])
        & (df["Pe"] <= pe_range[1])
    )
else:
    # If no discrete selection, use only range
    pe_mask = (df["Pe"] >= pe_range[0]) & (df["Pe"] <= pe_range[1])

# T filter logic
if len(t_selected) > 0:
    # If discrete selection is made, use intersection of range and selected values
    t_mask = (
        df["T"].isin(t_selected) & (df["T"] >= t_range[0]) & (df["T"] <= t_range[1])
    )
else:
    # If no discrete selection, use only range
    t_mask = (df["T"] >= t_range[0]) & (df["T"] <= t_range[1])

# κ filter logic
if len(kappa_selected) > 0:
    # If discrete selection is made, use intersection of range and selected values
    kappa_mask = (
        df["kappa"].isin(kappa_selected)
        & (df["kappa"] >= kappa_range[0])
        & (df["kappa"] <= kappa_range[1])
    )
else:
    # If no discrete selection, use only range
    kappa_mask = (df["kappa"] >= kappa_range[0]) & (df["kappa"] <= kappa_range[1])

# Combine all masks
mask = pe_mask & t_mask & kappa_mask

df_filtered = df[mask].copy()

if exclude_nan:
    df_filtered = df_filtered.dropna()

st.sidebar.info(f"**{len(df_filtered)}** / {len(df)} points displayed")

# ==================================================================================
# DATA PREPARATION
# ==================================================================================

# Available columns for plotting
numeric_cols = [
    "Pe",
    "T",
    "kappa",
    "H_free",
    "lp_free",
    "lp_free_individual",
    "tau_decorr",
    "D_long",
    "H_conf",
    "lp_conf",
    "lp_conf_individual",
    "ttrap",
]

# LaTeX labels mapping (using HTML entities for Greek letters)
latex_labels = {
    "Pe": "Pe",
    "T": "T",
    "kappa": "κ",
    "H_free": "H<sub>free</sub>",
    "lp_free": "ℓ<sub>p</sub> / L (corr, free)",
    "lp_free_individual": "ℓ<sub>p</sub> / L (indiv, free)",
    "tau_decorr": "τ<sub>decorr</sub> (s)",
    "D_long": "D<sub>long</sub> (mm²/s)",
    "H_conf": "H<sub>conf</sub>",
    "lp_conf": "ℓ<sub>p</sub> / L (corr, conf)",
    "lp_conf_individual": "ℓ<sub>p</sub> / L (indiv, conf)",
    "ttrap": "t<sub>trap</sub> (min)",
}

# ==================================================================================
# MAIN PANEL - TABS
# ==================================================================================

# Create tabs
tab1, tab2 = st.tabs(["📈 Scatter Correlations", "📊 Parameter Space"])

# ==================================================================================
# TAB 1: SCATTER CORRELATIONS
# ==================================================================================
with tab1:
    st.header("Interactive Scatter Plot")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        x_var = st.selectbox("X-axis", numeric_cols, index=11)  # ttrap

    with col2:
        y_var = st.selectbox("Y-axis", numeric_cols, index=3)  # H_free

    with col3:
        color_var = st.selectbox("Color by", ["None"] + numeric_cols, index=3)  # kappa

    with col4:
        size_var = st.selectbox("Size by", ["None"] + numeric_cols, index=1)  # Pe

    # Plot options
    col_opt1, col_opt2, col_opt3 = st.columns(3)

    with col_opt1:
        colorscale = st.selectbox(
            "Colorscale",
            ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo"],
            index=0,
        )

    with col_opt2:
        log_scale = st.multiselect("Log scale", ["X-axis", "Y-axis"], default=[])

    with col_opt3:
        connect_by = st.selectbox(
            "Connect points by",
            ["None", "Pe", "T", "kappa"],
            index=0,
            help="Draw lines connecting points with the same parameter value",
        )

    # Highlight series selector (only shown when connecting lines)
    highlight_series = None
    if connect_by != "None":
        # Get unique values for the selected connection parameter
        unique_vals = sorted(df_filtered[connect_by].unique())
        highlight_options = ["All (show all)"] + [
            f"{connect_by}={val:.2f}" for val in unique_vals
        ]
        highlight_series = st.selectbox(
            "Highlight series",
            highlight_options,
            index=0,
            help="Focus on a specific series by dimming others",
        )

    # Create plot
    if len(df_filtered) == 0:
        st.warning("⚠️ No data points match the current filters")
    else:
        # Prepare plot data
        plot_data = df_filtered[[x_var, y_var, "T", "Pe", "kappa"]].copy()

        # Handle color
        if color_var != "None":
            plot_data["color"] = df_filtered[color_var]
            color_col = "color"
        else:
            color_col = None

        # Handle size
        if size_var != "None":
            plot_data["size"] = df_filtered[size_var]
            size_col = "size"
        else:
            size_col = None

        # Always create the base scatter plot with color and size mapping
        fig = px.scatter(
            plot_data,
            x=x_var,
            y=y_var,
            color=color_col,
            size=size_col,
            color_continuous_scale=colorscale.lower(),
            hover_data={
                "Pe": ":.2f",
                "T": ":.2f",
                "kappa": ":.2f",
                x_var: ":.4f",
                y_var: ":.4f",
                color_col: ":.4f" if color_col else False,
                size_col: ":.4f" if size_col else False,
            },
            labels={
                "color": latex_labels[color_var] if color_var != "None" else "",
                "size": latex_labels[size_var] if size_var != "None" else "",
            },
        )

        # Dim scatter points if a specific series is highlighted
        if highlight_series and highlight_series != "All (show all)":
            fig.update_traces(marker=dict(opacity=0.5), selector=dict(mode="markers"))

        # Add connecting lines if requested
        if connect_by != "None":
            # Get unique values of the connection parameter
            unique_vals = sorted(plot_data[connect_by].unique())

            # Extract highlighted value if a specific series is selected
            highlighted_val = None
            if highlight_series and highlight_series != "All (show all)":
                # Parse "Pe=0.50" -> 0.50
                highlighted_val = float(highlight_series.split("=")[1])

            for val in unique_vals:
                # Filter data for this parameter value
                mask = plot_data[connect_by] == val
                subset = plot_data[mask].copy()

                # Remove NaN values in x_var and y_var to avoid gaps in lines
                subset = subset.dropna(subset=[x_var, y_var])

                # Skip if less than 2 points remain (can't draw a line)
                if len(subset) < 2:
                    continue

                # Sort by x_var, then by T and kappa for deterministic ordering
                # This ensures stable line drawing when multiple points have the same x_var value
                subset = subset.sort_values(by=[x_var, "T", "kappa"])

                # Determine line style based on highlighting
                if highlighted_val is None:
                    # No highlighting: default style
                    line_width = 2
                    line_opacity = 1.0
                    line_color = "gray"
                elif abs(val - highlighted_val) < 1e-6:
                    # This is the highlighted series: make it prominent
                    line_width = 4
                    line_opacity = 1.0
                    line_color = "#FF4B4B"  # Streamlit red
                else:
                    # Other series: dim them
                    line_width = 1
                    line_opacity = 0.15
                    line_color = "lightgray"

                # Add line trace (no markers, just lines)
                fig.add_trace(
                    go.Scatter(
                        x=subset[x_var],
                        y=subset[y_var],
                        mode="lines",
                        line=dict(color=line_color, width=line_width),
                        opacity=line_opacity,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        # Update layout
        fig.update_layout(
            height=600,
            template="plotly_white",
            font=dict(size=14),
            xaxis_title=latex_labels[x_var],
            yaxis_title=latex_labels[y_var],
        )

        # Apply log scales
        if "X-axis" in log_scale:
            fig.update_xaxes(type="log")
        if "Y-axis" in log_scale:
            fig.update_yaxes(type="log")

        # Add experimental data points if available
        if df_exp is not None and x_var in df_exp.columns and y_var in df_exp.columns:
            exp_plot_data = df_exp[["T_celsius", x_var, y_var]].dropna()
            if len(exp_plot_data) > 0:
                # Color map for temperatures
                temp_colors = {10: "blue", 20: "green", 30: "red"}

                # Add one trace per temperature
                for temp in exp_plot_data["T_celsius"].unique():
                    temp_data = exp_plot_data[exp_plot_data["T_celsius"] == temp]
                    fig.add_trace(
                        go.Scatter(
                            x=temp_data[x_var],
                            y=temp_data[y_var],
                            mode="markers",
                            marker=dict(
                                symbol="star",
                                size=20,
                                color=temp_colors.get(temp, "black"),
                                line=dict(color="black", width=2),
                            ),
                            name=f"Exp T={int(temp)}°C",
                            showlegend=False,
                        )
                )

    # Display plot
    st.plotly_chart(fig, use_container_width=True)

# ==================================================================================
# TAB 2: PARAMETER SPACE
# ==================================================================================
with tab2:
    st.header("Parameter Space Comparison")
    st.markdown("""
    Compare how observables vary across parameter space for different temperatures.
    """)

    # Display mode selector
    display_mode = st.radio(
        "Display mode",
        ["3 temperatures", "Single temperature"],
        horizontal=True
    )

    # Temperature selector (only for single temperature mode)
    if display_mode == "Single temperature":
        selected_temp = st.selectbox(
            "Select temperature",
            [0.05, 0.1, 0.2],
            index=1
        )

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        observable = st.selectbox(
            "Observable (Y-axis)",
            ["H_free", "lp_free", "lp_free_individual", "tau_decorr", "D_long",
             "H_conf", "lp_conf", "lp_conf_individual", "ttrap"],
            index=0
        )

    with col2:
        x_param = st.selectbox(
            "X-axis Parameter",
            ["Pe", "kappa", "lp_free", "lp_conf"],
            index=0
        )

    with col3:
        group_by = st.selectbox(
            "Group lines by",
            ["kappa", "Pe"],
            index=0 if x_param in ["Pe", "lp_free", "lp_conf"] else 1
        )

    # Plot options
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        use_log_y = st.checkbox("Log scale (Y-axis)", value=False)
    with opt_col2:
        color_palette = st.selectbox(
            "Color palette",
            ["Plotly", "D3", "G10", "T10", "Alphabet"],
            index=0
        )

    # Create plot based on display mode
    if len(df_filtered) > 0:
        # Get unique group values
        group_values = sorted(df_filtered[group_by].dropna().unique())

        # Color mapping
        colors = px.colors.qualitative.__dict__[color_palette]
        color_map = {val: colors[i % len(colors)] for i, val in enumerate(group_values)}

        if display_mode == "3 temperatures":
            # ============================================================
            # 3-TEMPERATURE MODE: Create subplots
            # ============================================================
            temps = [0.05, 0.1, 0.2]

            # Create subplots
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=[f"T = {T}" for T in temps],
                horizontal_spacing=0.08
            )

            # Plot for each temperature
            for col_idx, temp in enumerate(temps, 1):
                # Filter data for this temperature
                df_temp = df_filtered[abs(df_filtered["T"] - temp) < 0.01].copy()

                if len(df_temp) == 0:
                    continue

                # Plot for each group
                for group_val in group_values:
                    df_group = df_temp[df_temp[group_by] == group_val].copy()

                    if len(df_group) == 0:
                        continue

                    # Sort by x_param
                    df_group = df_group.sort_values(x_param)

                    # Add trace
                    fig.add_trace(
                        go.Scatter(
                            x=df_group[x_param],
                            y=df_group[observable],
                            mode="lines+markers",
                            name=f"{group_by}={group_val}",
                            line=dict(color=color_map[group_val], width=2),
                            marker=dict(size=8, color=color_map[group_val]),
                            showlegend=(col_idx == 1),  # Only show legend for first subplot
                            legendgroup=str(group_val),  # Group legend items
                            customdata=np.column_stack((df_group['Pe'], df_group['T'], df_group['kappa'])),
                            hovertemplate='<b>Pe</b>: %{customdata[0]:.2f}<br>' +
                                          '<b>T</b>: %{customdata[1]:.2f}<br>' +
                                          '<b>κ</b>: %{customdata[2]:.2f}<br>' +
                                          f'<b>{x_param}</b>: %{{x:.4f}}<br>' +
                                          f'<b>{observable}</b>: %{{y:.4f}}<extra></extra>',
                        ),
                        row=1, col=col_idx
                    )

            # Update layout
            fig.update_layout(
                height=500,
                template="plotly_white",
                font=dict(size=12),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.02
                )
            )

            # Update axes
            for col_idx in range(1, 4):
                fig.update_xaxes(title_text=x_param, row=1, col=col_idx)
                if col_idx == 1:
                    fig.update_yaxes(title_text=latex_labels[observable], row=1, col=col_idx)

                if use_log_y:
                    fig.update_yaxes(type="log", row=1, col=col_idx)

        else:
            # ============================================================
            # SINGLE-TEMPERATURE MODE: Create single plot
            # ============================================================
            # Filter data for selected temperature
            df_temp = df_filtered[abs(df_filtered["T"] - selected_temp) < 0.01].copy()

            if len(df_temp) == 0:
                st.warning(f"⚠️ No data available for T={selected_temp}")
                st.stop()

            # Create figure
            fig = go.Figure()

            # Plot for each group
            for group_val in group_values:
                df_group = df_temp[df_temp[group_by] == group_val].copy()

                if len(df_group) == 0:
                    continue

                # Sort by x_param
                df_group = df_group.sort_values(x_param)

                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=df_group[x_param],
                        y=df_group[observable],
                        mode="lines+markers",
                        name=f"{group_by}={group_val}",
                        line=dict(color=color_map[group_val], width=2),
                        marker=dict(size=8, color=color_map[group_val]),
                    )
                )

            # Update layout
            fig.update_layout(
                title=f"T = {selected_temp}",
                xaxis_title=x_param,
                yaxis_title=latex_labels[observable],
                height=500,
                template="plotly_white",
                font=dict(size=12),
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.02
                )
            )

            # Apply log scale if requested
            if use_log_y:
                fig.update_yaxes(type="log")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No data points match the current filters.")

# ==================================================================================
# DATA TABLE
# ==================================================================================
st.markdown("---")
if len(df_filtered) > 0:
    # Data table
    st.header("📋 Data Table")

    show_all = st.checkbox("Show all columns", value=False)

    if show_all:
        st.dataframe(df_filtered, width="stretch", height=400)
    else:
        display_cols = ["Pe", "T", "kappa", x_var, y_var]
        if color_var != "None" and color_var not in display_cols:
            display_cols.append(color_var)
        if size_var != "None" and size_var not in display_cols:
            display_cols.append(size_var)

        # Remove duplicates while preserving order
        seen = set()
        display_cols = [
            col for col in display_cols if not (col in seen or seen.add(col))
        ]

        st.dataframe(df_filtered[display_cols], width="stretch", height=400)

    # Download filtered data
    st.download_button(
        label="⬇️ Download filtered data (CSV)",
        data=df_filtered.to_csv(index=False),
        file_name="filtered_simulation_data.csv",
        mime="text/csv",
    )

# Footer
st.markdown("---")
st.markdown(
    """
    **Data source:** `data_freespace.csv` or `data_confinement.csv` (polymer with N=40 beads)

    **Note:** Uses fuzzy matching (tolerance=0.05) to combine free space and confinement data.
    """
)
