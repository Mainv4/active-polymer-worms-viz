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
    options=["confinement", "freespace"],
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

    # Error column mapping: maps observable to its error column
    ERROR_COLUMNS = {
        'D_long': 'D_long_error',
        'lp_free': 'lp_free_error',
        'lp_conf': 'lp_conf_error',
        'lp_free_individual': 'lp_free_individual_std',
        'tau_decorr_free': 'tau_decorr_free_error',
        'ttrap': 'ttrap_std'
    }

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
pe_range = st.sidebar.slider("Pe range", pe_min, pe_max, (0.30, 1.40), step=0.1)
pe_values = sorted(df["Pe"].unique())
pe_selected = st.sidebar.multiselect(
    "Pick specific Pe (optional)",
    options=pe_values,
    default=[],
    format_func=lambda x: f"{x:.2f}",
)

# T filter (range + discrete selection)
t_min, t_max = float(df["T"].min()), float(df["T"].max())
t_range = st.sidebar.slider("T range", t_min, t_max, (0.09, 0.30), step=0.01)
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
    "κ range", kappa_min, kappa_max, (0.50, 2.00), step=0.1
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
    "tau_decorr_free",
    "tau_decorr_cavity",
    "D_long",
    "H_conf",
    "lp_conf",
    "lp_conf_individual",
    "ttrap",
    "transloc_rate_per_hour",
    "transloc_success_rate",
]

# LaTeX labels mapping (using HTML entities for Greek letters)
latex_labels = {
    "Pe": "Pe",
    "T": "T",
    "kappa": "κ",
    "H_free": "H<sub>free</sub>",
    "lp_free": "ℓ<sub>p</sub> / L (corr, free)",
    "lp_free_individual": "ℓ<sub>p</sub> / L (indiv, free)",
    "tau_decorr_free": "τ<sub>decorr, free</sub> (s)",
    "tau_decorr_cavity": "τ<sub>decorr, cavity</sub> (s)",
    "D_long": "D<sub>long</sub> (mm²/s)",
    "H_conf": "H<sub>conf</sub>",
    "lp_conf": "ℓ<sub>p</sub> / L (corr, conf)",
    "lp_conf_individual": "ℓ<sub>p</sub> / L (indiv, conf)",
    "ttrap": "t<sub>trap</sub> (min)",
    "transloc_rate_per_hour": "Translocation rate (events/h)",
    "transloc_success_rate": "Translocation success rate (%)",
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
        x_var = st.selectbox("X-axis", numeric_cols, index=7)  # tau_decorr_cavity

    with col2:
        y_var = st.selectbox("Y-axis", numeric_cols, index=12)  # ttrap

    with col3:
        color_var = st.selectbox("Color by", ["None"] + numeric_cols, index=1)  # Pe

    with col4:
        size_var = st.selectbox("Size by", ["None"] + numeric_cols, index=3)  # kappa

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
            index=1,  # Pe
            help="Draw lines connecting points with the same parameter value",
        )

    # Division controls
    st.markdown("**Optional: Divide observables**")
    col_div1, col_div2, col_div3, col_div4 = st.columns(4)

    with col_div1:
        divide_x = st.checkbox("Divide X-axis by", value=False, key="divide_x_tab1")

    with col_div2:
        x_divisor = st.selectbox(
            "X divisor",
            numeric_cols,
            disabled=not divide_x,
            key="x_divisor_tab1"
        )

    with col_div3:
        divide_y = st.checkbox("Divide Y-axis by", value=False, key="divide_y_tab1")

    with col_div4:
        y_divisor = st.selectbox(
            "Y divisor",
            numeric_cols,
            disabled=not divide_y,
            key="y_divisor_tab1"
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
        # Prepare plot data (ensure unique columns)
        cols_needed = list(dict.fromkeys([x_var, y_var, "T", "Pe", "kappa"]))
        plot_data = df_filtered[cols_needed].copy()

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

        # Apply division transformations if requested
        x_col_plot = x_var
        y_col_plot = y_var
        x_var_display = x_var
        y_var_display = y_var
        latex_labels_custom = latex_labels.copy()

        # Check for self-division and warn
        if divide_x and x_divisor == x_var:
            st.warning("⚠️ Cannot divide X-axis by itself. Division disabled for X.")
            divide_x = False

        if divide_y and y_divisor == y_var:
            st.warning("⚠️ Cannot divide Y-axis by itself. Division disabled for Y.")
            divide_y = False

        # Add divisor columns to plot_data if needed
        if divide_x and x_divisor not in plot_data.columns:
            plot_data[x_divisor] = df_filtered[x_divisor]
        if divide_y and y_divisor not in plot_data.columns:
            plot_data[y_divisor] = df_filtered[y_divisor]

        # Perform X division
        if divide_x and x_divisor != x_var:
            x_col_plot = f"{x_var}_divided"
            plot_data[x_col_plot] = plot_data[x_var] / plot_data[x_divisor].replace(0, np.nan)
            x_var_display = f"{x_var} / {x_divisor}"
            latex_labels_custom[x_col_plot] = f"{latex_labels.get(x_var, x_var)} / {latex_labels.get(x_divisor, x_divisor)}"

        # Perform Y division
        if divide_y and y_divisor != y_var:
            y_col_plot = f"{y_var}_divided"
            plot_data[y_col_plot] = plot_data[y_var] / plot_data[y_divisor].replace(0, np.nan)
            y_var_display = f"{y_var} / {y_divisor}"
            latex_labels_custom[y_col_plot] = f"{latex_labels.get(y_var, y_var)} / {latex_labels.get(y_divisor, y_divisor)}"

        # Filter out NaN and infinite values from divided columns
        plot_data = plot_data.replace([np.inf, -np.inf], np.nan)
        plot_data = plot_data.dropna(subset=[x_col_plot, y_col_plot])

        # Always create the base scatter plot with color and size mapping
        # Build hover_data dynamically to handle divided columns
        hover_data_dict = {
            "Pe": ":.2f",
            "T": ":.2f",
            "kappa": ":.2f",
            x_col_plot: ":.4f",
            y_col_plot: ":.4f",
            color_col: ":.4f" if color_col else False,
            size_col: ":.4f" if size_col else False,
        }

        fig = px.scatter(
            plot_data,
            x=x_col_plot,
            y=y_col_plot,
            color=color_col,
            size=size_col,
            color_continuous_scale=colorscale.lower(),
            hover_data=hover_data_dict,
            labels={
                x_col_plot: latex_labels_custom.get(x_col_plot, x_var_display),
                y_col_plot: latex_labels_custom.get(y_col_plot, y_var_display),
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

                # Remove NaN values in x_col_plot and y_col_plot to avoid gaps in lines
                subset = subset.dropna(subset=[x_col_plot, y_col_plot])

                # Skip if less than 2 points remain (can't draw a line)
                if len(subset) < 2:
                    continue

                # Sort by x_col_plot, then by T and kappa for deterministic ordering
                # This ensures stable line drawing when multiple points have the same x_col_plot value
                subset = subset.sort_values(by=[x_col_plot, "T", "kappa"])

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
                        x=subset[x_col_plot],
                        y=subset[y_col_plot],
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
            xaxis_title=latex_labels_custom.get(x_col_plot, x_var_display),
            yaxis_title=latex_labels_custom.get(y_col_plot, y_var_display),
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
            ["H_free", "lp_free", "lp_free_individual", "tau_decorr_free", "tau_decorr_cavity", "D_long",
             "H_conf", "lp_conf", "lp_conf_individual", "ttrap",
             "transloc_rate_per_hour", "transloc_success_rate"],
            index=9  # ttrap
        )

    with col2:
        x_param = st.selectbox(
            "X-axis Parameter",
            ["Pe", "kappa", "H_free", "H_conf", "lp_free", "lp_conf",
             "tau_decorr_free", "tau_decorr_cavity",
             "transloc_rate_per_hour", "transloc_success_rate"],
            index=7  # tau_decorr_cavity
        )

    with col3:
        group_by = st.selectbox(
            "Group lines by",
            ["kappa", "Pe"],
            index=0 if x_param in ["Pe", "H_free", "H_conf", "lp_free", "lp_conf"] else 1
        )

    # Plot options
    opt_col1, opt_col2, opt_col3, opt_col4 = st.columns(4)
    with opt_col1:
        use_log_y = st.checkbox("Log scale (Y-axis)", value=False)
    with opt_col2:
        color_scheme_type = st.selectbox(
            "Color scheme",
            ["Discrete", "Gradient"],
            index=0
        )
        if color_scheme_type == "Discrete":
            color_palette = st.selectbox(
                "Color palette",
                ["Alphabet"],
                index=0,
                key="discrete_palette"
            )
        else:
            color_palette = st.selectbox(
                "Color scale",
                ["Viridis", "Plasma", "Turbo", "Inferno", "Magma", "Cividis", "Blues", "Greens", "Reds", "YlOrRd"],
                index=0,
                key="gradient_palette"
            )
    with opt_col3:
        show_secondary_lines = st.checkbox("Show secondary linking lines", value=False)
    with opt_col4:
        # Get available parameters for secondary linking (exclude current grouping parameter)
        available_params = ["Pe", "kappa", "T", "lp_free", "lp_conf"]
        available_params = [p for p in available_params if p != group_by]
        secondary_link_param = st.selectbox(
            "Secondary link by",
            available_params,
            index=0 if "kappa" not in available_params or group_by == "kappa" else available_params.index("kappa"),
            disabled=not show_secondary_lines
        )

    # Additional options row
    opt2_col1, opt2_col2, opt2_col3, opt2_col4 = st.columns(4)
    # Error bars functionality kept but button hidden for now
    # with opt2_col1:
    #     show_error_bars = st.checkbox(
    #         "Show error bars",
    #         value=False,
    #         disabled=observable not in ERROR_COLUMNS,
    #         help="Display error bars when available for selected observable"
    #     )
    show_error_bars = False  # Disabled for now

    # Division controls for Tab 2
    with opt2_col2:
        divide_y_tab2 = st.checkbox("Divide Y by observable", value=False, key="divide_y_tab2")

    with opt2_col3:
        y_divisor_tab2 = st.selectbox(
            "Y divisor",
            ["H_free", "lp_free", "lp_free_individual", "tau_decorr_free", "tau_decorr_cavity", "D_long",
             "H_conf", "lp_conf", "lp_conf_individual", "ttrap",
             "transloc_rate_per_hour", "transloc_success_rate"],
            disabled=not divide_y_tab2,
            key="y_divisor_tab2"
        )

    # Create plot based on display mode
    if len(df_filtered) > 0:
        # Get unique group values
        group_values = sorted(df_filtered[group_by].dropna().unique())

        # Color mapping based on scheme type
        if color_scheme_type == "Discrete":
            # Discrete color palette (qualitative)
            colors = px.colors.qualitative.__dict__[color_palette]
            color_map = {val: colors[i % len(colors)] for i, val in enumerate(group_values)}
        else:
            # Gradient color scale (sequential/continuous)
            import plotly.colors as pc

            # Normalize group values to [0, 1] range
            if len(group_values) > 1:
                min_val = min(group_values)
                max_val = max(group_values)
                normalized_vals = [(val - min_val) / (max_val - min_val) for val in group_values]
            else:
                normalized_vals = [0.5]  # Single value -> use middle of colorscale

            # Sample colors from the gradient scale
            color_map = {}
            for i, val in enumerate(group_values):
                # Get color at normalized position
                color_rgb = pc.sample_colorscale(color_palette.lower(), [normalized_vals[i]])[0]
                color_map[val] = color_rgb

        # Determine secondary linking parameter (if enabled)
        secondary_param = None
        secondary_values = []
        if show_secondary_lines:
            secondary_param = secondary_link_param
            secondary_values = sorted(df_filtered[secondary_param].dropna().unique())

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
                    # Use tolerance for filtering (similar to temperature filtering)
                    tolerance = 0.001
                    df_group = df_temp[np.abs(df_temp[group_by] - group_val) < tolerance].copy()

                    if len(df_group) == 0:
                        continue

                    # Remove any rows with NaN or inf in x_param or observable
                    df_group = df_group[
                        np.isfinite(df_group[x_param]) &
                        np.isfinite(df_group[observable])
                    ].copy()

                    if len(df_group) == 0:
                        continue

                    # Sort by x_param and reset index to ensure monotonic path
                    df_group = df_group.sort_values(x_param).reset_index(drop=True)

                    # Apply division transformation if requested
                    obs_col_plot = observable
                    obs_display = observable
                    obs_label = latex_labels.get(observable, observable)

                    # Check for self-division
                    if divide_y_tab2 and y_divisor_tab2 == observable:
                        if col_idx == 1 and group_val == group_values[0]:  # Show warning once
                            st.warning("⚠️ Cannot divide Y-axis by itself. Division disabled.")
                        divide_y_tab2_actual = False
                    else:
                        divide_y_tab2_actual = divide_y_tab2

                    # Perform Y division
                    if divide_y_tab2_actual and y_divisor_tab2 != observable:
                        # Ensure divisor column exists
                        if y_divisor_tab2 in df_group.columns:
                            obs_col_plot = f"{observable}_divided"
                            df_group[obs_col_plot] = df_group[observable] / df_group[y_divisor_tab2].replace(0, np.nan)
                            obs_display = f"{observable} / {y_divisor_tab2}"
                            obs_label = f"{latex_labels.get(observable, observable)} / {latex_labels.get(y_divisor_tab2, y_divisor_tab2)}"

                            # Filter out NaN and inf from division
                            df_group = df_group[np.isfinite(df_group[obs_col_plot])].copy()

                    # Prepare error bars if enabled
                    error_x_dict = None
                    error_y_dict = None

                    if show_error_bars:
                        # Check for X-axis error bars
                        if x_param in ERROR_COLUMNS:
                            error_col_x = ERROR_COLUMNS[x_param]
                            if error_col_x in df_group.columns:
                                error_values_x = df_group[error_col_x].fillna(0).values
                                if np.any(error_values_x > 0):
                                    error_x_dict = dict(
                                        type='data',
                                        array=error_values_x,
                                        visible=True
                                    )

                        # Check for Y-axis error bars
                        if observable in ERROR_COLUMNS:
                            error_col_y = ERROR_COLUMNS[observable]
                            if error_col_y in df_group.columns:
                                error_values_y = df_group[error_col_y].fillna(0).values
                                if np.any(error_values_y > 0):
                                    error_y_dict = dict(
                                        type='data',
                                        array=error_values_y,
                                        visible=True
                                    )

                    # Add trace with connectgaps=True to ensure continuous line
                    fig.add_trace(
                        go.Scatter(
                            x=df_group[x_param],
                            y=df_group[obs_col_plot],
                            mode="lines+markers",
                            name=f"{group_by}={group_val}",
                            line=dict(color=color_map[group_val], width=2),
                            marker=dict(size=8, color=color_map[group_val]),
                            error_x=error_x_dict,
                            error_y=error_y_dict,
                            connectgaps=True,
                            showlegend=(col_idx == 1),  # Only show legend for first subplot
                            legendgroup=str(group_val),  # Group legend items
                            customdata=np.column_stack((df_group['Pe'], df_group['T'], df_group['kappa'])),
                            hovertemplate='<b>Pe</b>: %{customdata[0]:.2f}<br>' +
                                          '<b>T</b>: %{customdata[1]:.2f}<br>' +
                                          '<b>κ</b>: %{customdata[2]:.2f}<br>' +
                                          f'<b>{x_param}</b>: %{{x:.4f}}<br>' +
                                          f'<b>{obs_display}</b>: %{{y:.4f}}<extra></extra>',
                        ),
                        row=1, col=col_idx
                    )

                # Add secondary linking lines (if enabled)
                if show_secondary_lines:
                    for sec_val in secondary_values:
                        # Use tolerance for filtering (similar to temperature filtering)
                        tolerance = 0.001
                        df_sec = df_temp[np.abs(df_temp[secondary_param] - sec_val) < tolerance].copy()

                        if len(df_sec) == 0:
                            continue

                        # Remove any rows with NaN or inf in x_param or observable
                        df_sec = df_sec[
                            np.isfinite(df_sec[x_param]) &
                            np.isfinite(df_sec[observable])
                        ].copy()

                        if len(df_sec) == 0:
                            continue

                        # Sort by x_param and reset index to ensure monotonic path
                        df_sec = df_sec.sort_values(x_param).reset_index(drop=True)

                        # Apply same division transformation for secondary lines
                        obs_col_sec = observable
                        if divide_y_tab2_actual and y_divisor_tab2 != observable:
                            if y_divisor_tab2 in df_sec.columns:
                                obs_col_sec = f"{observable}_divided"
                                df_sec[obs_col_sec] = df_sec[observable] / df_sec[y_divisor_tab2].replace(0, np.nan)
                                df_sec = df_sec[np.isfinite(df_sec[obs_col_sec])].copy()

                        # Add dashed trace with connectgaps=True to ensure continuous line
                        fig.add_trace(
                            go.Scatter(
                                x=df_sec[x_param],
                                y=df_sec[obs_col_sec],
                                mode="lines",
                                line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dash"),
                                connectgaps=True,
                                showlegend=False,
                                customdata=np.column_stack((df_sec['Pe'], df_sec['T'], df_sec['kappa'])),
                                hovertemplate='<b>[Secondary]</b><br>' +
                                              '<b>Pe</b>: %{customdata[0]:.2f}<br>' +
                                              '<b>T</b>: %{customdata[1]:.2f}<br>' +
                                              '<b>κ</b>: %{customdata[2]:.2f}<br>' +
                                              f'<b>{x_param}</b>: %{{x:.4f}}<br>' +
                                              f'<b>{obs_display}</b>: %{{y:.4f}}<extra></extra>',
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

            # Update axes (use obs_label defined earlier in the loop)
            # Note: obs_label is set in the last iteration of group_values loop
            for col_idx in range(1, 4):
                fig.update_xaxes(title_text=x_param, row=1, col=col_idx)
                if col_idx == 1:
                    # Use the transformed label if division was applied
                    y_label = obs_label if divide_y_tab2_actual else latex_labels.get(observable, observable)
                    fig.update_yaxes(title_text=y_label, row=1, col=col_idx)

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
                # Use tolerance for filtering (similar to temperature filtering)
                tolerance = 0.001
                df_group = df_temp[np.abs(df_temp[group_by] - group_val) < tolerance].copy()

                if len(df_group) == 0:
                    continue

                # Remove any rows with NaN or inf in x_param or observable
                df_group = df_group[
                    np.isfinite(df_group[x_param]) &
                    np.isfinite(df_group[observable])
                ].copy()

                if len(df_group) == 0:
                    continue

                # Sort by x_param and reset index to ensure monotonic path
                df_group = df_group.sort_values(x_param).reset_index(drop=True)

                # Apply division transformation if requested
                obs_col_plot = observable
                obs_display = observable
                obs_label = latex_labels.get(observable, observable)

                # Check for self-division
                if divide_y_tab2 and y_divisor_tab2 == observable:
                    if group_val == group_values[0]:  # Show warning once
                        st.warning("⚠️ Cannot divide Y-axis by itself. Division disabled.")
                    divide_y_tab2_actual = False
                else:
                    divide_y_tab2_actual = divide_y_tab2

                # Perform Y division
                if divide_y_tab2_actual and y_divisor_tab2 != observable:
                    # Ensure divisor column exists
                    if y_divisor_tab2 in df_group.columns:
                        obs_col_plot = f"{observable}_divided"
                        df_group[obs_col_plot] = df_group[observable] / df_group[y_divisor_tab2].replace(0, np.nan)
                        obs_display = f"{observable} / {y_divisor_tab2}"
                        obs_label = f"{latex_labels.get(observable, observable)} / {latex_labels.get(y_divisor_tab2, y_divisor_tab2)}"

                        # Filter out NaN and inf from division
                        df_group = df_group[np.isfinite(df_group[obs_col_plot])].copy()

                # Prepare error bars if enabled
                error_x_dict = None
                error_y_dict = None

                if show_error_bars:
                    # Check for X-axis error bars
                    if x_param in ERROR_COLUMNS:
                        error_col_x = ERROR_COLUMNS[x_param]
                        if error_col_x in df_group.columns:
                            error_values_x = df_group[error_col_x].fillna(0).values
                            if np.any(error_values_x > 0):
                                error_x_dict = dict(
                                    type='data',
                                    array=error_values_x,
                                    visible=True
                                )

                    # Check for Y-axis error bars
                    if observable in ERROR_COLUMNS:
                        error_col_y = ERROR_COLUMNS[observable]
                        if error_col_y in df_group.columns:
                            error_values_y = df_group[error_col_y].fillna(0).values
                            if np.any(error_values_y > 0):
                                error_y_dict = dict(
                                    type='data',
                                    array=error_values_y,
                                    visible=True
                                )

                # Add trace with connectgaps=True to ensure continuous line
                fig.add_trace(
                    go.Scatter(
                        x=df_group[x_param],
                        y=df_group[obs_col_plot],
                        mode="lines+markers",
                        name=f"{group_by}={group_val}",
                        line=dict(color=color_map[group_val], width=2),
                        marker=dict(size=8, color=color_map[group_val]),
                        error_x=error_x_dict,
                        error_y=error_y_dict,
                        connectgaps=True,
                        customdata=np.column_stack((df_group['Pe'], df_group['T'], df_group['kappa'])),
                        hovertemplate='<b>Pe</b>: %{customdata[0]:.2f}<br>' +
                                      '<b>T</b>: %{customdata[1]:.2f}<br>' +
                                      '<b>κ</b>: %{customdata[2]:.2f}<br>' +
                                      f'<b>{x_param}</b>: %{{x:.4f}}<br>' +
                                      f'<b>{obs_display}</b>: %{{y:.4f}}<extra></extra>',
                    )
                )

            # Add secondary linking lines (if enabled)
            if show_secondary_lines:
                for sec_val in secondary_values:
                    # Use tolerance for filtering (similar to temperature filtering)
                    tolerance = 0.001
                    df_sec = df_temp[np.abs(df_temp[secondary_param] - sec_val) < tolerance].copy()

                    if len(df_sec) == 0:
                        continue

                    # Remove any rows with NaN or inf in x_param or observable
                    df_sec = df_sec[
                        np.isfinite(df_sec[x_param]) &
                        np.isfinite(df_sec[observable])
                    ].copy()

                    if len(df_sec) == 0:
                        continue

                    # Sort by x_param and reset index to ensure monotonic path
                    df_sec = df_sec.sort_values(x_param).reset_index(drop=True)

                    # Apply same division transformation for secondary lines
                    obs_col_sec = observable
                    if divide_y_tab2_actual and y_divisor_tab2 != observable:
                        if y_divisor_tab2 in df_sec.columns:
                            obs_col_sec = f"{observable}_divided"
                            df_sec[obs_col_sec] = df_sec[observable] / df_sec[y_divisor_tab2].replace(0, np.nan)
                            df_sec = df_sec[np.isfinite(df_sec[obs_col_sec])].copy()

                    # Add dashed trace with connectgaps=True to ensure continuous line
                    fig.add_trace(
                        go.Scatter(
                            x=df_sec[x_param],
                            y=df_sec[obs_col_sec],
                            mode="lines",
                            line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dash"),
                            connectgaps=True,
                            showlegend=False,
                            customdata=np.column_stack((df_sec['Pe'], df_sec['T'], df_sec['kappa'])),
                            hovertemplate='<b>[Secondary]</b><br>' +
                                          '<b>Pe</b>: %{customdata[0]:.2f}<br>' +
                                          '<b>T</b>: %{customdata[1]:.2f}<br>' +
                                          '<b>κ</b>: %{customdata[2]:.2f}<br>' +
                                          f'<b>{x_param}</b>: %{{x:.4f}}<br>' +
                                          f'<b>{obs_display}</b>: %{{y:.4f}}<extra></extra>',
                        )
                    )

            # Update layout
            fig.update_layout(
                title=f"T = {selected_temp}",
                xaxis_title=x_param,
                yaxis_title=obs_label if divide_y_tab2_actual else latex_labels.get(observable, observable),
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
