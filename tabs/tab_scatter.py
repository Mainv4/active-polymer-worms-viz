"""
Tab 1: Scatter Correlations
Interactive scatter plot for exploring correlations between observables.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.constants import LATEX_LABELS, NUMERIC_COLS, EXP_TEMP_COLORS, ERROR_COLUMNS
from tabs.tab_trapping import (
    compute_ttrap_for_scatter,
    load_all_trapping_times_by_params,
    load_exp_trapping_data,
)

def _render_ttrap_sidebar():
    """Render τ_trap histogram settings in the sidebar. Returns config dict."""
    with st.sidebar.expander("τ_trap histogram settings", expanded=False):
        method = st.radio(
            "τ_trap value",
            ["Mean", "Fit (τ_c)"],
            index=0,
            key="ttrap_sidebar_method",
            help="Mean: average of trapping events. Fit: characteristic time from exponential fit.",
        )
        fit_type = "Log-space"
        if method == "Fit (τ_c)":
            fit_type = st.radio(
                "Fit method",
                ["Log-space", "Direct"],
                index=0,
                key="ttrap_sidebar_fit_type",
                help="Log-space: linear regression on log(P), robust. Direct: fit on P(τ).",
                horizontal=True,
            )
        max_cutoff = st.slider(
            "Max τ cutoff (min)", 0.5, 30.0, 15.0, step=0.5,
            key="ttrap_sidebar_cutoff",
        )
        n_bins = st.slider(
            "Number of bins", 5, 100, 30,
            key="ttrap_sidebar_bins",
            disabled=(method == "Mean"),
        )
        use_log_bins = st.checkbox(
            "Log-spaced bins", value=False,
            key="ttrap_sidebar_log_bins",
            disabled=(method == "Mean"),
            help="Use logarithmically spaced bins instead of linear",
        )

    # Map to internal method names used by compute_ttrap_for_scatter
    if method == "Fit (τ_c)":
        internal_method = f"τ_c ({fit_type.lower()})" if fit_type == "Log-space" else "τ_c (direct fit)"
    else:
        internal_method = "Mean"

    return {
        "method": internal_method,
        "n_bins": n_bins,
        "max_cutoff": max_cutoff,
        "use_log_bins": use_log_bins,
    }


def render_scatter_tab(df_filtered, df_exp):
    """Render the Scatter Correlations tab.

    Args:
        df_filtered: Filtered simulation DataFrame
        df_exp: Experimental data DataFrame (or None)
    """
    st.header("Interactive Scatter Plot")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        x_var = st.selectbox("X-axis", NUMERIC_COLS, index=NUMERIC_COLS.index("tau_decorr_cavity"))

    with col2:
        y_var = st.selectbox("Y-axis", NUMERIC_COLS, index=NUMERIC_COLS.index("ttrap"))

    with col3:
        color_var = st.selectbox("Color by", ["None"] + NUMERIC_COLS, index=1)  # Pe

    with col4:
        size_var = st.selectbox("Size by", ["None"] + NUMERIC_COLS, index=NUMERIC_COLS.index("kappa") + 1)

    # Trapping time config from sidebar
    ttrap_on_axis = (x_var == "ttrap" or y_var == "ttrap")
    ttrap_cfg = _render_ttrap_sidebar() if ttrap_on_axis else None
    ttrap_method = ttrap_cfg["method"] if ttrap_cfg else "Mean"
    ttrap_n_bins = ttrap_cfg["n_bins"] if ttrap_cfg else 30
    ttrap_max_cutoff = ttrap_cfg["max_cutoff"] if ttrap_cfg else 15.0
    ttrap_use_log_bins = ttrap_cfg["use_log_bins"] if ttrap_cfg else False

    # Plot options
    col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)

    with col_opt1:
        colorscale = st.selectbox(
            "Colorscale",
            ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Turbo"],
            index=0,
        )

    with col_opt2:
        log_scale = st.multiselect("Log scale", ["X-axis", "Y-axis", "Colorbar"], default=[])

    with col_opt3:
        connect_by = st.selectbox(
            "Connect points by",
            ["None", "Pe", "Pe_true", "T", "kappa"],
            index=1,  # Pe
            help="Draw lines connecting points with the same parameter value",
        )

    with col_opt4:
        # Error bars available if ttrap is on an axis (method error)
        # or if any plotted observable has an ERROR_COLUMNS entry
        has_error_x = (x_var == "ttrap" and ttrap_on_axis) or x_var in ERROR_COLUMNS
        has_error_y = (y_var == "ttrap" and ttrap_on_axis) or y_var in ERROR_COLUMNS
        show_error_bars = st.checkbox(
            "Show error bars",
            value=False,
            key="scatter_error_bars",
            disabled=not (has_error_x or has_error_y),
            help="Show uncertainty on axes where error data is available",
        )

    # Division controls
    st.markdown("**Optional: Divide observables**")
    col_div1, col_div2, col_div3, col_div4 = st.columns(4)

    with col_div1:
        divide_x = st.checkbox("Divide X-axis by", value=False, key="divide_x_tab1")

    with col_div2:
        x_divisor = st.selectbox(
            "X divisor",
            NUMERIC_COLS,
            disabled=not divide_x,
            key="x_divisor_tab1"
        )

    with col_div3:
        divide_y = st.checkbox("Divide Y-axis by", value=False, key="divide_y_tab1")

    with col_div4:
        y_divisor = st.selectbox(
            "Y divisor",
            NUMERIC_COLS,
            disabled=not divide_y,
            key="y_divisor_tab1"
        )

    # Highlight series selector (only shown when connecting lines)
    highlight_series = None
    if connect_by != "None":
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
        return x_var, y_var, color_var, size_var

    # Recompute ttrap if a non-default method is selected
    if ttrap_on_axis and ttrap_method != "Mean":
        all_times = load_all_trapping_times_by_params(N=40)
        new_values = []
        new_errors = []
        for _, row in df_filtered.iterrows():
            key = (round(row["Pe"], 2), round(row["T"], 2), round(row["kappa"], 2))
            times = all_times.get(key, np.array([]))
            val, err = compute_ttrap_for_scatter(
                ttrap_method, times, ttrap_n_bins, ttrap_max_cutoff, ttrap_use_log_bins
            )
            new_values.append(val)
            new_errors.append(err)
        df_filtered = df_filtered.copy()
        df_filtered["ttrap"] = new_values
        df_filtered["ttrap_method_error"] = new_errors
    elif ttrap_on_axis and ttrap_method == "Mean":
        # For Mean method with custom cutoff, recompute if cutoff != default
        if ttrap_max_cutoff != 15.0:
            all_times = load_all_trapping_times_by_params(N=40)
            new_values = []
            new_errors = []
            for _, row in df_filtered.iterrows():
                key = (round(row["Pe"], 2), round(row["T"], 2), round(row["kappa"], 2))
                times = all_times.get(key, np.array([]))
                val, err = compute_ttrap_for_scatter(
                    "Mean", times, ttrap_n_bins, ttrap_max_cutoff, ttrap_use_log_bins
                )
                new_values.append(val)
                new_errors.append(err)
            df_filtered = df_filtered.copy()
            df_filtered["ttrap"] = new_values
            df_filtered["ttrap_method_error"] = new_errors
        else:
            # Default Mean: compute SE from ttrap_std in CSV
            df_filtered = df_filtered.copy()
            if "ttrap_std" in df_filtered.columns and "transloc_total_events" in df_filtered.columns:
                n_events = df_filtered["transloc_total_events"].replace(0, np.nan)
                df_filtered["ttrap_method_error"] = df_filtered["ttrap_std"] / np.sqrt(n_events)
            elif "ttrap_std" in df_filtered.columns:
                # Fallback: use std as-is (overestimate, but visible)
                df_filtered["ttrap_method_error"] = df_filtered["ttrap_std"]

    # Prepare plot data (ensure unique columns)
    cols_needed = list(dict.fromkeys([x_var, y_var, "T", "Pe", "Pe_true", "kappa"]))
    plot_data = df_filtered[cols_needed].copy()

    # Handle color
    use_log_colorbar = "Colorbar" in log_scale
    if color_var != "None":
        if use_log_colorbar:
            # Apply log transformation to color values
            color_values = df_filtered[color_var].replace(0, np.nan)
            plot_data["color"] = np.log10(color_values)
        else:
            plot_data["color"] = df_filtered[color_var]
        color_col = "color"
    else:
        color_col = None
        use_log_colorbar = False  # No colorbar to log-scale

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
    latex_labels_custom = LATEX_LABELS.copy()

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
        latex_labels_custom[x_col_plot] = f"{LATEX_LABELS.get(x_var, x_var)} / {LATEX_LABELS.get(x_divisor, x_divisor)}"

    # Perform Y division
    if divide_y and y_divisor != y_var:
        y_col_plot = f"{y_var}_divided"
        plot_data[y_col_plot] = plot_data[y_var] / plot_data[y_divisor].replace(0, np.nan)
        y_var_display = f"{y_var} / {y_divisor}"
        latex_labels_custom[y_col_plot] = f"{LATEX_LABELS.get(y_var, y_var)} / {LATEX_LABELS.get(y_divisor, y_divisor)}"

    # Resolve error columns for each axis
    error_x_col = None
    error_y_col = None
    if show_error_bars:
        # X-axis error
        if x_var == "ttrap" and "ttrap_method_error" in df_filtered.columns:
            plot_data["_error_x"] = df_filtered["ttrap_method_error"]
            error_x_col = "_error_x"
        elif x_var in ERROR_COLUMNS and ERROR_COLUMNS[x_var] in df_filtered.columns:
            plot_data["_error_x"] = df_filtered[ERROR_COLUMNS[x_var]]
            error_x_col = "_error_x"

        # Y-axis error
        if y_var == "ttrap" and "ttrap_method_error" in df_filtered.columns:
            plot_data["_error_y"] = df_filtered["ttrap_method_error"]
            error_y_col = "_error_y"
        elif y_var in ERROR_COLUMNS and ERROR_COLUMNS[y_var] in df_filtered.columns:
            plot_data["_error_y"] = df_filtered[ERROR_COLUMNS[y_var]]
            error_y_col = "_error_y"

        # Scale errors when division is applied (error(a/b) ≈ error(a)/b)
        if error_x_col and divide_x and x_divisor != x_var:
            divisor_vals = plot_data[x_divisor].replace(0, np.nan)
            plot_data["_error_x"] = plot_data["_error_x"] / divisor_vals.abs()
        if error_y_col and divide_y and y_divisor != y_var:
            divisor_vals = plot_data[y_divisor].replace(0, np.nan)
            plot_data["_error_y"] = plot_data["_error_y"] / divisor_vals.abs()

    # Filter out NaN and infinite values from divided columns
    plot_data = plot_data.replace([np.inf, -np.inf], np.nan)
    plot_data = plot_data.dropna(subset=[x_col_plot, y_col_plot])

    # Build hover_data dynamically
    hover_data_dict = {
        "Pe": ":.2f",
        "T": ":.2f",
        "kappa": ":.2f",
        x_col_plot: ":.4f",
        y_col_plot: ":.4f",
        color_col: ":.4f" if color_col else False,
        size_col: ":.4f" if size_col else False,
    }

    # Prepare colorbar label (we'll customize ticks later for log scale)
    if color_var != "None":
        color_label = LATEX_LABELS.get(color_var, color_var)
    else:
        color_label = ""

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
            "color": color_label,
            "size": LATEX_LABELS[size_var] if size_var != "None" else "",
        },
    )

    # Add error bars to scatter traces
    if show_error_bars:
        error_y_dict = None
        error_x_dict = None
        if error_y_col and error_y_col in plot_data.columns:
            err_vals = plot_data[error_y_col].fillna(0).values
            error_y_dict = dict(type="data", array=err_vals, visible=True)
        if error_x_col and error_x_col in plot_data.columns:
            err_vals = plot_data[error_x_col].fillna(0).values
            error_x_dict = dict(type="data", array=err_vals, visible=True)
        fig.update_traces(
            error_y=error_y_dict,
            error_x=error_x_dict,
            selector=dict(mode="markers"),
        )

    # Customize colorbar ticks for log scale (show original values, not log values)
    if use_log_colorbar and color_col is not None:
        # Get the range of log values
        log_min = plot_data["color"].min()
        log_max = plot_data["color"].max()
        # Generate nice tick values in log space
        tick_log_values = np.arange(np.floor(log_min), np.ceil(log_max) + 1)
        # Filter to only include ticks within the data range
        tick_log_values = tick_log_values[(tick_log_values >= log_min - 0.5) & (tick_log_values <= log_max + 0.5)]
        # Convert to original values for display
        tick_original_values = 10 ** tick_log_values
        # Format tick labels
        tick_text = [f"{v:.3g}" for v in tick_original_values]

        fig.update_coloraxes(
            colorbar=dict(
                tickvals=tick_log_values,
                ticktext=tick_text,
                title=LATEX_LABELS.get(color_var, color_var),
            )
        )

    # Dim scatter points if a specific series is highlighted
    if highlight_series and highlight_series != "All (show all)":
        fig.update_traces(marker=dict(opacity=0.5), selector=dict(mode="markers"))

    # Add connecting lines if requested
    if connect_by != "None":
        unique_vals = sorted(plot_data[connect_by].unique())

        highlighted_val = None
        if highlight_series and highlight_series != "All (show all)":
            highlighted_val = float(highlight_series.split("=")[1])

        for val in unique_vals:
            mask = plot_data[connect_by] == val
            subset = plot_data[mask].copy()
            subset = subset.dropna(subset=[x_col_plot, y_col_plot])

            if len(subset) < 2:
                continue

            subset = subset.sort_values(by=[x_col_plot, "T", "kappa"])

            if highlighted_val is None:
                line_width = 2
                line_opacity = 1.0
                line_color = "gray"
            elif abs(val - highlighted_val) < 1e-6:
                line_width = 4
                line_opacity = 1.0
                line_color = "#FF4B4B"
            else:
                line_width = 1
                line_opacity = 0.15
                line_color = "lightgray"

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

        # Recompute exp ttrap if method selector is active
        exp_ttrap_recomputed = {}
        if ttrap_on_axis and (x_var == "ttrap" or y_var == "ttrap"):
            exp_trap_df = load_exp_trapping_data()
            if exp_trap_df is not None and len(exp_trap_df) > 0:
                for temp in exp_plot_data["T_celsius"].unique():
                    temp_int = int(temp)
                    times = np.array(
                        exp_trap_df[exp_trap_df["T_exp"] == temp_int]["time_min"].tolist()
                    )
                    if len(times) > 0:
                        val, err = compute_ttrap_for_scatter(
                            ttrap_method, times, ttrap_n_bins, ttrap_max_cutoff, ttrap_use_log_bins
                        )
                        exp_ttrap_recomputed[temp] = (val, err)

        if len(exp_plot_data) > 0:
            for temp in exp_plot_data["T_celsius"].unique():
                temp_data = exp_plot_data[exp_plot_data["T_celsius"] == temp]

                # Use recomputed values if available
                x_vals = temp_data[x_var].values
                y_vals = temp_data[y_var].values
                exp_err_x = None
                exp_err_y = None

                if temp in exp_ttrap_recomputed:
                    val, err = exp_ttrap_recomputed[temp]
                    if not np.isnan(val):
                        if x_var == "ttrap":
                            x_vals = [val]
                            if show_error_bars and not np.isnan(err):
                                exp_err_x = dict(type="data", array=[err], visible=True)
                        if y_var == "ttrap":
                            y_vals = [val]
                            if show_error_bars and not np.isnan(err):
                                exp_err_y = dict(type="data", array=[err], visible=True)

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers",
                        marker=dict(
                            symbol="star",
                            size=20,
                            color=EXP_TEMP_COLORS.get(temp, "black"),
                            line=dict(color="black", width=2),
                        ),
                        error_x=exp_err_x,
                        error_y=exp_err_y,
                        name=f"Exp T={int(temp)}°C",
                        showlegend=False,
                    )
            )

    # Display plot
    st.plotly_chart(fig, use_container_width=True)

    return x_var, y_var, color_var, size_var
