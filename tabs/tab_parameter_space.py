"""
Tab 2: Parameter Space
Compare how observables vary across parameter space for different temperatures.
"""

import numpy as np
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.constants import LATEX_LABELS, ERROR_COLUMNS, OBSERVABLE_OPTIONS, X_PARAM_OPTIONS


def render_parameter_space_tab(df_filtered, df_exp):
    """Render the Parameter Space tab.

    Args:
        df_filtered: Filtered simulation DataFrame
        df_exp: Experimental data DataFrame (or None)
    """
    st.header("Parameter Space Comparison")
    st.markdown("""
    Compare how observables vary across parameter space for different temperatures.
    """)

    # Display mode selector
    display_mode = st.radio(
        "Display mode",
        ["4 panels", "All temperatures", "Single temperature"],
        horizontal=True
    )

    # Temperature selector (only for single temperature mode)
    selected_temp = 0.1
    if display_mode == "Single temperature":
        selected_temp = st.selectbox(
            "Select temperature",
            [0.05, 0.1, 0.2, 0.3],
            index=1
        )

    # Controls
    col1, col2, col3 = st.columns(3)

    with col1:
        observable = st.selectbox(
            "Observable (Y-axis)",
            OBSERVABLE_OPTIONS,
            index=10  # ttrap
        )

    with col2:
        x_param = st.selectbox(
            "X-axis Parameter",
            X_PARAM_OPTIONS,
            index=8  # tau_decorr_cavity
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
    show_error_bars = False  # Disabled for now

    # Division controls for Tab 2
    with opt2_col2:
        divide_y_tab2 = st.checkbox("Divide Y by observable", value=False, key="divide_y_tab2")

    with opt2_col3:
        y_divisor_tab2 = st.selectbox(
            "Y divisor",
            OBSERVABLE_OPTIONS,
            disabled=not divide_y_tab2,
            key="y_divisor_tab2"
        )

    # Create plot based on display mode
    if len(df_filtered) == 0:
        st.warning("⚠️ No data points match the current filters.")
        return

    # Get unique group values
    group_values = sorted(df_filtered[group_by].dropna().unique())

    # Color mapping based on scheme type
    if color_scheme_type == "Discrete":
        colors = px.colors.qualitative.__dict__[color_palette]
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

    # Determine secondary linking parameter
    secondary_param = None
    secondary_values = []
    if show_secondary_lines:
        secondary_param = secondary_link_param
        secondary_values = sorted(df_filtered[secondary_param].dropna().unique())

    if display_mode == "4 panels":
        _render_4_panels(df_filtered, observable, x_param, group_by, group_values,
                         color_map, use_log_y, show_error_bars, show_secondary_lines,
                         secondary_param, secondary_values, divide_y_tab2, y_divisor_tab2)
    elif display_mode == "All temperatures":
        _render_all_temps(df_filtered, observable, x_param, group_by, group_values,
                          color_map, use_log_y, show_error_bars, divide_y_tab2, y_divisor_tab2)
    else:
        _render_single_temp(df_filtered, selected_temp, observable, x_param, group_by, group_values,
                            color_map, use_log_y, show_error_bars, show_secondary_lines,
                            secondary_param, secondary_values, divide_y_tab2, y_divisor_tab2)


def _render_4_panels(df_filtered, observable, x_param, group_by, group_values,
                     color_map, use_log_y, show_error_bars, show_secondary_lines,
                     secondary_param, secondary_values, divide_y_tab2, y_divisor_tab2):
    """Render 4-panel mode with one subplot per temperature."""
    temps = [0.05, 0.1, 0.2, 0.3]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[f"T = {T}" for T in temps],
        horizontal_spacing=0.06
    )

    obs_label = LATEX_LABELS.get(observable, observable)
    divide_y_tab2_actual = divide_y_tab2

    for col_idx, temp in enumerate(temps, 1):
        df_temp = df_filtered[abs(df_filtered["T"] - temp) < 0.01].copy()

        if len(df_temp) == 0:
            continue

        for group_val in group_values:
            tolerance = 0.001
            df_group = df_temp[np.abs(df_temp[group_by] - group_val) < tolerance].copy()

            if len(df_group) == 0:
                continue

            df_group = df_group[
                np.isfinite(df_group[x_param]) &
                np.isfinite(df_group[observable])
            ].copy()

            if len(df_group) == 0:
                continue

            df_group = df_group.sort_values(x_param).reset_index(drop=True)

            obs_col_plot = observable
            obs_display = observable

            if divide_y_tab2 and y_divisor_tab2 == observable:
                if col_idx == 1 and group_val == group_values[0]:
                    st.warning("⚠️ Cannot divide Y-axis by itself. Division disabled.")
                divide_y_tab2_actual = False
            else:
                divide_y_tab2_actual = divide_y_tab2

            if divide_y_tab2_actual and y_divisor_tab2 != observable:
                if y_divisor_tab2 in df_group.columns:
                    obs_col_plot = f"{observable}_divided"
                    df_group[obs_col_plot] = df_group[observable] / df_group[y_divisor_tab2].replace(0, np.nan)
                    obs_display = f"{observable} / {y_divisor_tab2}"
                    obs_label = f"{LATEX_LABELS.get(observable, observable)} / {LATEX_LABELS.get(y_divisor_tab2, y_divisor_tab2)}"
                    df_group = df_group[np.isfinite(df_group[obs_col_plot])].copy()

            error_x_dict = None
            error_y_dict = None

            if show_error_bars:
                if x_param in ERROR_COLUMNS:
                    error_col_x = ERROR_COLUMNS[x_param]
                    if error_col_x in df_group.columns:
                        error_values_x = df_group[error_col_x].fillna(0).values
                        if np.any(error_values_x > 0):
                            error_x_dict = dict(type='data', array=error_values_x, visible=True)

                if observable in ERROR_COLUMNS:
                    error_col_y = ERROR_COLUMNS[observable]
                    if error_col_y in df_group.columns:
                        error_values_y = df_group[error_col_y].fillna(0).values
                        if np.any(error_values_y > 0):
                            error_y_dict = dict(type='data', array=error_values_y, visible=True)

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
                    showlegend=(col_idx == 1),
                    legendgroup=str(group_val),
                    customdata=np.column_stack((df_group['Pe'], df_group['T'], df_group['kappa'])),
                    hovertemplate='<b>Pe</b>: %{customdata[0]:.2f}<br>' +
                                  '<b>T</b>: %{customdata[1]:.2f}<br>' +
                                  '<b>κ</b>: %{customdata[2]:.2f}<br>' +
                                  f'<b>{x_param}</b>: %{{x:.4f}}<br>' +
                                  f'<b>{obs_display}</b>: %{{y:.4f}}<extra></extra>',
                ),
                row=1, col=col_idx
            )

        if show_secondary_lines:
            for sec_val in secondary_values:
                tolerance = 0.001
                df_sec = df_temp[np.abs(df_temp[secondary_param] - sec_val) < tolerance].copy()

                if len(df_sec) == 0:
                    continue

                df_sec = df_sec[
                    np.isfinite(df_sec[x_param]) &
                    np.isfinite(df_sec[observable])
                ].copy()

                if len(df_sec) == 0:
                    continue

                df_sec = df_sec.sort_values(x_param).reset_index(drop=True)

                obs_col_sec = observable
                if divide_y_tab2_actual and y_divisor_tab2 != observable:
                    if y_divisor_tab2 in df_sec.columns:
                        obs_col_sec = f"{observable}_divided"
                        df_sec[obs_col_sec] = df_sec[observable] / df_sec[y_divisor_tab2].replace(0, np.nan)
                        df_sec = df_sec[np.isfinite(df_sec[obs_col_sec])].copy()

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
            y_label = obs_label if divide_y_tab2_actual else LATEX_LABELS.get(observable, observable)
            fig.update_yaxes(title_text=y_label, row=1, col=col_idx)
        if use_log_y:
            fig.update_yaxes(type="log", row=1, col=col_idx)

    st.plotly_chart(fig, use_container_width=True)


def _render_all_temps(df_filtered, observable, x_param, group_by, group_values,
                      color_map, use_log_y, show_error_bars, divide_y_tab2, y_divisor_tab2):
    """Render all temperatures on the same plot, distinguished by symbols."""
    temps = [0.05, 0.1, 0.2, 0.3]
    # Different symbols for each temperature
    TEMP_SYMBOLS = {0.05: "circle", 0.1: "square", 0.2: "diamond", 0.3: "triangle-up"}

    fig = go.Figure()
    obs_label = LATEX_LABELS.get(observable, observable)
    divide_y_tab2_actual = divide_y_tab2
    obs_display = observable

    # Track which legends we've shown
    shown_temp_legends = set()
    shown_group_legends = set()

    for temp in temps:
        df_temp = df_filtered[abs(df_filtered["T"] - temp) < 0.01].copy()

        if len(df_temp) == 0:
            continue

        for group_val in group_values:
            tolerance = 0.001
            df_group = df_temp[np.abs(df_temp[group_by] - group_val) < tolerance].copy()

            if len(df_group) == 0:
                continue

            df_group = df_group[
                np.isfinite(df_group[x_param]) &
                np.isfinite(df_group[observable])
            ].copy()

            if len(df_group) == 0:
                continue

            df_group = df_group.sort_values(x_param).reset_index(drop=True)

            obs_col_plot = observable

            if divide_y_tab2 and y_divisor_tab2 == observable:
                if temp == temps[0] and group_val == group_values[0]:
                    st.warning("⚠️ Cannot divide Y-axis by itself. Division disabled.")
                divide_y_tab2_actual = False
            else:
                divide_y_tab2_actual = divide_y_tab2

            if divide_y_tab2_actual and y_divisor_tab2 != observable:
                if y_divisor_tab2 in df_group.columns:
                    obs_col_plot = f"{observable}_divided"
                    df_group[obs_col_plot] = df_group[observable] / df_group[y_divisor_tab2].replace(0, np.nan)
                    obs_display = f"{observable} / {y_divisor_tab2}"
                    obs_label = f"{LATEX_LABELS.get(observable, observable)} / {LATEX_LABELS.get(y_divisor_tab2, y_divisor_tab2)}"
                    df_group = df_group[np.isfinite(df_group[obs_col_plot])].copy()

            error_y_dict = None
            if show_error_bars and observable in ERROR_COLUMNS:
                error_col_y = ERROR_COLUMNS[observable]
                if error_col_y in df_group.columns:
                    error_values_y = df_group[error_col_y].fillna(0).values
                    if np.any(error_values_y > 0):
                        error_y_dict = dict(type='data', array=error_values_y, visible=True)

            # Determine legend display
            # Show in legend if first occurrence of this (temp, group) combination
            show_legend = (temp, group_val) not in shown_temp_legends
            shown_temp_legends.add((temp, group_val))

            fig.add_trace(
                go.Scatter(
                    x=df_group[x_param],
                    y=df_group[obs_col_plot],
                    mode="lines+markers",
                    name=f"T={temp}, {group_by}={group_val}",
                    line=dict(color=color_map[group_val], width=2),
                    marker=dict(
                        symbol=TEMP_SYMBOLS[temp],
                        size=10,
                        color=color_map[group_val],
                        line=dict(width=1, color="black")
                    ),
                    error_y=error_y_dict,
                    connectgaps=True,
                    showlegend=show_legend,
                    legendgroup=f"{temp}_{group_val}",
                    customdata=np.column_stack((df_group['Pe'], df_group['T'], df_group['kappa'])),
                    hovertemplate='<b>Pe</b>: %{customdata[0]:.2f}<br>' +
                                  '<b>T</b>: %{customdata[1]:.2f}<br>' +
                                  '<b>κ</b>: %{customdata[2]:.2f}<br>' +
                                  f'<b>{x_param}</b>: %{{x:.4f}}<br>' +
                                  f'<b>{obs_display}</b>: %{{y:.4f}}<extra></extra>',
                )
            )

    fig.update_layout(
        title="All temperatures (symbols: ● T=0.05, ■ T=0.1, ◆ T=0.2, ▲ T=0.3)",
        xaxis_title=x_param,
        yaxis_title=obs_label if divide_y_tab2_actual else LATEX_LABELS.get(observable, observable),
        height=600,
        template="plotly_white",
        font=dict(size=12),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02)
    )

    if use_log_y:
        fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)


def _render_single_temp(df_filtered, selected_temp, observable, x_param, group_by, group_values,
                        color_map, use_log_y, show_error_bars, show_secondary_lines,
                        secondary_param, secondary_values, divide_y_tab2, y_divisor_tab2):
    """Render single-temperature mode."""
    df_temp = df_filtered[abs(df_filtered["T"] - selected_temp) < 0.01].copy()

    if len(df_temp) == 0:
        st.warning(f"⚠️ No data available for T={selected_temp}")
        return

    fig = go.Figure()
    obs_label = LATEX_LABELS.get(observable, observable)
    divide_y_tab2_actual = divide_y_tab2
    obs_display = observable

    for group_val in group_values:
        tolerance = 0.001
        df_group = df_temp[np.abs(df_temp[group_by] - group_val) < tolerance].copy()

        if len(df_group) == 0:
            continue

        df_group = df_group[
            np.isfinite(df_group[x_param]) &
            np.isfinite(df_group[observable])
        ].copy()

        if len(df_group) == 0:
            continue

        df_group = df_group.sort_values(x_param).reset_index(drop=True)

        obs_col_plot = observable

        if divide_y_tab2 and y_divisor_tab2 == observable:
            if group_val == group_values[0]:
                st.warning("⚠️ Cannot divide Y-axis by itself. Division disabled.")
            divide_y_tab2_actual = False
        else:
            divide_y_tab2_actual = divide_y_tab2

        if divide_y_tab2_actual and y_divisor_tab2 != observable:
            if y_divisor_tab2 in df_group.columns:
                obs_col_plot = f"{observable}_divided"
                df_group[obs_col_plot] = df_group[observable] / df_group[y_divisor_tab2].replace(0, np.nan)
                obs_display = f"{observable} / {y_divisor_tab2}"
                obs_label = f"{LATEX_LABELS.get(observable, observable)} / {LATEX_LABELS.get(y_divisor_tab2, y_divisor_tab2)}"
                df_group = df_group[np.isfinite(df_group[obs_col_plot])].copy()

        error_x_dict = None
        error_y_dict = None

        if show_error_bars:
            if x_param in ERROR_COLUMNS:
                error_col_x = ERROR_COLUMNS[x_param]
                if error_col_x in df_group.columns:
                    error_values_x = df_group[error_col_x].fillna(0).values
                    if np.any(error_values_x > 0):
                        error_x_dict = dict(type='data', array=error_values_x, visible=True)

            if observable in ERROR_COLUMNS:
                error_col_y = ERROR_COLUMNS[observable]
                if error_col_y in df_group.columns:
                    error_values_y = df_group[error_col_y].fillna(0).values
                    if np.any(error_values_y > 0):
                        error_y_dict = dict(type='data', array=error_values_y, visible=True)

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

    if show_secondary_lines:
        for sec_val in secondary_values:
            tolerance = 0.001
            df_sec = df_temp[np.abs(df_temp[secondary_param] - sec_val) < tolerance].copy()

            if len(df_sec) == 0:
                continue

            df_sec = df_sec[
                np.isfinite(df_sec[x_param]) &
                np.isfinite(df_sec[observable])
            ].copy()

            if len(df_sec) == 0:
                continue

            df_sec = df_sec.sort_values(x_param).reset_index(drop=True)

            obs_col_sec = observable
            if divide_y_tab2_actual and y_divisor_tab2 != observable:
                if y_divisor_tab2 in df_sec.columns:
                    obs_col_sec = f"{observable}_divided"
                    df_sec[obs_col_sec] = df_sec[observable] / df_sec[y_divisor_tab2].replace(0, np.nan)
                    df_sec = df_sec[np.isfinite(df_sec[obs_col_sec])].copy()

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

    fig.update_layout(
        title=f"T = {selected_temp}",
        xaxis_title=x_param,
        yaxis_title=obs_label if divide_y_tab2_actual else LATEX_LABELS.get(observable, observable),
        height=500,
        template="plotly_white",
        font=dict(size=12),
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left", x=1.02)
    )

    if use_log_y:
        fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)
