#!/usr/bin/env python3
"""
Interactive Data Visualization for Active Polymer Simulations
=============================================================

Streamlit application for exploring simulation data interactively.

Usage:
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from tabs.tab_scatter import render_scatter_tab
from tabs.tab_parameter_space import render_parameter_space_tab
from tabs.tab_trapping import render_trapping_tab
from tabs.tab_tau_params import render_tau_params_tab
from tabs.tab_violin_comparison import render_violin_comparison_tab
from tabs.tab_rotational import render_rotational_tab
from utils.constants import NUMERIC_COLS

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

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

# =============================================================================
# DATA LOADING
# =============================================================================


@st.cache_data
def load_data(data_source="freespace"):
    """Load the compiled simulation data."""
    if data_source == "freespace":
        csv_path = Path(__file__).parent / "data_freespace.csv"
    else:
        csv_path = Path(__file__).parent / "data_confinement.csv"

    df = pd.read_csv(csv_path, comment="#")
    df["Pe"] = df["Pe"].round(2)
    df["T"] = df["T"].round(2)
    df["kappa"] = df["kappa"].round(2)
    # Note: In CSV files, column "Pe" actually contains fa (active force in LJ units).
    # Historical naming issue in LAMMPS code. We compute the true Péclet number.
    df["fa"] = df["Pe"]  # "Pe" column IS the active force
    df["Pe_true"] = df["fa"] / df["T"]  # True dimensionless Péclet number
    return df


@st.cache_data
def load_exp_data():
    """Load experimental data if available."""
    csv_path = Path(__file__).parent / "data_exp.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header("🗂️ Data Source")

data_source = st.sidebar.radio(
    "Reference base:",
    options=["confinement", "freespace"],
    format_func=lambda x: (
        "Free Space (H_free)" if x == "freespace" else "Confinement (H_conf)"
    ),
    help="Choose which observable is used as the reference base for fuzzy matching.",
)

try:
    df = load_data(data_source)
    df_exp = load_exp_data()

    if data_source == "freespace":
        st.sidebar.success(f"✓ Loaded {len(df)} points (H_free base)")
    else:
        st.sidebar.success(f"✓ Loaded {len(df)} points (H_conf base)")

    if df_exp is not None:
        st.sidebar.info(f"✓ Loaded {len(df_exp)} experimental points")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Parameter filters
st.sidebar.header("📊 Filters")

with st.sidebar.expander("Parameters", expanded=True):
    # Pe/Pe_true toggle
    use_pe_true = st.toggle(
        "Use Pe_true (= fa/T)",
        value=False,
        help="Switch between Pe (active force fa) and Pe_true (dimensionless Péclet = fa/T)"
    )
    pe_col = "Pe_true" if use_pe_true else "Pe"
    pe_label = "Pe_true" if use_pe_true else "Pe"

    # Pe filter
    pe_min, pe_max = float(df[pe_col].min()), float(df[pe_col].max())
    # Adjust default range based on column
    if use_pe_true:
        pe_default = (max(pe_min, 1.0), min(pe_max, 15.0))
    else:
        pe_default = (max(pe_min, 0.30), min(pe_max, 1.40))
    pe_range = st.slider(f"{pe_label} range", pe_min, pe_max, pe_default, step=0.1 if not use_pe_true else 0.5)
    pe_values = sorted(df[pe_col].unique())
    pe_selected = st.multiselect(
        f"Pick specific {pe_label} (optional)",
        options=pe_values,
        default=[],
        format_func=lambda x: f"{x:.2f}",
    )

    # T filter
    t_min, t_max = float(df["T"].min()), float(df["T"].max())
    t_range = st.slider("T range", t_min, t_max, (0.09, 0.31), step=0.01)
    t_values = sorted(df["T"].unique())
    t_selected = st.multiselect(
        "Pick specific T (optional)",
        options=t_values,
        default=[],
        format_func=lambda x: f"{x:.2f}",
    )

    # κ filter
    kappa_min, kappa_max = float(df["kappa"].min()), float(df["kappa"].max())
    kappa_range = st.slider("κ range", kappa_min, kappa_max, (0.50, 2.00), step=0.1)
    kappa_values = sorted(df["kappa"].unique())
    kappa_selected = st.multiselect(
        "Pick specific κ (optional)",
        options=kappa_values,
        default=[],
        format_func=lambda x: f"{x:.2f}",
    )

# Data quality filter for rotational number
st.sidebar.subheader("Data Quality")
r2_min = st.sidebar.number_input("Min R²_trans (visited surface)", min_value=0.0, max_value=20.0, value=15.0, step=1.0)

# Documentation
st.sidebar.markdown("---")
with st.sidebar.expander("📖 Documentation"):
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        st.markdown(readme_path.read_text())
    else:
        st.warning("README.md not found")

# =============================================================================
# APPLY FILTERS
# =============================================================================

# Pe filter logic (using pe_col which is either "Pe" or "Pe_true")
if len(pe_selected) > 0:
    pe_mask = (
        df[pe_col].isin(pe_selected)
        & (df[pe_col] >= pe_range[0])
        & (df[pe_col] <= pe_range[1])
    )
else:
    pe_mask = (df[pe_col] >= pe_range[0]) & (df[pe_col] <= pe_range[1])

# T filter logic
if len(t_selected) > 0:
    t_mask = (
        df["T"].isin(t_selected) & (df["T"] >= t_range[0]) & (df["T"] <= t_range[1])
    )
else:
    t_mask = (df["T"] >= t_range[0]) & (df["T"] <= t_range[1])

# κ filter logic
if len(kappa_selected) > 0:
    kappa_mask = (
        df["kappa"].isin(kappa_selected)
        & (df["kappa"] >= kappa_range[0])
        & (df["kappa"] <= kappa_range[1])
    )
else:
    kappa_mask = (df["kappa"] >= kappa_range[0]) & (df["kappa"] <= kappa_range[1])

# Combine all masks
mask = pe_mask & t_mask & kappa_mask
df_filtered = df[mask].copy()

# Filter rotational data by minimum visited surface
if r2_min > 0 and "R2_trans" in df_filtered.columns:
    mask_low_r2 = df_filtered["R2_trans"] < r2_min
    df_filtered.loc[mask_low_r2, ["N_rot", "tau_trans", "R2_trans", "tau_rot"]] = np.nan

st.sidebar.info(f"**{len(df_filtered)}** / {len(df)} points displayed")

# =============================================================================
# MAIN PANEL - TABS
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Scatter Correlations",
    "📊 Parameter Space",
    "📉 Trapping Distributions",
    "⏱️ τ_trap vs Parameters",
    "💧 violin plots τ_trap",
    "🔄 Rotational Number"
])

with tab1:
    scatter_vars = render_scatter_tab(df_filtered, df_exp)
    x_var, y_var, color_var, size_var = scatter_vars if scatter_vars else ("Pe", "T", "None", "None")

with tab2:
    render_parameter_space_tab(df_filtered, df_exp)

with tab3:
    render_trapping_tab()

with tab4:
    render_tau_params_tab()

with tab5:
    render_violin_comparison_tab()

with tab6:
    render_rotational_tab(df_exp)

# =============================================================================
# DATA TABLE
# =============================================================================

st.markdown("---")
if len(df_filtered) > 0:
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
        display_cols = [col for col in display_cols if not (col in seen or seen.add(col))]
        st.dataframe(df_filtered[display_cols], width="stretch", height=400)

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
