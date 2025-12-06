#!/usr/bin/env python3
"""
Interactive Data Visualization for Active Polymer Simulations
=============================================================

Streamlit application for exploring simulation data interactively.

Usage:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from tabs.tab_scatter import render_scatter_tab
from tabs.tab_parameter_space import render_parameter_space_tab
from tabs.tab_trapping import render_trapping_tab
from tabs.tab_tau_params import render_tau_params_tab
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
st.sidebar.subheader("Parameters")

# Pe filter
pe_min, pe_max = float(df["Pe"].min()), float(df["Pe"].max())
pe_range = st.sidebar.slider("Pe range", pe_min, pe_max, (0.30, 1.40), step=0.1)
pe_values = sorted(df["Pe"].unique())
pe_selected = st.sidebar.multiselect(
    "Pick specific Pe (optional)",
    options=pe_values,
    default=[],
    format_func=lambda x: f"{x:.2f}",
)

# T filter
t_min, t_max = float(df["T"].min()), float(df["T"].max())
t_range = st.sidebar.slider("T range", t_min, t_max, (0.09, 0.30), step=0.01)
t_values = sorted(df["T"].unique())
t_selected = st.sidebar.multiselect(
    "Pick specific T (optional)",
    options=t_values,
    default=[],
    format_func=lambda x: f"{x:.2f}",
)

# κ filter
kappa_min, kappa_max = float(df["kappa"].min()), float(df["kappa"].max())
kappa_range = st.sidebar.slider("κ range", kappa_min, kappa_max, (0.50, 2.00), step=0.1)
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
        st.markdown(readme_path.read_text())
    else:
        st.warning("README.md not found")

# =============================================================================
# APPLY FILTERS
# =============================================================================

# Pe filter logic
if len(pe_selected) > 0:
    pe_mask = (
        df["Pe"].isin(pe_selected)
        & (df["Pe"] >= pe_range[0])
        & (df["Pe"] <= pe_range[1])
    )
else:
    pe_mask = (df["Pe"] >= pe_range[0]) & (df["Pe"] <= pe_range[1])

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

if exclude_nan:
    df_filtered = df_filtered.dropna()

st.sidebar.info(f"**{len(df_filtered)}** / {len(df)} points displayed")

# =============================================================================
# MAIN PANEL - TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Scatter Correlations",
    "📊 Parameter Space",
    "📉 Trapping Distributions",
    "⏱️ τ_trap vs Parameters"
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
