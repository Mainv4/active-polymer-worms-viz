"""
Export utilities for generating publication-quality matplotlib figures
and downloadable ZIP archives with data and reproducible scripts.
"""

import io
import json
import zipfile
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


# Color scheme matching the Streamlit app
TEMP_COLORS = {10: "#3498db", 20: "#2ecc71", 30: "#e74c3c"}
TEMP_MARKERS = {10: "s", 20: "o", 30: "^"}  # square, circle, triangle


def setup_latex_style(use_latex=True):
    """Configure matplotlib for publication-quality LaTeX style.

    Args:
        use_latex: If True, use LaTeX rendering (requires texlive installation)
    """
    plt.rcParams.update({
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.figsize": (8, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    })


def fit_exponential_log_space(times, bins):
    """Fit exponential decay using linear regression in log space.

    Args:
        times: Array of trapping times
        bins: Bin edges for histogram

    Returns:
        (tau, tau_error, A) or (None, None, None) if fit fails
    """
    counts, bin_edges = np.histogram(times, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mask = (counts > 0) & (bin_centers > 0)
    if np.sum(mask) < 3:
        return None, None, None

    log_counts = np.log(counts[mask])
    fit_centers = bin_centers[mask]

    try:
        def linear_func(x, a, b):
            return a * x + b

        popt, pcov = optimize.curve_fit(
            linear_func, fit_centers, log_counts, p0=[-1, 0]
        )

        slope, intercept = popt
        tau = -1.0 / slope
        A = np.exp(intercept)

        if pcov[0, 0] > 0:
            tau_error = np.sqrt(pcov[0, 0]) / (slope ** 2)
        else:
            tau_error = 0

        return tau, tau_error, A
    except Exception:
        return None, None, None


def create_trapping_histogram_mpl(times_dict, bins, normalize=True,
                                   fit_results=None, use_log_y=True,
                                   title=None):
    """Create publication-quality matplotlib figure for trapping time distributions.

    Args:
        times_dict: {temp: np.array} data per temperature
        bins: bin edges (np.array)
        normalize: density histogram if True
        fit_results: {temp: {'tau': ..., 'tau_error': ..., 'A': ...}} or None to compute
        use_log_y: log scale on Y axis
        title: optional figure title

    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute fit results if not provided
    if fit_results is None:
        fit_results = {}
        for temp, times in times_dict.items():
            tau, tau_error, A = fit_exponential_log_space(times, bins)
            if tau is not None:
                fit_results[temp] = {'tau': tau, 'tau_error': tau_error, 'A': A}

    # Plot each temperature
    for temp, times in sorted(times_dict.items()):
        if len(times) == 0:
            continue

        # Compute histogram
        counts, bin_edges = np.histogram(times, bins=bins, density=normalize)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Filter zero counts
        mask = counts > 0
        x_plot = bin_centers[mask]
        y_plot = counts[mask]

        # Scatter points
        if plt.rcParams["text.usetex"]:
            label = rf"$T = {temp}^\circ$C"
        else:
            label = f"T = {temp}°C"
        ax.scatter(x_plot, y_plot, s=100, c=TEMP_COLORS[temp],
                   marker=TEMP_MARKERS[temp], label=label, alpha=0.8,
                   edgecolors='black', linewidths=0.5, zorder=3)

        # Exponential fit line
        if temp in fit_results:
            tau = fit_results[temp]['tau']
            A = fit_results[temp]['A']
            x_fit = np.linspace(0.01, bins[-1], 200)
            y_fit = A * np.exp(-x_fit / tau)

            tau_label = rf"$\tau = {tau:.2f}$ min" if plt.rcParams["text.usetex"] else f"τ = {tau:.2f} min"
            ax.plot(x_fit, y_fit, '--', color=TEMP_COLORS[temp], linewidth=2,
                    label=tau_label, zorder=2)

    # Axes configuration
    if plt.rcParams["text.usetex"]:
        ax.set_xlabel(r"$\tau_{tr}$ (min)")
        ax.set_ylabel(r"$P(\tau_{tr})$" if normalize else "Count")
    else:
        ax.set_xlabel("τ_tr (min)")
        ax.set_ylabel("P(τ_tr)" if normalize else "Count")

    if use_log_y:
        ax.set_yscale('log')

    ax.set_xlim(0, bins[-1])

    if title:
        ax.set_title(title)

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig, ax


def create_by_length_figure_mpl(times_by_length, bins, normalize=True,
                                 use_log_y=True, temp=None):
    """Create matplotlib figure for distributions grouped by worm length.

    Args:
        times_by_length: {label: np.array} data per length class
        bins: bin edges
        normalize: density histogram
        use_log_y: log scale Y
        temp: temperature for title

    Returns:
        fig, ax
    """
    # Length class colors (using ASCII keys)
    LENGTH_COLORS = {
        "lc: 10-21 mm": "#a8e6cf",
        "lc: 23-27 mm": "#64b5f6",
        "lc: 27-34 mm": "#1a237e",
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for label_key, times in times_by_length.items():
        if len(times) < 5:
            continue

        color = LENGTH_COLORS.get(label_key, "#888888")

        # Convert label to LaTeX format if needed
        if plt.rcParams["text.usetex"]:
            # "lc: 10-21 mm" -> "$\ell_c$: 10-21 mm"
            label = label_key.replace("lc:", r"$\ell_c$:")
        else:
            label = label_key

        # Histogram
        counts, bin_edges = np.histogram(times, bins=bins, density=normalize)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = counts > 0

        # Scatter
        ax.scatter(bin_centers[mask], counts[mask], s=100, c=color,
                   marker='o', label=label, alpha=0.8,
                   edgecolors='black', linewidths=0.5, zorder=3)

        # Fit
        tau, tau_error, A = fit_exponential_log_space(times, bins)
        if tau is not None:
            x_fit = np.linspace(0.01, bins[-1], 200)
            y_fit = A * np.exp(-x_fit / tau)
            tau_label = rf"$\tau = {tau:.2f}$ min" if plt.rcParams["text.usetex"] else f"τ = {tau:.2f} min"
            ax.plot(x_fit, y_fit, '--', color=color, linewidth=2,
                    label=tau_label, zorder=2)

    if plt.rcParams["text.usetex"]:
        ax.set_xlabel(r"$\tau_{tr}$ (min)")
        ax.set_ylabel(r"$P(\tau_{tr})$" if normalize else "Count")
    else:
        ax.set_xlabel("τ_tr (min)")
        ax.set_ylabel("P(τ_tr)" if normalize else "Count")

    if use_log_y:
        ax.set_yscale('log')

    ax.set_xlim(0, bins[-1])

    if temp:
        if plt.rcParams["text.usetex"]:
            title = rf"$T = {temp}^\circ$C - Distribution by worm length"
        else:
            title = f"T = {temp}°C - Distribution by worm length"
        ax.set_title(title)

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig, ax


def generate_script_template(config, data_files, figure_type="histogram"):
    """Generate a minimal Python script to reproduce the figure.

    Args:
        config: dict with parameters (n_bins, max_cutoff, use_log_bins, etc.)
        data_files: list of data file names
        figure_type: "histogram" or "by_length"

    Returns:
        str: Python script content
    """
    script = '''#!/usr/bin/env python3
"""
Reproduce trapping time distribution figure.
Generated by active-polymer-worms-viz on {date}
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# ============================================================================
# Configuration
# ============================================================================

# LaTeX style (set to False if LaTeX not installed)
USE_LATEX = True

plt.rcParams.update({{
    "text.usetex": USE_LATEX,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.figsize": (8, 6),
    "savefig.dpi": 300,
    "axes.linewidth": 1.2,
}})

# Parameters used for the figure
N_BINS = {n_bins}
MAX_CUTOFF = {max_cutoff}
USE_LOG_BINS = {use_log_bins}
NORMALIZE = {normalize}
USE_LOG_Y = {use_log_y}

# Colors
TEMP_COLORS = {{10: "#3498db", 20: "#2ecc71", 30: "#e74c3c"}}
TEMP_MARKERS = {{10: "s", 20: "o", 30: "^"}}

# ============================================================================
# Helper functions
# ============================================================================

def make_bins(max_cutoff, n_bins, use_log_bins):
    if use_log_bins:
        return np.logspace(np.log10(0.05), np.log10(max_cutoff), n_bins + 1)
    return np.linspace(0, max_cutoff, n_bins + 1)


def fit_exponential_log_space(times, bins):
    counts, bin_edges = np.histogram(times, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = (counts > 0) & (bin_centers > 0)
    if np.sum(mask) < 3:
        return None, None, None
    log_counts = np.log(counts[mask])
    fit_centers = bin_centers[mask]
    try:
        popt, pcov = optimize.curve_fit(
            lambda x, a, b: a * x + b, fit_centers, log_counts, p0=[-1, 0]
        )
        slope, intercept = popt
        tau = -1.0 / slope
        A = np.exp(intercept)
        tau_error = np.sqrt(pcov[0, 0]) / (slope ** 2) if pcov[0, 0] > 0 else 0
        return tau, tau_error, A
    except Exception:
        return None, None, None

# ============================================================================
# Load data
# ============================================================================

{load_data_section}

# ============================================================================
# Create figure
# ============================================================================

bins = make_bins(MAX_CUTOFF, N_BINS, USE_LOG_BINS)
fig, ax = plt.subplots(figsize=(8, 6))

{plot_section}

# Axes
if USE_LATEX:
    ax.set_xlabel(r"$\\tau_{{tr}}$ (min)")
    ax.set_ylabel(r"$P(\\tau_{{tr}})$" if NORMALIZE else "Count")
else:
    ax.set_xlabel("τ_tr (min)")
    ax.set_ylabel("P(τ_tr)" if NORMALIZE else "Count")

if USE_LOG_Y:
    ax.set_yscale('log')

ax.set_xlim(0, MAX_CUTOFF)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save and show
plt.savefig("figure_reproduced.pdf", bbox_inches="tight")
print("Figure saved to figure_reproduced.pdf")
plt.show()
'''

    # Generate load data section
    if figure_type == "histogram":
        load_lines = []
        for fname in data_files:
            var_name = fname.replace('.npy', '').replace('times_', 'times_')
            load_lines.append(f'{var_name} = np.load("data/{fname}")')
        load_data_section = '\n'.join(load_lines)

        # Generate plot section
        plot_lines = ['times_dict = {']
        for fname in data_files:
            temp = fname.replace('times_', '').replace('C.npy', '')
            var_name = fname.replace('.npy', '')
            plot_lines.append(f'    {temp}: {var_name},')
        plot_lines.append('}')
        plot_lines.append('')
        plot_lines.append('for temp, times in sorted(times_dict.items()):')
        plot_lines.append('    counts, bin_edges = np.histogram(times, bins=bins, density=NORMALIZE)')
        plot_lines.append('    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2')
        plot_lines.append('    mask = counts > 0')
        plot_lines.append('')
        plot_lines.append('    label = rf"$T = {temp}^\\circ$C" if USE_LATEX else f"T = {temp}C"')
        plot_lines.append('    ax.scatter(bin_centers[mask], counts[mask], s=100, c=TEMP_COLORS[temp],')
        plot_lines.append('               marker=TEMP_MARKERS[temp], label=label, alpha=0.8,')
        plot_lines.append('               edgecolors="black", linewidths=0.5, zorder=3)')
        plot_lines.append('')
        plot_lines.append('    tau, tau_error, A = fit_exponential_log_space(times, bins)')
        plot_lines.append('    if tau is not None:')
        plot_lines.append('        x_fit = np.linspace(0.01, MAX_CUTOFF, 200)')
        plot_lines.append('        y_fit = A * np.exp(-x_fit / tau)')
        plot_lines.append('        tau_label = rf"$\\tau = {tau:.2f}$ min" if USE_LATEX else f"τ = {tau:.2f} min"')
        plot_lines.append('        ax.plot(x_fit, y_fit, "--", color=TEMP_COLORS[temp], linewidth=2,')
        plot_lines.append('                label=tau_label, zorder=2)')
        plot_section = '\n'.join(plot_lines)
    else:
        load_data_section = "# Load length-grouped data\n# times_by_length = {...}"
        plot_section = "# Plot by length groups\n# ..."

    return script.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        n_bins=config.get('n_bins', 30),
        max_cutoff=config.get('max_cutoff', 10.0),
        use_log_bins=config.get('use_log_bins', False),
        normalize=config.get('normalize', True),
        use_log_y=config.get('use_log_y', True),
        load_data_section=load_data_section,
        plot_section=plot_section,
    )


def create_export_zip(fig, times_dict, config, fig_format='pdf'):
    """Create ZIP archive with figure, data, and script.

    Args:
        fig: matplotlib figure
        times_dict: {temp: np.array} or {label: np.array}
        config: dict with parameters
        fig_format: 'pdf', 'png', or 'svg'

    Returns:
        bytes: ZIP file content
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Save figure
        fig_buffer = io.BytesIO()
        fig.savefig(fig_buffer, format=fig_format, bbox_inches='tight',
                    dpi=300 if fig_format == 'png' else None)
        fig_buffer.seek(0)
        zf.writestr(f'figure.{fig_format}', fig_buffer.getvalue())

        # Save data files
        data_files = []
        for key, times in times_dict.items():
            if isinstance(key, int):
                fname = f'times_{key}C.npy'
            else:
                fname = f'times_{key.replace(" ", "_").replace(":", "")}.npy'
            data_files.append(fname)

            npy_buffer = io.BytesIO()
            np.save(npy_buffer, times)
            npy_buffer.seek(0)
            zf.writestr(f'data/{fname}', npy_buffer.getvalue())

        # Save config
        zf.writestr('data/config.json', json.dumps(config, indent=2))

        # Generate and save script
        script = generate_script_template(config, data_files)
        zf.writestr('reproduce_figure.py', script)

    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# VIOLIN PLOT EXPORT FUNCTIONS
# =============================================================================


def create_violin_figure_mpl(grouped_data_by_temp, layout="stacked",
                              selected_temp=None, sim_data=None,
                              sim_params=None, length_threshold=2.0):
    """Create matplotlib violin plot figure.

    Args:
        grouped_data_by_temp: {temp: [(mean_length, times_list), ...]}
        layout: "stacked" (3 panels) or "single"
        selected_temp: temperature for single layout (10, 20, or 30)
        sim_data: optional simulation data {temp: {N: times_array}}
        sim_params: optional simulation parameters {temp: (Pe, T, kappa)}
        length_threshold: length grouping threshold (for annotation)

    Returns:
        fig, axes (list or single ax)
    """
    from matplotlib.lines import Line2D

    # Determine temperatures to plot
    if layout == "single" and selected_temp is not None:
        temps_to_plot = [selected_temp]
        nrows = 1
        figsize = (8, 4)
    else:
        temps_to_plot = sorted(grouped_data_by_temp.keys())
        nrows = len(temps_to_plot)
        figsize = (8, 2.5 * nrows)

    fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0)

    # N to length mapping for simulation data
    n_to_length = {40: 20, 50: 25, 60: 30}

    # Check if any simulation is enabled
    any_sim_enabled = sim_data is not None and any(
        temp in sim_data and sim_data[temp] for temp in temps_to_plot
    )

    for idx, temp in enumerate(temps_to_plot):
        ax = axes[idx]

        grouped = grouped_data_by_temp.get(temp, [])

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
                pc.set_facecolor(TEMP_COLORS[temp])
                pc.set_alpha(0.5)
                pc.set_edgecolor("black")
                pc.set_linewidth(1)

            parts["cmeans"].set_edgecolor("darkred")
            parts["cmeans"].set_linewidth(2)
            parts["cmedians"].set_edgecolor("darkblue")
            parts["cmedians"].set_linewidth(2)

        # Overlay simulation violins if available
        if sim_data is not None and temp in sim_data and sim_data[temp]:
            sim_positions = []
            sim_datasets = []

            for N, x_pos in n_to_length.items():
                if N in sim_data[temp] and len(sim_data[temp][N]) >= 3:
                    sim_positions.append(x_pos + 2.5)
                    sim_datasets.append(sim_data[temp][N].tolist())

            if len(sim_datasets) > 0:
                parts_sim = ax.violinplot(
                    sim_datasets,
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
        if idx == len(temps_to_plot) - 1:
            if plt.rcParams["text.usetex"]:
                ax.set_xlabel(r"$\ell_c$ (mm)", fontsize=16)
            else:
                ax.set_xlabel("l_c (mm)", fontsize=16)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        if plt.rcParams["text.usetex"]:
            ax.set_ylabel(r"$\tau_{\rm trap}$ (min)", fontsize=16)
        else:
            ax.set_ylabel("τ_trap (min)", fontsize=16)

        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(-0.5, 10.5)

        # Legend only on top panel
        if idx == 0:
            legend_elements = [
                Line2D([0], [0], color="darkred", linewidth=2, label="Mean"),
                Line2D([0], [0], color="darkblue", linewidth=2, label="Median"),
            ]
            if any_sim_enabled:
                legend_elements.append(
                    Line2D(
                        [0], [0],
                        marker="s", color="w",
                        markerfacecolor="orange", markersize=10,
                        label="Simulation",
                    )
                )
            ax.legend(handles=legend_elements, loc="upper left", fontsize=11)

        # Build annotation text with parameters
        if plt.rcParams["text.usetex"]:
            annotation_text = f"$T={temp}^\\circ$C"
        else:
            annotation_text = f"T={temp}C"

        if sim_params is not None and temp in sim_params:
            Pe, T_sim, kappa = sim_params[temp]
            if plt.rcParams["text.usetex"]:
                annotation_text += f"\nPe={Pe:.2f}, T={T_sim:.2f}, $\\kappa$={kappa:.2f}"
            else:
                annotation_text += f"\nPe={Pe:.2f}, T={T_sim:.2f}, kappa={kappa:.2f}"

        # Temperature annotation box
        ax.text(
            0.95, 0.95, annotation_text,
            transform=ax.transAxes,
            fontsize=11,
            va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", linewidth=1.5),
        )

    axes[-1].set_xlim(12, 40)

    plt.tight_layout()
    return fig, axes


def generate_violin_script_template(config, data_files, sim_files=None):
    """Generate a minimal Python script to reproduce the violin figure.

    Args:
        config: dict with parameters (layout, temperatures, etc.)
        data_files: list of experimental data file names
        sim_files: list of simulation data file names (optional)

    Returns:
        str: Python script content
    """
    script = '''#!/usr/bin/env python3
"""
Reproduce violin plot figure for trapping time distributions.
Generated by active-polymer-worms-viz on {date}
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================================
# Configuration
# ============================================================================

# LaTeX style (set to False if LaTeX not installed)
USE_LATEX = True

plt.rcParams.update({{
    "text.usetex": USE_LATEX,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "savefig.dpi": 300,
    "axes.linewidth": 1.2,
}})

# Parameters
LAYOUT = "{layout}"  # "stacked" or "single"
TEMPERATURES = {temperatures}
LENGTH_THRESHOLD = {length_threshold}

# Colors
TEMP_COLORS = {{10: "#3498db", 20: "#2ecc71", 30: "#e74c3c"}}

# ============================================================================
# Helper: group data by length
# ============================================================================

def group_worms_by_length(lengths, times, threshold=2.0, min_events=3):
    """Group trapping events by worm length."""
    if len(lengths) == 0:
        return []

    sorted_idx = np.argsort(lengths)
    lengths_sorted = lengths[sorted_idx]
    times_sorted = times[sorted_idx]

    grouped = []
    i = 0
    while i < len(lengths_sorted):
        current_length = lengths_sorted[i]
        group_lengths = [current_length]
        group_times = [times_sorted[i]]

        j = i + 1
        while j < len(lengths_sorted):
            if abs(lengths_sorted[j] - current_length) <= threshold:
                group_lengths.append(lengths_sorted[j])
                group_times.append(times_sorted[j])
                j += 1
            else:
                break

        if len(group_times) >= min_events:
            grouped.append((float(np.mean(group_lengths)), group_times))
        i = j

    return grouped

# ============================================================================
# Load data
# ============================================================================

{load_data_section}

# ============================================================================
# Create figure
# ============================================================================

temps_to_plot = TEMPERATURES if LAYOUT == "stacked" else [TEMPERATURES[0]]
nrows = len(temps_to_plot)
figsize = (8, 2.5 * nrows)

fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True, squeeze=False)
axes = axes.flatten()
fig.subplots_adjust(hspace=0)

for idx, temp in enumerate(temps_to_plot):
    ax = axes[idx]

    grouped = grouped_data[temp]

    if len(grouped) > 0:
        positions = [g[0] for g in grouped]
        data_list = [g[1] for g in grouped]

        parts = ax.violinplot(
            data_list, positions=positions, widths=1.5,
            showmeans=True, showmedians=True, showextrema=False,
        )

        for pc in parts["bodies"]:
            pc.set_facecolor(TEMP_COLORS[temp])
            pc.set_alpha(0.5)
            pc.set_edgecolor("black")
            pc.set_linewidth(1)

        parts["cmeans"].set_edgecolor("darkred")
        parts["cmeans"].set_linewidth(2)
        parts["cmedians"].set_edgecolor("darkblue")
        parts["cmedians"].set_linewidth(2)

    # X-label only on bottom
    if idx == len(temps_to_plot) - 1:
        ax.set_xlabel(r"$\\ell_c$ (mm)" if USE_LATEX else "l_c (mm)", fontsize=16)
    else:
        ax.tick_params(labelbottom=False)

    ax.set_ylabel(r"$\\tau_{{\\rm trap}}$ (min)" if USE_LATEX else "τ_trap (min)", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(-0.5, 10.5)

    # Legend on top panel
    if idx == 0:
        legend_elements = [
            Line2D([0], [0], color="darkred", linewidth=2, label="Mean"),
            Line2D([0], [0], color="darkblue", linewidth=2, label="Median"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=11)

    # Temperature annotation
    if USE_LATEX:
        annotation = f"$T={{temp}}^\\circ$C"
    else:
        annotation = f"T={{temp}}C"
    ax.text(0.95, 0.95, annotation, transform=ax.transAxes, fontsize=11,
            va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", linewidth=1.5))

axes[-1].set_xlim(12, 40)
plt.tight_layout()

# Save and show
plt.savefig("violin_reproduced.pdf", bbox_inches="tight")
print("Figure saved to violin_reproduced.pdf")
plt.show()
'''

    # Generate load data section
    load_lines = []
    for fname in data_files:
        # Extract temp from filename: exp_times_10C.npy -> 10
        temp_str = fname.replace('exp_times_', '').replace('C.npy', '')
        load_lines.append(f'times_{temp_str}C = np.load("data/{fname.replace("times", "times")}")')
        load_lines.append(f'lengths_{temp_str}C = np.load("data/{fname.replace("times", "lengths")}")')

    load_lines.append('')
    load_lines.append('# Group data by length')
    load_lines.append('grouped_data = {}')
    for fname in data_files:
        temp_str = fname.replace('exp_times_', '').replace('C.npy', '')
        load_lines.append(f'grouped_data[{temp_str}] = group_worms_by_length(lengths_{temp_str}C, times_{temp_str}C, LENGTH_THRESHOLD)')

    load_data_section = '\n'.join(load_lines)

    return script.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        layout=config.get('layout', 'stacked'),
        temperatures=config.get('temperatures', [10, 20, 30]),
        length_threshold=config.get('length_threshold', 2.0),
        load_data_section=load_data_section,
    )


def create_violin_export_zip(fig, exp_data_dict, config, sim_data_dict=None,
                              sim_params=None, fig_format='pdf'):
    """Create ZIP archive with violin figure, data, and script.

    Args:
        fig: matplotlib figure
        exp_data_dict: {temp: {'times': np.array, 'lengths': np.array}}
        config: dict with parameters
        sim_data_dict: optional {temp: {N: times_array}}
        sim_params: optional {temp: (Pe, T, kappa)}
        fig_format: 'pdf', 'png', or 'svg'

    Returns:
        bytes: ZIP file content
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Save figure
        fig_buffer = io.BytesIO()
        fig.savefig(fig_buffer, format=fig_format, bbox_inches='tight',
                    dpi=300 if fig_format == 'png' else None)
        fig_buffer.seek(0)
        zf.writestr(f'figure.{fig_format}', fig_buffer.getvalue())

        # Save experimental data files
        data_files = []
        for temp, data in exp_data_dict.items():
            # Times
            times_fname = f'exp_times_{temp}C.npy'
            data_files.append(times_fname)
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, data['times'])
            npy_buffer.seek(0)
            zf.writestr(f'data/{times_fname}', npy_buffer.getvalue())

            # Lengths
            lengths_fname = f'exp_lengths_{temp}C.npy'
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, data['lengths'])
            npy_buffer.seek(0)
            zf.writestr(f'data/{lengths_fname}', npy_buffer.getvalue())

        # Save simulation data if present
        sim_files = []
        if sim_data_dict is not None and sim_params is not None:
            for temp, n_data in sim_data_dict.items():
                if temp in sim_params:
                    Pe, T_sim, kappa = sim_params[temp]
                    for N, times in n_data.items():
                        sim_fname = f'sim_N{N}_T{temp}C_Pe{Pe:.2f}_k{kappa:.2f}.npy'
                        sim_files.append(sim_fname)
                        npy_buffer = io.BytesIO()
                        np.save(npy_buffer, times)
                        npy_buffer.seek(0)
                        zf.writestr(f'data/{sim_fname}', npy_buffer.getvalue())

        # Save config
        zf.writestr('data/config.json', json.dumps(config, indent=2))

        # Generate and save script
        script = generate_violin_script_template(config, data_files, sim_files)
        zf.writestr('reproduce_figure.py', script)

    buffer.seek(0)
    return buffer.getvalue()
