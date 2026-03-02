"""
Shared constants for the Active Polymer Data Explorer.
"""

# Error column mapping: maps observable to its error column
ERROR_COLUMNS = {
    'D_long': 'D_long_error',
    'lp_free': 'lp_free_error',
    'lp_conf': 'lp_conf_error',
    'lp_free_individual': 'lp_free_individual_std',
    'tau_decorr_free_full': 'tau_decorr_free_full_error',
    'tau_decorr_free_unit': 'tau_decorr_free_unit_error',
    'ttrap': 'ttrap_std'
}

# Available numeric columns for plotting
NUMERIC_COLS = [
    "Pe",
    "T",
    "kappa",
    "fa",
    "Pe_true",
    "H_free",
    "lp_free",
    "lp_free_individual",
    "tau_decorr_free_full",
    "tau_decorr_free_unit",
    "tau_decorr_cavity_full",
    "tau_decorr_cavity_unit",
    "D_long",
    "H_conf",
    "lp_conf",
    "lp_conf_individual",
    "ttrap",
    "transloc_rate_per_hour",
    "transloc_success_rate",
    # Rotational number (τ_trans = min(τ_15, τ_0))
    "N_rot",
    "tau_trans",
    "plateau_mean",
    "tau_rot",
]

# LaTeX labels mapping (using HTML entities for Greek letters)
LATEX_LABELS = {
    "Pe": "Pe",
    "T": "T",
    "kappa": "κ",
    "fa": "f<sub>a</sub>",
    "Pe_true": "Pe<sub>true</sub> (= f<sub>a</sub>/T)",
    "H_free": "H<sub>free</sub>",
    "lp_free": "ℓ<sub>p</sub> / L (corr, free)",
    "lp_free_individual": "ℓ<sub>p</sub> / L (indiv, free)",
    "tau_decorr_free_full": "τ<sub>decorr, free</sub> full vec (s)",
    "tau_decorr_free_unit": "τ<sub>decorr, free</sub> unit vec (s)",
    "tau_decorr_cavity_full": "τ<sub>decorr, cav</sub> full vec (min)",
    "tau_decorr_cavity_unit": "τ<sub>decorr, cav</sub> unit vec (min)",
    "D_long": "D<sub>long</sub> (mm²/s)",
    "H_conf": "H<sub>conf</sub>",
    "lp_conf": "ℓ<sub>p</sub> / L (corr, conf)",
    "lp_conf_individual": "ℓ<sub>p</sub> / L (indiv, conf)",
    "ttrap": "t<sub>trap</sub> (min)",
    "transloc_rate_per_hour": "Translocation rate (events/h)",
    "transloc_success_rate": "Translocation success rate (%)",
    # Rotational number (τ_trans = min(τ_15, τ_0))
    "N_rot": "N<sub>rot</sub>",
    "tau_trans": "τ<sub>trans</sub> (min)",
    "plateau_mean": "Plateau mean",
    "tau_rot": "τ<sub>rot</sub> (min)",
}

# Observable options for Tab 2
OBSERVABLE_OPTIONS = [
    "fa", "H_free", "lp_free", "lp_free_individual",
    "tau_decorr_free_full", "tau_decorr_free_unit",
    "tau_decorr_cavity_full", "tau_decorr_cavity_unit",
    "D_long",
    "H_conf", "lp_conf", "lp_conf_individual", "ttrap",
    "transloc_rate_per_hour", "transloc_success_rate",
    # Rotational number (τ_trans = min(τ_15, τ_0))
    "N_rot", "tau_trans", "plateau_mean", "tau_rot"
]

# X-axis parameter options for Tab 2
X_PARAM_OPTIONS = [
    "Pe", "kappa", "fa", "Pe_true", "H_free", "H_conf", "lp_free", "lp_conf",
    "tau_decorr_free_full", "tau_decorr_free_unit",
    "tau_decorr_cavity_full", "tau_decorr_cavity_unit",
    "transloc_rate_per_hour", "transloc_success_rate",
    # Rotational number (τ_trans = min(τ_15, τ_0))
    "N_rot", "tau_trans", "plateau_mean", "tau_rot"
]

# Experimental temperature colors
EXP_TEMP_COLORS = {10: "blue", 20: "green", 30: "red"}
