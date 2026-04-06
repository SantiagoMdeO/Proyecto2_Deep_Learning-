import torch
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from scipy import stats
import numpy as np

# Patch torch.load to remap MPS tensors to CPU for environments without Apple Silicon
_original_torch_load = torch.load

def _cpu_map_load(*args, **kwargs):
    kwargs.setdefault('map_location', 'cpu')
    return _original_torch_load(*args, **kwargs)

torch.load = _cpu_map_load


def generate_synthetic_training_data(n=30_000):
    """Generates synthetic training data using pre-trained CTGAN models for each credit score category.

    Args:
        n (int, optional): The number of samples to generate for each category. Defaults to 30_000.
    Returns:
        pd.DataFrame: The generated synthetic training data.
    """
    good_generator = CTGANSynthesizer.load("../models/v4/synth_good.pkl")
    poor_generator = CTGANSynthesizer.load("../models/v4/synth_poor.pkl")
    standard_generator = CTGANSynthesizer.load("../models/v4/synth_standard.pkl")

    synth_good = good_generator.sample(n)
    synth_poor = poor_generator.sample(n)
    synth_standard = standard_generator.sample(n)

    full_data = pd.concat([synth_good, synth_poor, synth_standard], ignore_index=True)
    shuffled_data = full_data.sample(frac=1).reset_index(drop=True)
    return shuffled_data


def evaluate_synthetic_data(real_df, synthetic_df,
                             categorical_cols=None, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = real_df.select_dtypes(include="number").columns.tolist()
    if categorical_cols is None:
        categorical_cols = real_df.select_dtypes(include="object").columns.tolist()

    # ── KS Test ───────────────────────────────────────────────────────────
    ks_results = []
    for col in numeric_cols:
        stat, p_value = stats.ks_2samp(
            real_df[col].dropna(),
            synthetic_df[col].dropna()
        )
        ks_results.append({
            "column" : col,
            "ks_stat": round(stat, 4),
            "p_value": round(p_value, 4),
            "pass"   : p_value > 0.05
        })
    ks_df = pd.DataFrame(ks_results)

    # ── Chi-Square Test ───────────────────────────────────────────────────
    chi_results = []
    for col in categorical_cols:
        real_counts  = real_df[col].value_counts()
        synth_counts = synthetic_df[col].value_counts()
        all_cats     = real_counts.index.union(synth_counts.index)
        real_freq    = real_counts.reindex(all_cats, fill_value=0)
        synth_freq   = synth_counts.reindex(all_cats, fill_value=0)
        n            = real_freq.sum()
        f_exp        = (real_freq  / real_freq.sum())  * n
        f_obs        = (synth_freq / synth_freq.sum()) * n
        stat, p_value = stats.chisquare(f_obs=f_obs, f_exp=f_exp)
        chi_results.append({
            "column"  : col,
            "chi_stat": round(stat, 4),
            "p_value" : round(p_value, 4),
            "pass"    : p_value > 0.05
        })
    chi_df = pd.DataFrame(chi_results)

    # ── Correlation Matrix ────────────────────────────────────────────────
    real_corr      = real_df[numeric_cols].corr()
    synth_corr     = synthetic_df[numeric_cols].corr()
    corr_diff      = (real_corr - synth_corr).abs()
    upper_idx      = np.triu_indices_from(corr_diff.values, k=1)
    mean_corr_diff = corr_diff.values[upper_idx].mean()

    ks_pass_rate  = ks_df["pass"].mean()
    mean_ks       = ks_df["ks_stat"].mean()
    chi_pass_rate = chi_df["pass"].mean() if not chi_df.empty else None

    return ks_pass_rate, mean_ks, chi_pass_rate, mean_corr_diff


def data_evaluation(
        real_list, synthetic_list, class_names=None, categorical_cols=None, numeric_cols=None
):
    if class_names is None:
        class_names = ["good", "poor", "standard"]

    summary_rows = []

    for cls, real_df, syn_df in zip(class_names, real_list, synthetic_list):

        ks_pass_rate, mean_ks, chi_pass_rate, mean_corr_diff = evaluate_synthetic_data(
            real_df          = real_df,
            synthetic_df     = syn_df,
            categorical_cols = categorical_cols,
            numeric_cols     = numeric_cols
        )

        summary_rows.append({
            "class"         : cls,
            "ks_pass_rate"  : round(ks_pass_rate, 4),
            "mean_ks_stat"  : round(mean_ks, 4),
            "chi_pass_rate" : round(chi_pass_rate, 4) if chi_pass_rate is not None else None,
            "mean_corr_diff": round(mean_corr_diff, 4)
        })

    return pd.DataFrame(summary_rows).set_index("class")