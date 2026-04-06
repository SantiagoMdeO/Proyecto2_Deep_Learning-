import gradio as gr
import pandas as pd

from data_generation import generate_synthetic_training_data, evaluate_synthetic_data
from data_preprocessing import preprocess_real_data, preprocess_synthetic_data
from credit_models import real_data_credit_model, synthetic_data_credit_model
from visualization import (
    plot_feature_distributions,
    plot_comparative_credit_score_distribution,
    plot_comparison_table,
    plot_comparative_confusion_matrices,
    plot_comparative_credit_score_distribution_by_actual_class,
    plot_evaluation_table,
    get_metrics_df,
)

COLOR_MAP = {
    'Good': '#28B463',
    'Standard': '#F1C40F',
    'Poor': '#E74C3C',
}

LABEL_ORDER = ['Good', 'Standard', 'Poor']
TARGET = 'Credit_Score'

# Load and preprocess real data once at startup
real_train = pd.read_csv('../data/processed/v4/real_train_data.csv')
real_test = pd.read_csv('../data/processed/v4/real_test_data.csv')

X_real_train, y_real_train, X_real_test, y_real_test = preprocess_real_data(
    real_train, real_test, TARGET
)

# Train real-data model once at startup
real_scores, real_classification = real_data_credit_model(
    X_real_train, y_real_train, X_real_test
)

def run_analysis():
    """Generate new synthetic data, train the synthetic model, and return all comparison plots."""
    synthetic_data = generate_synthetic_training_data(n=int(len(X_real_train)/3)) # Same number of samples as real training data
    X_synth_train, y_synth_train = preprocess_synthetic_data(synthetic_data, TARGET)

    categorical_cols = [col for col in X_real_train.columns if X_real_train[col].dtype in ["object", "bool", "uint8"]]
    numeric_cols = [col for col in X_real_train.columns if X_real_train[col].dtype in ["int64", "float64"]]

    fig_feature_dist = plot_feature_distributions(
        X_real_train, X_synth_train
    )

    summary_rows = []
    for cls in LABEL_ORDER:
        real_cls = X_real_train[y_real_train == cls]
        synth_cls = X_synth_train[y_synth_train == cls]
        ks_pass_rate, mean_ks, chi_pass_rate, mean_corr_diff = evaluate_synthetic_data(
            real_cls, synth_cls, categorical_cols, numeric_cols
        )
        summary_rows.append({
            "ks_passed": ks_pass_rate == 1.0,
            "mean_ks_stat": round(mean_ks, 4),
            "chi_passed": chi_pass_rate == 1.0 if chi_pass_rate is not None else None,
            "mean_corr_diff": round(mean_corr_diff, 4),
        })
    summary_df = pd.DataFrame(summary_rows, index=LABEL_ORDER)
    summary = plot_evaluation_table(summary_df)

    synth_scores, synth_classification = synthetic_data_credit_model(
        X_synth_train, y_synth_train, X_real_test
    )

    fig_score_dist = plot_comparative_credit_score_distribution(
        real_scores, synth_scores
    )
    fig_score_by_class = plot_comparative_credit_score_distribution_by_actual_class(
        y_real_test, real_scores, synth_scores,
        color_map=COLOR_MAP,
        label_order=LABEL_ORDER,
    )
    fig_metrics = plot_comparison_table(
        y_real_test, real_classification, synth_classification
    )
    fig_cm = plot_comparative_confusion_matrices(
        y_real_test, real_classification, synth_classification,
        labels=LABEL_ORDER,
    )

    metrics_df = get_metrics_df(y_real_test, real_classification, synth_classification)
    metrics_df = metrics_df.round(4)

    return fig_feature_dist, summary, fig_score_dist, fig_score_by_class, fig_metrics, fig_cm, metrics_df


with gr.Blocks(title="Credit Score Model Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Credit Score Model Dashboard
        Compare a **Real-Data Model** vs a **Synthetic-Data Model** trained with CTGAN-generated data.
        Click the button to regenerate synthetic data and retrain the synthetic model.
        """
    )

    run_btn = gr.Button(
        "Generate New Synthetic Data & Analyze", variant="primary", size="lg"
    )

    gr.Markdown(
    """
    ## Feature Distribution Comparison

    Below are the distributions of the features in the real vs synthetic training datasets.
    """
    )

    with gr.Row():
        plot_feature_dist = gr.Plot(label='')

    gr.Markdown(
    """
    ## Generated Data Quality Summary

    Below is a summary of the data quality evaluation comparing the synthetic training data to the real training data across multiple metrics.
    """
    )

    with gr.Row():
        plot_summary = gr.Plot(label='')

    gr.Markdown(
    """
    ## Credit Models Metrics

    Below are the metrics for the real-data and synthetic-data models.
    """
    )

    with gr.Row():
        plot_metrics = gr.Plot(label='')

    gr.Markdown(
    """
    ## Credit Score Distribution Comparison
    Below are the distributions of the predicted credit scores for the real-data and synthetic-data models.
    """
    )

    with gr.Row():
        plot_score_dist = gr.Plot(label='')

    gr.Markdown(
    """
    ## Credit Score Distribution by Actual Class
    Below are the distributions of the predicted credit scores for each actual class (Good, Standard, Poor) for both models.
    """
    )

    with gr.Row():
        plot_score_by_class = gr.Plot(label='')

    gr.Markdown(
    """
    ## Confusion Matrix Comparison
    Below are the confusion matrices for the real-data and synthetic-data models.
    """
    )

    with gr.Row():
        plot_cm = gr.Plot(label='')

    run_btn.click(
        fn=run_analysis,
        inputs=[],
        outputs=[plot_feature_dist, plot_summary, plot_score_dist, plot_score_by_class, plot_metrics, plot_cm],
    )

demo.launch()
