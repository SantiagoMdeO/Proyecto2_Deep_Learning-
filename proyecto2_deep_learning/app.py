import gradio as gr
import pandas as pd

from data_generation import generate_synthetic_training_data
from data_preprocessing import preprocess_real_data, preprocess_synthetic_data
from credit_models import real_data_credit_model, synthetic_data_credit_model
from visualization import (
    plot_comparative_credit_score_distribution,
    plot_comparison_table,
    plot_comparative_confusion_matrices,
    plot_comparative_credit_score_distribution_by_actual_class,
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
    synthetic_data = generate_synthetic_training_data(n=30_000)
    X_synth_train, y_synth_train = preprocess_synthetic_data(synthetic_data, TARGET)

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

    return fig_score_dist, fig_score_by_class, fig_metrics, fig_cm, metrics_df


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

    with gr.Row():
        plot_metrics = gr.Plot(label='')

    with gr.Row():
        plot_score_dist = gr.Plot(label='')

    with gr.Row():
        plot_score_by_class = gr.Plot(label='')

    with gr.Row():
        plot_cm = gr.Plot(label='')

    run_btn.click(
        fn=run_analysis,
        inputs=[],
        outputs=[plot_score_dist, plot_score_by_class, plot_metrics, plot_cm],
    )

demo.launch()
