import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def get_metrics_df(y_true, y_real_pred, y_synth_pred):
    metrics = {
        'Model': ['Real Data Model', 'Synthetic Data Model'],
        'Accuracy': [
            accuracy_score(y_true, y_real_pred),
            accuracy_score(y_true, y_synth_pred)
        ],
        'Precision': [
            precision_score(y_true, y_real_pred, average='weighted'),
            precision_score(y_true, y_synth_pred, average='weighted')
        ],
        'Recall': [
            recall_score(y_true, y_real_pred, average='weighted'),
            recall_score(y_true, y_synth_pred, average='weighted')
        ],
        'F1-Score': [
            f1_score(y_true, y_real_pred, average='weighted'),
            f1_score(y_true, y_synth_pred, average='weighted')
        ]
    }
    return pd.DataFrame(metrics)


def plot_comparative_credit_score_distribution(
    real_scores,
    synth_scores,
    bins=50,
    kde=True,
    title='Comparative Credit Score Distribution: Real vs Synthetic Models'
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    sns.histplot(
        real_scores,
        bins=bins,
        kde=kde,
        color='#1F77B4',
        edgecolor='white',
        alpha=0.2,
        ax=axes[0]
    )
    axes[0].set_title('Real-Data Model Score Distribution')
    axes[0].set_xlabel('Predicted Credit Score')
    axes[0].set_ylabel('Frequency')

    sns.histplot(
        synth_scores,
        bins=bins,
        kde=kde,
        color='#1F77B4',
        edgecolor='white',
        alpha=0.2,
        ax=axes[1]
    )
    axes[1].set_title('Synthetic-Data Model Score Distribution')
    axes[1].set_xlabel('Predicted Credit Score')
    axes[1].set_ylabel('Frequency')

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_comparison_table(
        y_true, y_real_pred, y_synth_pred, 
        title='Model Comparison: Real Data vs Synthetic Data'
):
    metrics_df = get_metrics_df(y_true, y_real_pred, y_synth_pred)
    display_df = metrics_df.copy().round(4).set_index('Model')

    fig, ax = plt.subplots(figsize=(18, 2))
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)

    table = ax.table(
        cellText=display_df.values,
        rowLabels=display_df.index,
        colLabels=display_df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1.2, 1.9)

    for j in range(len(display_df.columns)):
        table[(0, j)].set_facecolor('#1F77B4')
        table[(0, j)].set_text_props(color='white', weight='bold')

    for i in range(1, len(display_df.index) + 1):
        bg = '#F7F9FB' if i % 2 else '#EAF2FA'
        table[(i, -1)].set_text_props(weight='bold')
        table[(i, -1)].set_facecolor(bg)
        for j in range(len(display_df.columns)):
            table[(i, j)].set_facecolor(bg)

    plt.tight_layout()
    return fig


def plot_comparative_confusion_matrices(
    y_true,
    y_pred_real,
    y_pred_synth,
    labels=None,
    normalize=False,
    cmap='Blues'
):
    cm_real = confusion_matrix(y_true, y_pred_real, labels=labels)
    cm_synth = confusion_matrix(y_true, y_pred_synth, labels=labels)

    if normalize:
        cm_real_plot = cm_real.astype(float) / cm_real.sum(axis=1, keepdims=True)
        cm_synth_plot = cm_synth.astype(float) / cm_synth.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        cm_real_plot = cm_real
        cm_synth_plot = cm_synth
        fmt = 'd'

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        cm_real_plot, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=labels, yticklabels=labels, ax=axes[0]
    )
    axes[0].set_title(f"Real Data Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(
        cm_synth_plot, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=labels, yticklabels=labels, ax=axes[1]
    )
    axes[1].set_title(f"Synthetic Data Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    fig.suptitle("Comparative Confusion Matrices: Real vs Synthetic Models", fontsize=16, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_comparative_credit_score_distribution_by_actual_class(
    y_true,
    real_scores,
    synth_scores,
    color_map,
    label_order=None,
    bins=50,
):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    y_true_arr = pd.Series(y_true).values

    for label in label_order:
        mask = (y_true_arr == label)

        sns.histplot(
            real_scores[mask],
            bins=bins,
            stat='count',
            element='step',
            fill=True,
            alpha=0.2,
            color=color_map.get(label, None),
            label=label,
            ax=ax_left
        )

        sns.histplot(
            synth_scores[mask],
            bins=bins,
            stat='count',
            element='step',
            fill=True,
            alpha=0.2,
            color=color_map.get(label, None),
            label=label,
            ax=ax_right
        )

    ax_left.set_title('Real-Data Model: Actual Class Distribution')
    ax_left.set_xlabel('Predicted Credit Score')
    ax_left.set_ylabel('Frequency')
    ax_left.legend(title='Actual Class')

    ax_right.set_title('Synthetic-Data Model: Actual Class Distribution')
    ax_right.set_xlabel('Predicted Credit Score')
    ax_right.set_ylabel('Frequency')
    ax_right.legend(title='Actual Class')

    plt.suptitle('Comparative Credit Score Distribution by Actual Class: Real vs Synthetic Models', fontsize=16, fontweight='bold')

    plt.tight_layout()
    return fig