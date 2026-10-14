from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

# Pretty labels for legend
pretty_name = {
    'MLSE_speech': 'Motor speech',
    'MLSE_phonemic': 'Phonology',
    'MLSE_semantic': 'Semantics knowledge',
    'MLSE_grammar': 'Syntax',
    'MLSE_WorkingMemory': 'Working memory'
}

# Target diseases and MLSE domains
diseases  = ['lvPPA', 'naPPA', 'svPPA']
functions = ['MLSE_speech', 'MLSE_phonemic', 'MLSE_semantic',
             'MLSE_grammar', 'MLSE_WorkingMemory']

def calculate_roc(disease, df):
    """
    Build HP-vs-disease dataset and compute ROC per domain.
    Scores are flipped as (max - raw) so that higher = more abnormal.
    Returns {domain: (fpr, tpr, auc)} preserving the order in `functions`.
    """
    disease_data  = df[df['disease'] == disease]
    control_data  = df[df['disease'] == 'HP']  # controls: HP only
    combined_data = pd.concat([disease_data, control_data])

    roc_data = {}
    for func in functions:
        y_true   = (combined_data['disease'] == disease).astype(int)
        y_scores = combined_data[func].max() - combined_data[func]  # flip: higher = more abnormal
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_value   = roc_auc_score(y_true, y_scores)
        roc_data[func] = (fpr, tpr, auc_value)
    return roc_data

def plot_roc_curves(ax, data, title, pretty_name_map):
    """
    Plot multiple ROC curves on the given Axes.
    Line colors/styles match the original script.
    """
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    styles = ['-', ':', '--', '-.', '-']

    for (name, (fpr, tpr, auc_value)), c, ls in zip(data.items(), colors, styles):
        label_txt = f"{pretty_name_map.get(name, name)} (AUC = {auc_value:.2f})"
        ax.plot(fpr, tpr, linestyle=ls, color=c, label=label_txt)

    # Reference diagonal
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Reference Line')

    # Axes styling (kept as in the original)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.05)
    ax.set_xlabel('1 - Specificity', fontsize=14)
    ax.set_ylabel('Sensitivity', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=14, width=2)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')

# --- One figure per disease (HP vs disease) ---
for disease in diseases:
    data = calculate_roc(disease, df)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_roc_curves(ax, data, f'{disease} vs HP', pretty_name)
    plt.tight_layout()
    plt.show()
    # To save instead of just showing:
    # fig.savefig(f'roc_{disease}_vs_HP.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
