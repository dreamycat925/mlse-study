"""
HP-derived MLSE cutoffs and abnormality visualization
- Identical computations to the original code
- 95% HDI lower bound from healthy participants (HP) per domain
- Per-domain strip plot with cutoff line
- Group-wise abnormality heatmap (< cutoff)
"""

import pandas as pd
import numpy as np
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----- MLSE domains and display settings (same as original) -----
mlse_cols = ['MLSE_speech', 'MLSE_phonemic', 'MLSE_semantic',
             'MLSE_grammar', 'MLSE_WorkingMemory']

pretty_name = {
    'MLSE_speech': 'Motor speech',
    'MLSE_phonemic': 'Phonology',
    'MLSE_semantic': 'Semantics knowledge',
    'MLSE_grammar': 'Syntax',
    'MLSE_WorkingMemory': 'Working memory'
}
domain_max = {'MLSE_speech': 30, 'MLSE_phonemic': 30, 'MLSE_semantic': 20,
              'MLSE_grammar': 10, 'MLSE_WorkingMemory': 10}
domain_color = {
    'MLSE_speech':       '#1f77b4',  # blue
    'MLSE_phonemic':     '#ff7f0e',  # orange
    'MLSE_semantic':     '#2ca02c',  # green
    'MLSE_grammar':      '#d62728',  # red
    'MLSE_WorkingMemory':'#9467bd'   # purple
}

# ----- 1) 95% HDI lower-bound cutoffs from healthy participants -----
df_hs = df[df['disease'] == 'HP'].copy()

cutoff_dict = {}
print("95% HDI (per MLSE domain) and cutoffs (lower bounds):")
print("-" * 60)
for col in mlse_cols:
    values = df_hs[col].dropna().values
    hpd_interval = az.hdi(values, hdi_prob=0.95)
    cutoff = float(hpd_interval[0])          # use the lower bound as the cutoff
    cutoff_dict[col] = cutoff
    print(f"{col:<20}:  Cutoff = {cutoff:.3f}   (95% HDI = [{hpd_interval[0]:.3f}, {hpd_interval[1]:.3f}])")

# Optional: save cutoffs for later use
# import yaml
# with open('results/hp_cutoffs.yaml', 'w') as f:
#     yaml.safe_dump({k: float(v) for k, v in cutoff_dict.items()}, f)

# ----- 2) Per-domain strip plots with cutoff line -----
sns.set(style="whitegrid")
for col in mlse_cols:
    vals = df_hs[col].dropna()
    cutoff = cutoff_dict[col]
    color  = domain_color[col]
    label  = pretty_name[col]

    fig, ax = plt.subplots(figsize=(5, 5))
    # HP dots with small horizontal jitter
    sns.stripplot(x=['HP']*len(vals), y=vals, color='0.35', alpha=0.6,
                  jitter=0.15, size=6, ax=ax, order=['HP'])

    # cutoff line and annotation
    ax.axhline(cutoff, color=color, linewidth=2.5)
    ax.text(0.32, cutoff, f'Cutoff = {cutoff:.2f}',
            color=color, va='bottom', ha='left', fontsize=11)

    # axes styling (kept consistent with original)
    ax.set_xlabel('')
    ax.set_ylabel('Raw score', fontsize=16, fontweight='bold')
    ax.set_title(f'Healthy participants • {label}', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.5, domain_max.get(col, max(vals) + 1) + 0.5)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([0])
    ax.set_xticklabels(['HP'])
    ax.tick_params(axis='both', which='major', labelsize=22, width=2)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight('bold')

    # legend (line only)
    legend_line = [Line2D([0], [0], color=color, lw=2.5)]
    ax.legend(legend_line, ['95% HDI lower bound'], loc='lower right', fontsize=10)

    plt.tight_layout()
    # To save:
    # safe = label.lower().replace(' ', '_')
    # plt.savefig(f'cutoff_{safe}.png', dpi=300, bbox_inches='tight')
    plt.show()

# ----- 3) Abnormal flags (< cutoff) and group-wise abnormality heatmap -----
df_flagged = df.copy()
for col in mlse_cols:
    df_flagged[f'{col}_abnormal'] = (df_flagged[col] < cutoff_dict[col]).astype(int)

abnormal_cols = [f'{col}_abnormal' for col in mlse_cols]
group_abnormal_rate = df_flagged.groupby('disease')[abnormal_cols].mean().round(3)
group_abnormal_rate.columns = mlse_cols  # revert to original domain names

# desired row order: naPPA, svPPA, lvPPA, HP (keep only those present)
desired_order = ['naPPA', 'svPPA', 'lvPPA', 'HP']
present_order = [g for g in desired_order if g in group_abnormal_rate.index]
group_abnormal_rate = group_abnormal_rate.reindex(present_order)

# rename columns for display
group_abnormal_rate = group_abnormal_rate.rename(columns=pretty_name)

plt.figure(figsize=(10, 6))
sns.heatmap(
    group_abnormal_rate,
    annot=True,
    cmap="coolwarm",
    vmin=0, vmax=1,
    fmt=".2f",
    cbar_kws={"label": "Abnormal Rate"},
    annot_kws={"fontsize": 20, "weight": "bold"}
)
plt.title("Abnormality Rate per MLSE Domain by Disease Group", fontsize=14)
plt.ylabel("Disease Group")
plt.xlabel("MLSE Domain")
plt.tight_layout()
plt.show()

# 'cutoff_dict' now holds the cutoffs, e.g., cutoff_dict['MLSE_speech']
