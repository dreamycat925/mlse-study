"""
Bayesian Multinomial Logistic Regression for PPA Classification
- Preserves original behavior and results
- English comments and light formatting only
"""

import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns  # kept for compatibility if used elsewhere
import pytensor.tensor as pt

# ----------------------------- Data preparation -----------------------------
mlse_cols = ['MLSE_speech', 'MLSE_phonemic', 'MLSE_semantic',
             'MLSE_grammar', 'MLSE_WorkingMemory']

# Use PPA-only subset
df_ppa = df[df['disease'].isin(['lvPPA', 'naPPA', 'svPPA'])].copy()

# Standardize predictors exactly as in the original code
X = df_ppa[mlse_cols].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=mlse_cols)

# Class encoding: lvPPA=0 (baseline), naPPA=1, svPPA=2
y = df_ppa['disease'].map({'lvPPA': 0, 'naPPA': 1, 'svPPA': 2}).values

print(f"Sample size: {X.shape[0]}")
print("Class distribution:", dict(zip(['lvPPA', 'naPPA', 'svPPA'], np.bincount(y))))

# ---------------- Bayesian multinomial logistic regression (baseline=lvPPA) --
with pm.Model() as multinomial_model:
    # Priors (same shapes and scales)
    beta = pm.Normal('beta', mu=0, sigma=0.25, shape=(X.shape[1], 2))
    intercept = pm.Normal('intercept', mu=0, sigma=0.5, shape=(2,))

    # Linear predictors for naPPA and svPPA (lvPPA is baseline)
    eta_12 = pm.math.dot(X.values, beta) + intercept  # shape: (n_obs, 2) -> [na, sv]
    eta_0 = pt.zeros_like(eta_12[:, 0])               # baseline column (lv)
    eta_all = pt.stack([eta_0, eta_12[:, 0], eta_12[:, 1]], axis=1)  # (n_obs, 3)

    # Softmax with max-shift for numerical stability (exactly as original)
    eta_max = pt.max(eta_all, axis=1, keepdims=True)
    exp_eta = pt.exp(eta_all - eta_max)
    prob_raw = exp_eta / pt.sum(exp_eta, axis=1, keepdims=True)

    # Clip to avoid 0/1 at the boundaries (kept as-is)
    epsilon = 1e-8
    prob = pm.Deterministic('p', pt.clip(prob_raw, epsilon, 1 - epsilon))

    # Likelihood uses probabilities (same as original)
    yl = pm.Categorical('yl', p=prob, observed=y)

    # Same initialization strategy as the original code
    start_vals = {
        'beta': np.random.normal(0, 0.01, size=(X.shape[1], 2)),
        'intercept': np.array([0.01, -0.01])
    }

    # Sampling settings unchanged
    trace = pm.sample(
        draws=2000, tune=1000, chains=4, init="adapt_diag",
        initvals=start_vals, target_accept=0.9,
        return_inferencedata=True, progressbar=True, cores=1, random_seed=42
    )

print("MCMC sampling completed!")

# ----------------------------- Convergence check -----------------------------
summary = az.summary(trace, hdi_prob=0.95)
r_hat_max = summary['r_hat'].max()
print(f"Maximum R-hat: {r_hat_max:.4f}")

# ---------------- In-sample classification (posterior mean probabilities) ----
p_mean = trace.posterior['p'].mean(dim=['chain', 'draw']).values  # (n_obs, 3)
y_pred = np.argmax(p_mean, axis=1)
class_names = ['lvPPA', 'naPPA', 'svPPA']

print("\nClassification Report (in-sample):")
print(classification_report(y, y_pred, target_names=class_names, zero_division=0))

# ---------------------- Trace and posterior plots (beta only) ----------------
print("\nGenerating Trace & Posterior plots...")
az.plot_trace(trace, var_names=['beta'], compact=False)
plt.suptitle('MCMC Trace Plots: Beta Coefficients', fontsize=14, y=0.98)
plt.tight_layout(); plt.show()

az.plot_posterior(trace, var_names=['beta'], hdi_prob=0.95)
plt.suptitle('Posterior Distributions: Beta Coefficients (95% HPD)', fontsize=14, y=0.98)
plt.tight_layout(); plt.show()

# ---------------- ROC curves with AUC uncertainty (class-wise + macro) -------
print("Generating ROC curves (class-wise + macro, 95% HPD)...")

def auc_samples_from_post(y_true_binary, prob_samples):
    """
    y_true_binary: (n_obs,) in {0,1}
    prob_samples : (n_draws, n_obs) posterior probabilities for the class
    returns      : (n_draws,) AUC per posterior draw
    """
    out = []
    for i in range(prob_samples.shape[0]):
        try:
            fpr, tpr, _ = roc_curve(y_true_binary, prob_samples[i])
            out.append(auc(fpr, tpr))
        except Exception:
            continue
    return np.asarray(out)

# (chains, draws, obs, class) -> (draws, obs, class)
p_samples = trace.posterior['p'].values
p_samples = p_samples.reshape(-1, p_samples.shape[2], p_samples.shape[3])

# One-vs-rest binarization
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['blue', 'red', 'green']
auc_stats = {}

# Class-wise ROC + AUC HDI
for i, (name, col) in enumerate(zip(class_names, colors)):
    auc_samps = auc_samples_from_post(y_bin[:, i], p_samples[:, :, i])
    if len(auc_samps) == 0:
        continue
    med = float(np.median(auc_samps))
    lo, hi = az.hdi(auc_samps, hdi_prob=0.95)
    auc_stats[name] = {'median': med, 'hdi_lower': float(lo), 'hdi_upper': float(hi), 'samples': auc_samps}

    # Representative ROC from posterior-mean probabilities
    fpr, tpr, _ = roc_curve(y_bin[:, i], p_mean[:, i])
    ax.plot(fpr, tpr, color=col, lw=2,
            label=f'{name} (AUC = {med:.2f}, 95% HPD: {lo:.2f}–{hi:.2f})')

# Macro-average AUC uncertainty (equal weights), subsampling draws to limit cost
print("Calculating macro-average AUC uncertainty...")
macro_samples = []
rng = np.random.default_rng(123)
idx = rng.choice(p_samples.shape[0], size=min(500, p_samples.shape[0]), replace=False)
for k in idx:
    try:
        class_aucs = []
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], p_samples[k, :, i])
            class_aucs.append(auc(fpr_i, tpr_i))
        macro_samples.append(np.mean(class_aucs))
    except Exception:
        pass
macro_samples = np.asarray(macro_samples)
macro_med = float(np.median(macro_samples))
macro_lo, macro_hi = az.hdi(macro_samples, hdi_prob=0.95)

# Macro-average ROC (display only; posterior-mean probabilities)
all_fpr = np.unique(np.concatenate([roc_curve(y_bin[:, i], p_mean[:, i])[0] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], p_mean[:, i])
    mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
mean_tpr /= n_classes
ax.plot(all_fpr, mean_tpr, color='navy', linestyle=':', lw=2,
        label=f'Macro-avg (AUC = {macro_med:.2f}, 95% HPD: {macro_lo:.2f}–{macro_hi:.2f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
ax.set_xlabel('1 − Specificity', fontsize=14, fontweight='bold')
ax.set_ylabel('Sensitivity', fontsize=14, fontweight='bold')
ax.set_title('ROC Curves with AUC Uncertainty (95% HPD)', fontsize=14, fontweight='bold', pad=14)
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# --------------------------- AUC summary printout ----------------------------
print("\nAUC STATISTICS (95% HPD)")
for name in class_names:
    if name in auc_stats:
        s = auc_stats[name]
        print(f"- {name}: {s['median']:.3f}  [95% HPD {s['hdi_lower']:.3f}, {s['hdi_upper']:.3f}]")
print(f"- Macro-average (equal weights): {macro_med:.3f}  [95% HPD {macro_lo:.3f}, {macro_hi:.3f}]")

# ------------------- Standardized odds ratios (per +1 SD) --------------------
print("\nCalculating standardized ORs...")
beta_samples = trace.posterior['beta']
or_samples = np.exp(beta_samples)

rows = []
feat_labels = ['Motor Speech', 'Phonology', 'Semantics', 'Syntax', 'Working Memory']
comparisons = ['naPPA vs lvPPA', 'svPPA vs lvPPA']

for i, feat in enumerate(feat_labels):
    for j, comp in enumerate(comparisons):
        draws = or_samples[:, :, i, j].values.flatten()
        med = float(np.median(draws))
        lo, hi = az.hdi(draws, 0.95)
        rows.append([feat, comp, med, float(lo), float(hi)])

or_table = pd.DataFrame(rows, columns=['MLSE Feature', 'Comparison', 'Median OR', 'HDI 2.5%', 'HDI 97.5%'])
print("\nStandardized ORs (95% HPD), baseline = lvPPA")
print(or_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ------------------------------- Save posterior ------------------------------
trace.to_netcdf('ppa_bayesian_classification.nc')
print("\nSaved posterior to 'ppa_bayesian_classification.nc'")
