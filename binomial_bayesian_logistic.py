import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Pretty labels for plots
pretty_name = {
    'MLSE_speech': 'Motor speech',
    'MLSE_phonemic': 'Phonology',
    'MLSE_semantic': 'Semantics knowledge',
    'MLSE_grammar': 'Syntax',
    'MLSE_WorkingMemory': 'Working memory'
}

diseases = ['lvPPA', 'naPPA', 'svPPA']
features = ['MLSE_speech', 'MLSE_phonemic', 'MLSE_semantic',
            'MLSE_grammar', 'MLSE_WorkingMemory']

def zscore(x):
    """Return z-scored vector; if std==0, return zeros to avoid division by zero."""
    m = np.nanmean(x)
    s = np.nanstd(x)
    if s == 0 or np.isclose(s, 0.0):
        return np.zeros_like(x, dtype=float)
    return (x - m) / s

def bayesian_roc_analysis(
    df,
    disease,
    feature,
    draws=2000,
    tune=1000,
    chains=4,
    target_accept=0.9,
    seed=42,
    show_summary=False,
):
    """
    Fit a 1-predictor Bayesian logistic regression (standardized) for HP vs target disease.
    Returns dict with ROC (fpr,tpr), AUC posterior (median & 95% HDI), standardized OR, and idata.
    """
    # Subset to HP and the target disease; label disease=1, HP=0
    sub = df[df["disease"].isin(["HP", disease])].copy()
    y = (sub["disease"] == disease).astype(int).to_numpy()
    x = zscore(sub[feature].to_numpy().astype(float))

    with pm.Model() as model:
        beta0 = pm.Normal("beta0", mu=0, sigma=1)
        beta1 = pm.Normal("beta1", mu=0, sigma=1)
        eta = beta0 + beta1 * x
        p = pm.Deterministic("p", pm.math.sigmoid(eta))
        pm.Bernoulli("y_obs", p=p, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
            progressbar=False,
        )

    # Posterior AUC: ROC/AUC for each posterior draw of p
    p_post = idata.posterior["p"].values.reshape(-1, y.shape[0])  # (samples, n_obs)
    auc_samples = np.array([auc(*roc_curve(y, s)[:2]) for s in p_post])
    auc_median = float(np.median(auc_samples))
    auc_hdi_low, auc_hdi_high = az.hdi(auc_samples, hdi_prob=0.95)

    # ROC of mean posterior probability (for plotting)
    fpr, tpr, _ = roc_curve(y, p_post.mean(axis=0))

    # Standardized odds ratio (per +1 SD)
    or_samples = np.exp(idata.posterior["beta1"].values.flatten())
    or_median = float(np.median(or_samples))
    or_hdi_low, or_hdi_high = az.hdi(or_samples, hdi_prob=0.95)

    if show_summary:
        feat = pretty_name.get(feature, feature)
        print(f"=== {disease} vs HP | Feature: {feat} ===")
        print(f"AUC (median, 95% HDI): {auc_median:.3f} "
              f"[{float(auc_hdi_low):.3f}, {float(auc_hdi_high):.3f}]")
        print(f"Standardized OR (median, 95% HDI): {or_median:.3f} "
              f"[{float(or_hdi_low):.3f}, {float(or_hdi_high):.3f}]")

    return {
        "fpr": fpr, "tpr": tpr,
        "auc_median": auc_median,
        "auc_hdi": (float(auc_hdi_low), float(auc_hdi_high)),
        "auc_samples": auc_samples,
        "or_median": or_median,
        "or_hdi": (float(or_hdi_low), float(or_hdi_high)),
        "idata": idata,
    }

def plot_bayesian_roc(fpr, tpr, auc_median, auc_hdi, label_text):
    """Plot a ROC curve and show AUC (median) with 95% HDI in legend."""
    plt.plot(
        fpr, tpr, linewidth=2.0,
        label=f"{label_text} (AUC = {auc_median:.2f}, 95% HDI: {auc_hdi[0]:.2f}–{auc_hdi[1]:.2f})"
    )

# One figure per disease
for disease in diseases:
    plt.figure(figsize=(8, 6))
    for feature in features:
        res = bayesian_roc_analysis(df, disease, feature, show_summary=False)
        plot_bayesian_roc(
            res["fpr"], res["tpr"], res["auc_median"], res["auc_hdi"],
            pretty_name.get(feature, feature)
        )
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.2)
    plt.title(f"{disease} vs HP (Bayesian ROC)")
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
