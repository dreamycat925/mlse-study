from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.tree import export_text, plot_tree


FEATURES = [
    "MLSE_speech", "MLSE_phonemic", "MLSE_semantic",
    "MLSE_grammar", "MLSE_WorkingMemory",
]
FEATURE_NAMES = [
    "Motor speech", "Phonology", "Semantic knowledge", "Syntax", "Working memory",
]
LABELS = ["naPPA", "svPPA", "lvPPA"]  # naPPA corresponds to nfvPPA in the paper.
TREE_GRID = [50, 75, 100, 150, 200, 300, 450, 600, 900, 1200]


def run_rf_analysis(
    df, *, outer_splits=5, repeats=10, inner_splits=4,
    tree_grid=None, n_jobs=-1, make_plots=True,
):
    """Run nested CV and return numeric results and the illustrative tree.

    All folds use the original seeds: outer=1, inner=42, forest=3.
    SD is calculated across outer folds with ddof=0, as in the notebook.
    Voting ties retain the first encountered prediction, as in the notebook.
    The illustrative tree is selected using outer-test accuracy; its selected
    accuracy is descriptive and is not an unbiased estimate of tree performance.
    """
    grid = list(TREE_GRID if tree_grid is None else tree_grid)
    required = ["disease", *FEATURES]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df["disease"].eq("nfvPPA").any():
        raise ValueError("This notebook uses 'naPPA'. Confirm label mapping before analysis.")
    data = df.loc[df["disease"].isin(LABELS), required].copy().reset_index(drop=True)
    X = data[FEATURES].apply(pd.to_numeric, errors="raise")
    y = data["disease"]
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("MLSE domain scores contain missing/non-finite values; check the data.")
    counts = y.value_counts().reindex(LABELS, fill_value=0)
    if counts.min() < outer_splits:
        raise ValueError("Each PPA subtype needs at least outer_splits participants.")
    print("PPA sample size:", len(data), "| Class counts:", counts.to_dict())

    def forest(n):
        return RandomForestClassifier(
            n_estimators=n, max_depth=4, random_state=3, n_jobs=n_jobs,
        )

    def tune(X_train, y_train):
        if y_train.value_counts().min() < inner_splits:
            raise ValueError("Too few training participants for stratified inner CV.")
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)
        best_n, best_score = None, -np.inf
        for n in grid:
            scores = []
            for tr, va in inner_cv.split(X_train, y_train):
                model = forest(n)
                model.fit(X_train.iloc[tr], y_train.iloc[tr])
                pred = model.predict(X_train.iloc[va])
                scores.append(f1_score(y_train.iloc[va], pred, average="macro"))
            score = float(np.mean(scores))
            if score > best_score or (
                np.isclose(score, best_score) and (best_n is None or n < best_n)
            ):
                best_n, best_score = n, score
        return best_n, best_score

    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_splits, n_repeats=repeats, random_state=1,
    )
    rows, selected, importances, accuracies = [], [], [], []
    votes = defaultdict(list)
    all_true, all_pred = [], []
    best_tree, best_tree_score, best_tree_classes = None, -np.inf, None

    for fold, (tr, te) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]
        best_n, inner_score = tune(X_train, y_train)
        model = forest(best_n)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        selected.append({"fold": fold, "n_estimators": best_n, "inner_macro_f1": inner_score})
        print(f"Fold {fold:02d}: n_estimators={best_n}, inner macro-F1={inner_score:.3f}")

        report = classification_report(
            y_test, pred, labels=LABELS, target_names=LABELS,
            zero_division=0, output_dict=True,
        )
        for label in LABELS + ["macro avg", "weighted avg"]:
            rows.append({"fold": fold, "class": label, **report[label]})
        accuracies.append(accuracy_score(y_test, pred))
        importances.append(model.feature_importances_)
        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())
        for position, prediction in zip(te, pred):
            votes[int(position)].append(prediction)

        # Individual trees predict internal numeric IDs; decode using this forest.
        for tree in model.estimators_:
            ids = tree.predict(X_test.to_numpy()).astype(int)
            tree_pred = model.classes_[ids]
            score = accuracy_score(y_test, tree_pred)
            if score > best_tree_score:  # First tree wins exact ties, as with argmax.
                best_tree, best_tree_score = tree, score
                best_tree_classes = model.classes_.copy()

    if len(votes) != len(data) or any(len(v) != repeats for v in votes.values()):
        raise RuntimeError("Unexpected number of out-of-fold predictions per participant.")
    fold_metrics = pd.DataFrame(rows)
    metric_names = ["precision", "recall", "f1-score"]
    summary_rows = []
    for label in LABELS + ["macro avg", "weighted avg"]:
        subset = fold_metrics.loc[fold_metrics["class"].eq(label)]
        for metric in metric_names:
            values = subset[metric].to_numpy()
            summary_rows.append({
                "class": label, "metric": metric,
                "mean": values.mean(), "sd": values.std(ddof=0),
            })
    summary = pd.DataFrame(summary_rows)
    display_table = summary.assign(
        value=summary.apply(lambda r: f"{r['mean']:.2f} ± {r['sd']:.2f}", axis=1)
    ).pivot(index="class", columns="metric", values="value")
    print("\nFold-averaged metrics (mean ± SD):")
    print(display_table.reindex(LABELS + ["macro avg", "weighted avg"])[metric_names])
    print(f"Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies, ddof=0):.3f}")

    majority_pred = [Counter(votes[i]).most_common(1)[0][0] for i in range(len(data))]
    cm_oof = confusion_matrix(all_true, all_pred, labels=LABELS)
    cm_majority = confusion_matrix(y, majority_pred, labels=LABELS)
    print("\nMajority-vote report:")
    print(classification_report(y, majority_pred, labels=LABELS, zero_division=0))
    print("OOF confusion matrix:\n", cm_oof)
    print("Majority-vote confusion matrix:\n", cm_majority)
    selected = pd.DataFrame(selected)
    print("Selected tree counts:\n", selected["n_estimators"].value_counts().sort_index())
    mean_importance = pd.Series(np.mean(importances, axis=0), index=FEATURE_NAMES)
    print("\nMean importance across outer folds:\n", mean_importance.sort_values(ascending=False))

    tree_labels = [str(best_tree_classes[int(c)]) for c in best_tree.classes_]
    print("\nIllustrative tree (selected outer-test accuracy, not independent validation):",
          f"{best_tree_score:.3f}")
    print(export_text(best_tree, feature_names=FEATURE_NAMES, class_names=tree_labels))
    if make_plots:
        for cm, title in [(cm_oof, "Pooled OOF predictions"), (cm_majority, "Majority vote per participant")]:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
            plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
            plt.tight_layout(); plt.show()
        mean_importance.sort_values().plot.barh(figsize=(8, 5))
        plt.xlabel("Mean feature importance across outer folds")
        plt.tight_layout(); plt.show()
        plt.figure(figsize=(20, 10))
        plot_tree(best_tree, feature_names=FEATURE_NAMES, class_names=tree_labels, filled=True, rounded=True)
        plt.title("Illustrative RF tree (selected using outer-test accuracy)")
        plt.tight_layout(); plt.show()

    return {
        "summary": summary, "fold_metrics": fold_metrics,
        "fold_accuracy": np.asarray(accuracies), "selected_n_estimators": selected,
        "confusion_oof": pd.DataFrame(cm_oof, index=LABELS, columns=LABELS),
        "confusion_majority": pd.DataFrame(cm_majority, index=LABELS, columns=LABELS),
        "mean_feature_importance": mean_importance,
        "best_tree": best_tree, "best_tree_class_names": tree_labels,
        "best_tree_selected_test_accuracy": best_tree_score,
    }


# Run after preparing and verifying the analysis DataFrame:
# results = run_rf_analysis(df)
