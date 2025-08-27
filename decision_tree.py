# MLSE: Diversity‑constrained CART + Bayesian bootstrap
# - No feature reuse along a path (diversity constraint)
# - For each internal node: bootstrap the split-feature selection prob and the 95% HDI of thresholds
# - Inline Graphviz (SVG) rendering + saving to SVG/PNG

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import classification_report
from IPython.display import SVG, Image, display
from graphviz import Digraph

# ----- Assumptions: df exists and includes the following columns -----
mlse_cols = ['MLSE_speech', 'MLSE_phonemic', 'MLSE_semantic',
             'MLSE_grammar', 'MLSE_WorkingMemory']
target = 'disease'
classes = ['lvPPA', 'naPPA', 'svPPA']  # display order (adjust if needed)
pretty = {
    'MLSE_speech': 'Motor speech',
    'MLSE_phonemic': 'Phonology',
    'MLSE_semantic': 'Semantics knowledge',
    'MLSE_grammar': 'Syntax',
    'MLSE_WorkingMemory': 'Working memory',
}

# ----- Hyperparameters (same values as original) -----
max_depth = 4
min_samples_leaf = 5
criterion = 'gini'
seed = 42

# ----- Restrict to the 3 PPA classes -----
df_ppa = df[df[target].isin(classes)].copy()
df_ppa = df_ppa[mlse_cols + [target]].copy()

# ---------------- Diversity‑constrained CART (no feature reuse per path) ----------------
@dataclass
class Node:
    feature = None
    threshold = None
    n = 0
    counts = None
    pred = None
    left = None
    right = None

def majority(y):
    return y.value_counts().idxmax()

def stump_on_feature(xy, feat, sample_weight=None):
    # Fit a 1‑level stump on a single feature; return None if not splittable
    if xy[target].nunique() == 1:
        return None
    clf = DecisionTreeClassifier(
        max_depth=1, criterion=criterion, random_state=seed,
        min_samples_leaf=min_samples_leaf, class_weight='balanced'
    )
    try:
        clf.fit(xy[[feat]].values, xy[target].values, sample_weight=sample_weight)
    except ValueError:
        return None
    tr = clf.tree_
    if tr.feature[0] == _tree.TREE_UNDEFINED:
        return None
    nL, nR = tr.n_node_samples[tr.children_left[0]], tr.n_node_samples[tr.children_right[0]]
    if nL < min_samples_leaf or nR < min_samples_leaf:
        return None
    return clf

def impurity_decrease(clf):
    # Parent impurity minus weighted child impurities
    tr = clf.tree_
    nP = tr.n_node_samples[0]
    L, R = tr.children_left[0], tr.children_right[0]
    imp_parent = tr.impurity[0]
    imp_child = (tr.n_node_samples[L]/nP)*tr.impurity[L] + (tr.n_node_samples[R]/nP)*tr.impurity[R]
    return imp_parent - imp_child

def build_tree(xy, depth, used_feats):
    # Recursively build a tree with the diversity constraint
    node = Node()
    node.n = len(xy)
    node.counts = xy[target].value_counts().reindex(classes).fillna(0).astype(int).to_dict()
    node.pred = majority(xy[target])

    # Stop conditions
    if depth >= max_depth or xy[target].nunique() == 1 or node.n < 2*min_samples_leaf:
        return node

    avail = [f for f in mlse_cols if f not in used_feats]
    if not avail:
        return node

    best_feat, best_clf, best_gain = None, None, -np.inf
    for f in avail:
        clf = stump_on_feature(xy, f)
        if clf is None:
            continue
        gain = impurity_decrease(clf)
        if gain > best_gain + 1e-12:
            best_feat, best_clf, best_gain = f, clf, gain

    if best_clf is None or best_gain <= 0:
        return node

    thr = float(best_clf.tree_.threshold[0])
    mask_left = xy[best_feat].values <= thr
    xyL, xyR = xy.loc[mask_left], xy.loc[~mask_left]
    if len(xyL) < min_samples_leaf or len(xyR) < min_samples_leaf:
        return node

    node.feature = best_feat
    node.threshold = thr
    node.left  = build_tree(xyL, depth+1, used_feats + (best_feat,))
    node.right = build_tree(xyR, depth+1, used_feats + (best_feat,))
    return node

def predict_one(row, nd):
    # Walk down the tree until a leaf
    while nd and nd.feature is not None:
        nd = nd.left if row[nd.feature] <= nd.threshold else nd.right
    return nd.pred if nd else majority(df_ppa[target])

def dump_rules(nd, depth=0):
    # Print human‑readable rules
    ind = "  " * depth
    if nd.feature is None:
        print(f"{ind}return {nd.pred} (n={nd.n}, counts={nd.counts})"); return
    lab = pretty.get(nd.feature, nd.feature)
    print(f"{ind}if {lab} ≤ {nd.threshold:.1f}:");  dump_rules(nd.left, depth+1)
    print(f"{ind}else:  # {lab} > {nd.threshold:.1f}"); dump_rules(nd.right, depth+1)

# ----- Train, evaluate, and print rules -----
tree_root = build_tree(df_ppa, depth=0, used_feats=tuple())
y_pred = df_ppa.apply(lambda r: predict_one(r, tree_root), axis=1)
print("[In-sample report]\n", classification_report(df_ppa[target], y_pred, zero_division=0))
print("\n[Rules]"); dump_rules(tree_root)

# ---------------- Bayesian bootstrap: selection prob + threshold intervals ----------------
try:
    import arviz as az
    def hdi_1d(x, prob=0.95):
        lo, hi = az.hdi(np.asarray(x), hdi_prob=prob); return float(lo), float(hi)
except Exception:
    def hdi_1d(x, prob=0.95):
        x = np.sort(np.asarray(x)); a = (1-prob)/2; return float(np.quantile(x, a)), float(np.quantile(x, 1-a))

def collect_node_info(nd, df_in, used_feats=(), mask=None, path="N0"):
    # Collect masks/paths for each internal node
    if mask is None: mask = np.ones(len(df_in), dtype=bool)
    info = []
    if nd.feature is None: return info
    feat, thr = nd.feature, nd.threshold
    mask_left  = mask & (df_in[feat].values <= thr)
    mask_right = mask & (df_in[feat].values >  thr)
    info.append({'path': path, 'feat': feat, 'thr': float(thr),
                 'used_feats': tuple(used_feats), 'mask': mask.copy(), 'n': int(mask.sum())})
    info += collect_node_info(nd.left,  df_in, used_feats + (feat,), mask_left,  path+"L")
    info += collect_node_info(nd.right, df_in, used_feats + (feat,), mask_right, path+"R")
    return info

def bayes_bootstrap_summary(root, df_block, B=1000, seed=42):
    # For each node: Dirichlet weights -> best stump (feature+threshold)
    rng = np.random.default_rng(seed)
    rows = []
    for nd in collect_node_info(root, df_block):
        m = nd['mask'].sum()
        if m < 2*min_samples_leaf: 
            continue
        Xb = df_block.loc[nd['mask'], mlse_cols].copy()
        yb = df_block.loc[nd['mask'], target].values
        avail = [f for f in mlse_cols if f not in nd['used_feats']]
        if not avail: 
            continue

        for b in range(B):
            w = rng.dirichlet(np.ones(m))
            best_feat, best_thr, best_gain = None, None, -np.inf
            for f in avail:
                tmp = pd.concat([Xb[[f]], pd.Series(yb, name=target)], axis=1)
                clf = stump_on_feature(tmp, f, sample_weight=w)
                if clf is None: 
                    continue
                g = impurity_decrease(clf)
                if g > best_gain + 1e-12:
                    best_feat, best_thr, best_gain = f, float(clf.tree_.threshold[0]), g
            if best_feat is not None:
                rows.append({'node_path': nd['path'], 'feature': pretty.get(best_feat, best_feat), 'thr': best_thr})

    if not rows:
        return pd.DataFrame(columns=['node_path','feature','select_prob','thr_median','thr_95%_lo','thr_95%_hi','n_boot'])

    boot = pd.DataFrame(rows)
    out = []
    for (path, feat), grp in boot.groupby(['node_path','feature']):
        c = len(grp)
        tot = len(boot[boot['node_path'] == path])
        med = float(np.median(grp['thr']))
        lo, hi = hdi_1d(grp['thr'], 0.95)
        out.append({'node_path': path, 'feature': feat, 'select_prob': c/tot,
                    'thr_median': med, 'thr_95%_lo': lo, 'thr_95%_hi': hi, 'n_boot': c})
    return pd.DataFrame(out).sort_values(['node_path', 'select_prob'], ascending=[True, False])

boot_summary = bayes_bootstrap_summary(tree_root, df_ppa, B=1000, seed=42)
print("\n[Bayesian bootstrap] selection probabilities & threshold intervals")
print(boot_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ---------------- Graphviz rendering (inline) + save to files ----------------
def render_tree_with_intervals_inline(root, boot_summary, show_alt=True, font='Helvetica'):
    # Pick the most frequent feature per node_path
    top = (boot_summary.sort_values(['node_path','select_prob'], ascending=[True, False])
                        .groupby('node_path').head(1).set_index('node_path'))

    g = Digraph('MLSE',
        node_attr={'shape':'box','style':'rounded,filled','fontname':font,'color':'#888'},
        edge_attr={'fontname':font}
    )

    def add(nd, path):
        if nd.feature is None:
            g.node(path, f"{nd.pred}\n(n={nd.n})", fillcolor='#CDECCE'); return
        feat = pretty.get(nd.feature, nd.feature)
        label = f"{feat} ≤ {nd.threshold:.1f}\n(n={nd.n})"
        if path in top.index:
            r = top.loc[path]
            if r['feature'] == pretty.get(nd.feature, nd.feature):
                label = (f"{feat} ≤ {nd.threshold:.1f}\n"
                         f"[{r['thr_95%_lo']:.1f}–{r['thr_95%_hi']:.1f}]  "
                         f"p={r['select_prob']:.2f}\n(n={nd.n})")
            elif show_alt:
                label = f"{label}\n(p_sel={r['select_prob']:.2f} for {r['feature']})"
        g.node(path, label=label, fillcolor='#F5EEDF')
        if nd.left:  g.edge(path, path+'L', label='True');  add(nd.left,  path+'L')
        if nd.right: g.edge(path, path+'R', label='False'); add(nd.right, path+'R')

    add(root, "N0")

    # Try SVG first; fall back to PNG if SVG fails
    try:
        svg = g.pipe(format='svg')
        display(SVG(svg))
    except Exception as e:
        try:
            png = g.pipe(format='png')
            display(Image(png))
            print("SVG rendering failed; showing PNG instead.", type(e).__name__, str(e))
        except Exception as e2:
            print("Graphviz rendering failed.", type(e2).__name__, str(e2))
            print("Please check that Graphviz (dot) is installed and on PATH.")
    return g

g = render_tree_with_intervals_inline(tree_root, boot_summary)
g.graph_attr.update(dpi='400', bgcolor='transparent')

# Save both SVG and PNG (same as original behavior)
with open('mlse_tree.svg', 'wb') as f:
    f.write(g.pipe(format='svg'))
with open('mlse_tree.png', 'wb') as f:
    f.write(g.pipe(format='png'))
# Alternatively:
# g.render('mlse_tree', format='png', cleanup=True)
# g.render('mlse_tree', format='svg', cleanup=True)
