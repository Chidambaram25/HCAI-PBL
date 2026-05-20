from django.shortcuts import render
from palmerpenguins import load_penguins
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import io
import base64

def get_penguins():
    df = load_penguins()
    df = df.dropna()
    return df

def train_tree(max_leaves):
    df = get_penguins()
    features = ['bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g']
    X = df[features]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(
        max_leaf_nodes=max_leaves, random_state=42)
    clf.fit(X_train, y_train)
    test_acc = round(accuracy_score(y_test, clf.predict(X_test)), 3)
    n_leaves = clf.get_n_leaves()
    return clf, test_acc, n_leaves

def plot_decision_tree(clf, feature_names):
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(clf, feature_names=feature_names,
              class_names=clf.classes_,
              filled=True, ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img

def get_all_trees():
    results = []
    for max_leaves in range(2, 21):
        clf, test_acc, n_leaves = train_tree(max_leaves)
        results.append({
            'max_leaves': max_leaves,
            'test_acc': test_acc,
            'n_leaves': n_leaves,
            'clf': clf,
        })
    return results

def best_tree_for_lambda(lam):
    results = get_all_trees()
    max_leaves_count = max(r['n_leaves'] for r in results)
    best = min(results,
        key=lambda r: (1 - r['test_acc']) + lam * (r['n_leaves'] / max_leaves_count))
    return best

def get_all_logregs():
    df = get_penguins()
    features = ['bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g']
    X = df[features].values
    y = df['species'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    results = []
    for C in np.logspace(-2, 2, 20):
        clf = LogisticRegression(C=C, max_iter=1000)
        clf.fit(X_train, y_train)
        test_acc = round(accuracy_score(y_test, clf.predict(X_test)), 3)
        omega = float(np.sum(clf.coef_ ** 2))
        results.append({
            'C': C,
            'test_acc': test_acc,
            'omega': omega,
            'clf': clf,
            'scaler': scaler,
        })
    return results

def best_logreg_for_lambda(lam):
    results = get_all_logregs()
    max_omega = max(r['omega'] for r in results)
    best = min(results,
        key=lambda r: (1 - r['test_acc']) + lam * (r['omega'] / max_omega))
    return best

def generate_counterfactuals(clf, x, target_label, features, df,
                              scaler=None, N=5000, k=3):
    x = np.array(x)
    stds = df[features].std().values
    counterfactuals = []
    for _ in range(N):
        noise = np.random.normal(0, stds * 0.5, size=x.shape)
        x_new = x + noise
        x_input = scaler.transform([x_new]) if scaler else [x_new]
        pred = clf.predict(x_input)[0]
        if pred == target_label:
            mad = np.median(np.abs(
                df[features].values - np.median(df[features].values, axis=0)
            ), axis=0)
            mad[mad == 0] = 1e-6
            dist = float(np.sum(np.abs(x_new - x) / mad))
            counterfactuals.append((dist, x_new))
    counterfactuals.sort(key=lambda c: c[0])
    return counterfactuals[:k]

def compute_pdp(clf, X, feature_idx, scaler=None, grid_points=50):
    X = np.array(X)
    grid = np.linspace(X[:, feature_idx].min(),
                       X[:, feature_idx].max(), grid_points)
    pdp_values = []
    for val in grid:
        X_copy = X.copy()
        X_copy[:, feature_idx] = val
        X_input = scaler.transform(X_copy) if scaler else X_copy
        probs = clf.predict_proba(X_input)
        pdp_values.append(probs.mean(axis=0))
    return grid, np.array(pdp_values)

def compute_ale(clf, X, feature_idx, scaler=None, n_bins=20):
    X = np.array(X)
    quantiles = np.percentile(X[:, feature_idx],
                              np.linspace(0, 100, n_bins + 1))
    quantiles = np.unique(quantiles)
    ale_values = []
    bin_centers = []
    for i in range(len(quantiles) - 1):
        mask = ((X[:, feature_idx] >= quantiles[i]) &
                (X[:, feature_idx] <= quantiles[i + 1]))
        if mask.sum() == 0:
            continue
        X_low = X[mask].copy()
        X_high = X[mask].copy()
        X_low[:, feature_idx] = quantiles[i]
        X_high[:, feature_idx] = quantiles[i + 1]
        X_low_input = scaler.transform(X_low) if scaler else X_low
        X_high_input = scaler.transform(X_high) if scaler else X_high
        effect = (clf.predict_proba(X_high_input) -
                  clf.predict_proba(X_low_input)).mean(axis=0)
        ale_values.append(effect)
        bin_centers.append((quantiles[i] + quantiles[i + 1]) / 2)
    ale_values = np.cumsum(ale_values, axis=0)
    ale_values -= ale_values.mean(axis=0)
    return np.array(bin_centers), np.array(ale_values)

def plot_pdp_ale(clf, X, feature_idx, feature_name, classes, scaler=None):
    grid, pdp = compute_pdp(clf, X, feature_idx, scaler=scaler)
    centers, ale = compute_ale(clf, X, feature_idx, scaler=scaler)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, cls in enumerate(classes):
        ax1.plot(grid, pdp[:, i], label=cls, marker='')
        ax2.plot(centers, ale[:, i], label=cls, marker='')
    ax1.set_title(f'PDP - {feature_name}')
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel('Average predicted probability')
    ax1.legend()
    ax2.set_title(f'ALE - {feature_name}')
    ax2.set_xlabel(feature_name)
    ax2.set_ylabel('Effect')
    ax2.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img

def index(request):
    features = ['bill_length_mm', 'bill_depth_mm',
                'flipper_length_mm', 'body_mass_g']
    lam = float(request.GET.get('lam', 0.0))
    model_type = request.GET.get('model_type', 'tree')
    plot_feature = request.GET.get('plot_feature', 'bill_length_mm')
    df = get_penguins()
    X = df[features].values

    if model_type == 'tree':
        best = best_tree_for_lambda(lam)
        clf = best['clf']
        scaler = None
        tree_img = plot_decision_tree(clf, features)
        context = {
            'tree_img': tree_img,
            'test_acc': best['test_acc'],
            'n_leaves': best['n_leaves'],
            'model_type': model_type,
            'lam': lam,
        }
    else:
        best = best_logreg_for_lambda(lam)
        clf = best['clf']
        scaler = best['scaler']
        context = {
            'test_acc': best['test_acc'],
            'omega': round(best['omega'], 3),
            'C': round(best['C'], 4),
            'model_type': model_type,
            'lam': lam,
        }

    feat_idx = features.index(plot_feature)
    effect_img = plot_pdp_ale(clf, X, feat_idx,
                               plot_feature, clf.classes_,
                               scaler=scaler)

    context['species_list'] = ['Adelie', 'Chinstrap', 'Gentoo']
    context['n_examples'] = len(df)
    context['effect_img'] = effect_img
    context['plot_feature'] = plot_feature
    context['features'] = features

    if request.method == 'POST' and request.POST.get('action') == 'counterfactual':
        lam = float(request.POST.get('lam', 0.0))
        model_type = request.POST.get('model_type', 'tree')
        idx = int(request.POST.get('example_idx', 0))
        target = request.POST.get('target_label')
        x = df[features].iloc[idx].values
        current_pred_input = scaler.transform([x]) if scaler else [x]
        current_pred = clf.predict(current_pred_input)[0]
        cfs = generate_counterfactuals(
            clf, x, target, features, df, scaler=scaler)
        context['original'] = list(zip(features, [round(v, 2) for v in x]))
        context['counterfactuals'] = [
            list(zip(features, [round(v, 2) for v in cf[1]]))
            for cf in cfs
        ]
        context['target'] = target
        context['current_pred'] = current_pred
        context['example_idx'] = idx
        context['cf_found'] = len(cfs) > 0

    return render(request, 'project2/index.html', context)