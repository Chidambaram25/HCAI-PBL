from datasets import load_dataset
import numpy as np

LABEL_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']


def load_ag_news():
    ds = load_dataset("fancyzhx/ag_news")
    train_texts = ds['train']['text']
    train_labels = np.array(ds['train']['label'])
    test_texts = ds['test']['text']
    test_labels = np.array(ds['test']['label'])
    return train_texts, train_labels, test_texts, test_labels


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_baseline_classifier():
    train_texts, train_labels, test_texts, test_labels = load_ag_news()

    vectorizer = TfidfVectorizer(max_features=20000, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)

    test_acc = accuracy_score(test_labels, clf.predict(X_test))
    return clf, vectorizer, test_acc

import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)


def get_or_train_baseline():
    clf_path = os.path.join(MODEL_DIR, 'baseline_clf.joblib')
    vec_path = os.path.join(MODEL_DIR, 'baseline_vec.joblib')

    if os.path.exists(clf_path) and os.path.exists(vec_path):
        clf = joblib.load(clf_path)
        vectorizer = joblib.load(vec_path)
        return clf, vectorizer

    clf, vectorizer, test_acc = train_baseline_classifier()
    joblib.dump(clf, clf_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Trained and cached baseline. Test acc: {test_acc:.4f}")
    return clf, vectorizer

import random


def simulate_expert(true_label, rng=None):
    rng = rng or random.Random()
    strong_classes = {0, 1}
    weak_confusion = {2: 3, 3: 2}

    if true_label in strong_classes:
        if rng.random() < 0.95:
            return true_label
        return rng.choice([c for c in range(4) if c != true_label])
    else:
        if rng.random() < 0.60:
            return true_label
        return weak_confusion[true_label]
    
def get_expert_labels(true_labels, seed=42):
    rng = random.Random(seed)
    return np.array([simulate_expert(int(y), rng) for y in true_labels])

def confidence_based_defer(clf, X, threshold=0.5):
    probs = clf.predict_proba(X)
    max_probs = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    defer_mask = max_probs < threshold
    return predictions, defer_mask, max_probs

def evaluate_l2d_system(clf, X_test, y_test, expert_labels, threshold=0.5):
    predictions, defer_mask, max_probs = confidence_based_defer(
        clf, X_test, threshold)
    final_predictions = predictions.copy()
    final_predictions[defer_mask] = expert_labels[defer_mask]

    system_acc = (final_predictions == y_test).mean()
    defer_rate = defer_mask.mean()
    clf_only_acc_on_deferred = (
        (predictions[defer_mask] == y_test[defer_mask]).mean()
        if defer_mask.sum() > 0 else 0
    )
    expert_acc_on_deferred = (
        (expert_labels[defer_mask] == y_test[defer_mask]).mean()
        if defer_mask.sum() > 0 else 0
    )

    return {
        'system_accuracy': system_acc,
        'defer_rate': defer_rate,
        'clf_acc_on_deferred_cases': clf_only_acc_on_deferred,
        'expert_acc_on_deferred_cases': expert_acc_on_deferred,
    }

def active_learning_loop(clf, vectorizer, X_pool_texts, y_pool,
                          X_test, y_test, expert_labels_test,
                          n_rounds=10, queries_per_round=50):
    X_pool = vectorizer.transform(X_pool_texts)
    queried_indices = []
    history = []

    rng_expert = random.Random(123)
    remaining = list(range(len(y_pool)))
    candidate_thresholds = np.linspace(0.3, 0.95, 14)

    for round_i in range(n_rounds):
        probs = clf.predict_proba(X_pool[remaining])
        max_probs = probs.max(axis=1)
        order = np.argsort(max_probs)
        chosen_local = order[:queries_per_round]
        chosen_global = [remaining[i] for i in chosen_local]

        for idx in chosen_global:
            expert_label = simulate_expert(int(y_pool[idx]), rng_expert)
            queried_indices.append((idx, expert_label))

        chosen_set = set(chosen_global)
        remaining = [i for i in remaining if i not in chosen_set]

        best_threshold, best_acc = 0.5, -1
        for t in candidate_thresholds:
            correct = 0
            for idx, expert_label in queried_indices:
                conf = clf.predict_proba(X_pool[idx]).max()
                clf_pred = clf.predict(X_pool[idx])[0]
                true_label = y_pool[idx]
                final_pred = expert_label if conf < t else clf_pred
                correct += int(final_pred == true_label)
            acc = correct / len(queried_indices)
            if acc > best_acc:
                best_acc, best_threshold = acc, t

        results = evaluate_l2d_system(
            clf, X_test, y_test, expert_labels_test, threshold=best_threshold)
        history.append({
            'round': round_i,
            'n_queried': len(queried_indices),
            'learned_threshold': round(best_threshold, 3),
            'system_accuracy': results['system_accuracy'],
            'defer_rate': results['defer_rate'],
        })

    return queried_indices, history

def random_sampling_loop(clf, vectorizer, X_pool_texts, y_pool,
                          X_test, y_test, expert_labels_test,
                          n_rounds=10, queries_per_round=50, seed=7):
    X_pool = vectorizer.transform(X_pool_texts)
    queried_indices = []
    history = []

    rng_query = random.Random(seed)
    rng_expert = random.Random(123)
    remaining = list(range(len(y_pool)))
    candidate_thresholds = np.linspace(0.3, 0.95, 14)

    for round_i in range(n_rounds):
        chosen_global = rng_query.sample(remaining,
                                          min(queries_per_round, len(remaining)))

        for idx in chosen_global:
            expert_label = simulate_expert(int(y_pool[idx]), rng_expert)
            queried_indices.append((idx, expert_label))

        chosen_set = set(chosen_global)
        remaining = [i for i in remaining if i not in chosen_set]

        best_threshold, best_acc = 0.5, -1
        for t in candidate_thresholds:
            correct = 0
            for idx, expert_label in queried_indices:
                conf = clf.predict_proba(X_pool[idx]).max()
                clf_pred = clf.predict(X_pool[idx])[0]
                true_label = y_pool[idx]
                final_pred = expert_label if conf < t else clf_pred
                correct += int(final_pred == true_label)
            acc = correct / len(queried_indices)
            if acc > best_acc:
                best_acc, best_threshold = acc, t

        results = evaluate_l2d_system(
            clf, X_test, y_test, expert_labels_test, threshold=best_threshold)
        history.append({
            'round': round_i,
            'n_queried': len(queried_indices),
            'learned_threshold': round(best_threshold, 3),
            'system_accuracy': results['system_accuracy'],
            'defer_rate': results['defer_rate'],
        })

    return queried_indices, history

def estimate_expert_competence_by_confidence(clf, vectorizer,
                                              queried_indices, X_pool_texts,
                                              y_pool, n_bins=5):
    X_pool = vectorizer.transform(X_pool_texts)
    confidences = []
    expert_correct = []

    for idx, expert_label in queried_indices:
        true_label = y_pool[idx]
        conf = clf.predict_proba(X_pool[idx]).max()
        confidences.append(conf)
        expert_correct.append(int(expert_label == true_label))

    confidences = np.array(confidences)
    expert_correct = np.array(expert_correct)

    bin_edges = np.linspace(confidences.min(), confidences.max(), n_bins + 1)
    bin_competence = []
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_competence.append(expert_correct[mask].mean())
        else:
            bin_competence.append(None)

    return bin_edges, bin_competence