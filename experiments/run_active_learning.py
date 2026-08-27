import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbl.settings')
import django
django.setup()

from project3.ml_core import (get_or_train_baseline, load_ag_news,
    get_expert_labels, active_learning_loop, random_sampling_loop)

clf, vectorizer = get_or_train_baseline()
train_texts, train_labels, test_texts, test_labels = load_ag_news()
expert_labels_test = get_expert_labels(test_labels)

X_test = vectorizer.transform(test_texts)

print("=== Uncertainty sampling ===")
queried_unc, history_unc = active_learning_loop(
    clf, vectorizer, train_texts[:5000], train_labels[:5000],
    X_test, test_labels, expert_labels_test)
for h in history_unc:
    print(h)

print("\n=== Random sampling ===")
queried_rand, history_rand = random_sampling_loop(
    clf, vectorizer, train_texts[:5000], train_labels[:5000],
    X_test, test_labels, expert_labels_test)
for h in history_rand:
    print(h)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n_queried_unc = [h['n_queried'] for h in history_unc]
accs_unc = [h['system_accuracy'] for h in history_unc]
n_queried_rand = [h['n_queried'] for h in history_rand]
accs_rand = [h['system_accuracy'] for h in history_rand]

plt.figure()
plt.plot(n_queried_unc, accs_unc, marker='o', label='Uncertainty sampling')
plt.plot(n_queried_rand, accs_rand, marker='s', label='Random sampling')
plt.axhline(y=0.9153, color='gray', linestyle='--', label='Baseline (no deferral)')
plt.xlabel('Number of expert queries')
plt.ylabel('System accuracy')
plt.title('Active learning efficiency: uncertainty vs random sampling')
plt.legend()
plt.savefig('experiments/active_learning_comparison.png', bbox_inches='tight')
print("Saved active learning comparison plot")