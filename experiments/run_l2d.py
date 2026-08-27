import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbl.settings')
import django
django.setup()

from project3.ml_core import (get_or_train_baseline, load_ag_news,
    get_expert_labels, evaluate_l2d_system)
import numpy as np

clf, vectorizer = get_or_train_baseline()
_, _, test_texts, test_labels = load_ag_news()
X_test = vectorizer.transform(test_texts)
expert_labels = get_expert_labels(test_labels)

for threshold in np.linspace(0.3, 0.9, 7):
    results = evaluate_l2d_system(clf, X_test, test_labels,
                                   expert_labels, threshold)
    print(f"threshold={threshold:.2f}: {results}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

thresholds = np.linspace(0.1, 0.95, 15)
accs, defer_rates = [], []
for t in thresholds:
    r = evaluate_l2d_system(clf, X_test, test_labels, expert_labels, t)
    accs.append(r['system_accuracy'])
    defer_rates.append(r['defer_rate'])

plt.figure()
plt.plot(defer_rates, accs, marker='o')
plt.axhline(y=0.9153, color='gray', linestyle='--', label='Baseline (no deferral)')
plt.xlabel('Deferral rate')
plt.ylabel('System accuracy')
plt.title('Accuracy vs deferral rate (Learning to Defer)')
plt.legend()
plt.savefig('experiments/l2d_tradeoff.png', bbox_inches='tight')
print("Saved tradeoff plot")