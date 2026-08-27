import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbl.settings')
import django
django.setup()

from project3.ml_core import load_ag_news, get_expert_labels, LABEL_NAMES
from sklearn.metrics import accuracy_score, confusion_matrix

_, _, test_texts, test_labels = load_ag_news()
expert_labels = get_expert_labels(test_labels)

overall_acc = accuracy_score(test_labels, expert_labels)
print(f"Overall expert accuracy: {overall_acc:.4f}")

for c in range(4):
    mask = test_labels == c
    acc_c = accuracy_score(test_labels[mask], expert_labels[mask])
    print(f"{LABEL_NAMES[c]}: accuracy = {acc_c:.4f}")

print(confusion_matrix(test_labels, expert_labels))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(test_labels, expert_labels)
disp = ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES)
disp.plot()
plt.title('Simulated expert confusion matrix')
plt.savefig('experiments/expert_confusion_matrix.png', bbox_inches='tight')
print("Saved confusion matrix plot")