import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pbl.settings')
import django
django.setup()

from project3.ml_core import get_or_train_baseline

clf, vectorizer = get_or_train_baseline()
print("Model ready (trained or loaded from cache)")