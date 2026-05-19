import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def make_plot(df, col_x, col_y):
    fig, ax = plt.subplots()
    last_col = df.columns[-1]
    if df[last_col].nunique() <= 10:
        for label in df[last_col].unique():
            sub = df[df[last_col] == label]
            ax.scatter(sub[col_x], sub[col_y], label=str(label))
        ax.legend()
    else:
        ax.scatter(df[col_x], df[col_y])
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_b64


def make_curve_plot(train_scores, test_scores, hp_range, model_name):
    fig, ax = plt.subplots()
    ax.plot(hp_range, train_scores, label='Train accuracy', marker='o')
    ax.plot(hp_range, test_scores, label='Test accuracy', marker='o')
    ax.set_xlabel('Hyperparameter value')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Learning Curve - {model_name}')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_b64


def get_model(model_name, hp):
    if model_name == 'knn':
        return KNeighborsClassifier(n_neighbors=hp)
    elif model_name == 'dtree':
        return DecisionTreeClassifier(max_depth=hp)
    elif model_name == 'svm':
        return SVC(C=hp)


def index(request):
    context = {}
    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'plot':
            try:
                df = pd.read_json(io.StringIO(request.session['csv_data']))
                context['columns'] = df.columns.tolist()
                context['preview'] = df.head().to_html()
                col_x = request.POST.get('col_x')
                col_y = request.POST.get('col_y')
                context['plot'] = make_plot(df, col_x, col_y)
            except Exception as e:
                context['error'] = f"Could not generate plot: {e}"

        elif action == 'train':
            try:
                df = pd.read_json(io.StringIO(request.session['csv_data']))
                context['columns'] = df.columns.tolist()
                context['preview'] = df.head().to_html()

                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]

                test_size = int(request.POST.get('test_size', 20))
                model_name = request.POST.get('model_name', 'knn')
                hp = int(request.POST.get('hyperparam', 5))

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42
                )

                clf = get_model(model_name, hp)
                clf.fit(X_train, y_train)
                train_score = round(accuracy_score(y_train, clf.predict(X_train)), 3)
                test_score = round(accuracy_score(y_test, clf.predict(X_test)), 3)

                hp_range = range(1, 21)
                train_scores = []
                test_scores = []
                for h in hp_range:
                    m = get_model(model_name, h)
                    m.fit(X_train, y_train)
                    train_scores.append(accuracy_score(y_train, m.predict(X_train)))
                    test_scores.append(accuracy_score(y_test, m.predict(X_test)))

                context['results'] = {
                    'model': model_name,
                    'test_size': test_size,
                    'hyperparam': hp,
                    'train_score': train_score,
                    'test_score': test_score,
                    'curve_plot': make_curve_plot(train_scores, test_scores, list(hp_range), model_name),
                }

            except Exception as e:
                context['error'] = f"Could not train model: {e}"

        elif request.FILES.get('csv_file'):
            try:
                f = request.FILES['csv_file']
                df = pd.read_csv(f)
                context['columns'] = df.columns.tolist()
                context['preview'] = df.head().to_html()
                request.session['csv_data'] = df.to_json()
            except Exception as e:
                context['error'] = f"Could not read file: {e}"

    return render(request, 'project1/index.html', context)