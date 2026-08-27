import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
PROCESSED_PATH = os.path.join(DATA_DIR, 'processed_movies.csv')
RAW_PATH = os.path.join(DATA_DIR, 'movie_metadata.csv')

META_COLS = ['movie_title', 'genres', 'duration',
             'title_year', 'language', 'country', 'content_rating']


def get_movie_data():
    df = pd.read_csv(PROCESSED_PATH)
    titles = df['movie_title'].values
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].values.astype(float)
    return df, X, titles, feature_cols


def get_random_pair(exclude_indices=None):
    df, X, titles, _ = get_movie_data()
    available = list(range(len(titles)))
    if exclude_indices:
        available = [i for i in available if i not in exclude_indices]
    idx = np.random.choice(available, size=2, replace=False)
    movies = []
    for i in idx:
        row = df.iloc[i]
        movies.append({
            'index': int(i),
            'title': titles[i],
            'genres': row.get('genres', ''),
            'year': int(row.get('title_year', 0)),
            'duration': int(row.get('duration', 0)),
        })
    return movies


def get_random_set(n=10, exclude_indices=None):
    df, X, titles, _ = get_movie_data()
    available = list(range(len(titles)))
    if exclude_indices:
        available = [i for i in available if i not in exclude_indices]
    idx = np.random.choice(available, size=n, replace=False)
    movies = []
    for i in idx:
        row = df.iloc[i]
        movies.append({
            'index': int(i),
            'title': titles[i],
            'genres': row.get('genres', ''),
            'year': int(row.get('title_year', 0)),
            'duration': int(row.get('duration', 0)),
        })
    return movies

from scipy.special import logsumexp
from scipy.optimize import minimize


def pairwise_log_likelihood(w, xi, xj):
    ui = w @ xi
    uj = w @ xj
    return ui - np.logaddexp(ui, uj)


def ranking_log_likelihood(w, ranked_X):
    utilities = ranked_X @ w
    total_ll = 0
    for k in range(len(ranked_X) - 1):
        remaining_utils = utilities[k:]
        total_ll += utilities[k] - logsumexp(remaining_utils)
    return total_ll


def estimate_preferences_pairwise(comparisons, n_features):
    if len(comparisons) == 0:
        return np.zeros(n_features)

    def neg_ll(w):
        ll = 0
        for xi, xj in comparisons:
            ll += pairwise_log_likelihood(w, xi, xj)
        return -ll

    w0 = np.zeros(n_features)
    result = minimize(neg_ll, w0, method='L-BFGS-B')
    return result.x


def estimate_preferences_ranking(rankings, n_features):
    if len(rankings) == 0:
        return np.zeros(n_features)

    def neg_ll(w):
        ll = sum(ranking_log_likelihood(w, ranked_X)
                 for ranked_X in rankings)
        return -ll

    w0 = np.zeros(n_features)
    result = minimize(neg_ll, w0, method='L-BFGS-B')
    return result.x


def recommend_movies(w, X, titles, df, top_k=5, exclude_indices=None):
    scores = X @ w
    if exclude_indices:
        scores = scores.copy()
        for i in exclude_indices:
            scores[i] = -np.inf
    top_indices = np.argsort(scores)[::-1][:top_k]
    recommendations = []
    for i in top_indices:
        row = df.iloc[i]
        recommendations.append({
            'index': int(i),
            'title': titles[i],
            'genres': row.get('genres', ''),
            'year': int(row.get('title_year', 0)),
            'duration': int(row.get('duration', 0)),
            'score': round(float(scores[i]), 4),
        })
    return recommendations