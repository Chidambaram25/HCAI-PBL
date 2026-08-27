import random
import numpy as np
from django.shortcuts import render, redirect
from .ml_core import (get_movie_data, get_random_pair, get_random_set,
                      estimate_preferences_pairwise,
                      estimate_preferences_ranking, recommend_movies)

MAX_ROUNDS_PAIRWISE = 10
MAX_ROUNDS_RANKING = 2


def index(request):
    return render(request, 'project4/index.html', {})


def consent(request):
    if request.method == 'POST':
        if request.POST.get('consent') == 'yes':
            request.session['consented'] = True
            request.session['design'] = random.choice(['pairwise', 'ranking'])
            request.session['interactions'] = []
            request.session['round'] = 0
            request.session['seen_indices'] = []
            return redirect('project4:instructions')
        else:
            return render(request, 'project4/consent.html',
                          {'declined': True})
    return render(request, 'project4/consent.html', {})


def instructions(request):
    if not request.session.get('consented'):
        return redirect('project4:consent')
    design = request.session.get('design')
    return render(request, 'project4/instructions.html',
                  {'design': design})


def study(request):
    if not request.session.get('consented'):
        return redirect('project4:consent')

    if request.method == 'POST':
        design = request.session['design']
        interactions = request.session.get('interactions', [])
        seen = request.session.get('seen_indices', [])

        if design == 'pairwise':
            winner = request.POST.get('winner')
            loser = request.POST.get('loser')
            if winner and loser:
                interactions.append(('pairwise', int(winner), int(loser)))
                seen.extend([int(winner), int(loser)])

        elif design == 'ranking':
            order_str = request.POST.get('ranking_order', '')
            if order_str:
                order = [int(i) for i in order_str.split(',')
                         if i.strip().isdigit()]
                interactions.append(('ranking', order))
                seen.extend(order)

        request.session['interactions'] = interactions
        request.session['seen_indices'] = seen
        request.session['round'] = request.session.get('round', 0) + 1

    design = request.session['design']
    round_num = request.session.get('round', 0)
    seen = request.session.get('seen_indices', [])
    max_rounds = (MAX_ROUNDS_PAIRWISE if design == 'pairwise'
                  else MAX_ROUNDS_RANKING)

    if round_num >= max_rounds:
        return redirect('project4:results')

    if design == 'pairwise':
        movies = get_random_pair(exclude_indices=seen)
    else:
        movies = get_random_set(n=10, exclude_indices=seen)

    progress = int((round_num / max_rounds) * 100)

    context = {
        'design': design,
        'round': round_num + 1,
        'max_rounds': max_rounds,
        'movies': movies,
        'progress': progress,
    }
    return render(request, 'project4/study.html', context)


def results(request):
    if not request.session.get('consented'):
        return redirect('project4:consent')

    df, X, titles, _ = get_movie_data()
    n_features = X.shape[1]
    interactions = request.session.get('interactions', [])
    design = request.session.get('design', 'pairwise')

    if design == 'pairwise':
        pairs = [(X[wi], X[li]) for _, wi, li in interactions]
        w = estimate_preferences_pairwise(pairs, n_features)
    else:
        ranked_Xs = [np.array([X[i] for i in order])
                     for _, order in interactions]
        w = estimate_preferences_ranking(ranked_Xs, n_features)

    seen = request.session.get('seen_indices', [])
    recs = recommend_movies(w, X, titles, df, top_k=5,
                            exclude_indices=seen)

    if request.method == 'POST':
        request.session['feedback'] = {
            'ease': request.POST.get('ease', ''),
            'satisfaction': request.POST.get('satisfaction', ''),
            'confidence': request.POST.get('confidence', ''),
            'ratings': request.POST.getlist('rating'),
        }
        request.session.flush()
        return redirect('project4:thankyou')

    context = {
        'recommendations': recs,
        'design': design,
    }
    return render(request, 'project4/results.html', context)


def thankyou(request):
    return render(request, 'project4/thankyou.html', {})