from django.shortcuts import render


def index(request):
    context = {
        'baseline_acc': 0.9153,
        'expert_overall_acc': 0.7776,
        'expert_world_acc': 0.9511,
        'expert_sports_acc': 0.9542,
        'expert_business_acc': 0.5979,
        'expert_scitech_acc': 0.6074,
        'l2d_best_threshold': 0.6,
        'l2d_best_acc': 0.9259,
        'al_uncertainty_acc': 0.9252,
        'al_random_acc': 0.9185,
    }
    return render(request, 'project3/index.html', context)