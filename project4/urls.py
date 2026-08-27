from django.urls import path
from . import views

app_name = 'project4'
urlpatterns = [
    path('', views.index, name='index'),
    path('consent/', views.consent, name='consent'),
    path('instructions/', views.instructions, name='instructions'),
    path('study/', views.study, name='study'),
    path('results/', views.results, name='results'),
    path('thankyou/', views.thankyou, name='thankyou'),
]