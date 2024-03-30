from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('form/', views.PredictForm, name='form'),
    path('result/', views.result, name='result'),
]
