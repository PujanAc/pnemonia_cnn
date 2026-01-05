"""
URL configuration for detector app.
"""

from django.urls import path
from . import views

app_name = 'detector'

urlpatterns = [
    path('', views.home, name='home'),
    path('select-model/', views.select_model, name='select_model'),
    path('upload/<str:model_type>/', views.upload_image, name='upload'),
    path('predict/', views.predict, name='predict'),
    path('result/<int:prediction_id>/', views.result, name='result'),
    path('compare/', views.compare_mode, name='compare'),
    path('compare-predict/', views.compare_predict, name='compare_predict'),
]