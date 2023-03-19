from django.urls import path
from .views import check_face
urlpatterns = [
    path("",check_face),
]
