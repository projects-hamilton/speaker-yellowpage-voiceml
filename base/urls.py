from django.urls import path
from .views import FAQAPIView

urlpatterns = [
    path('predict/', FAQAPIView.as_view(), name='faq_predict'),
]
