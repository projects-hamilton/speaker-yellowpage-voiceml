from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .ml_model import predict_answer


class FAQAPIView(APIView):
    def post(self, request):
        question = request.data.get('question')
        if not question:
            return Response({'error': 'Question field is required.'}, status=status.HTTP_400_BAD_REQUEST)

        # Generate prediction
        answer = predict_answer(question)
        return Response({'question': question, 'answer': answer})
