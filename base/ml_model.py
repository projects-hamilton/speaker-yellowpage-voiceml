import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import nltk

# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Paths to the model and associated files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR, "BASE_DIR")
MODEL_PATH = os.path.join(BASE_DIR, 'faq_model.h5')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

# Load model, vectorizer, and label encoder
model = load_model(MODEL_PATH)
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)

# Preprocessing function
lemmatizer = nltk.stem.WordNetLemmatizer()


def preprocess(sentence):
    words = nltk.word_tokenize(sentence.lower())
    return " ".join([lemmatizer.lemmatize(w) for w in words if w.isalnum()])

# Prediction function


def predict_answer(question):
    processed_question = preprocess(question)
    input_vector = vectorizer.transform([processed_question]).toarray()
    predictions = model.predict(input_vector)
    max_index = np.argmax(predictions)
    confidence = predictions[0][max_index]

    if confidence > 0.3:  # Adjust confidence threshold as needed
        return label_encoder.inverse_transform([max_index])[0]
    else:
        return "I'm sorry, I couldn't understand your question. Could you rephrase?"
