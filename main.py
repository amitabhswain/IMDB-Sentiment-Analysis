# Import Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
try:
    model = load_model('simple_rnn_imdb.keras')
except Exception as e:
    st.error("Failed to load the model. Please check the file path and format.")
    raise e

# Helper Functions
def decode_review(encoded_review):
    """Decode a numeric review back into words."""
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    """Preprocess user input for the model."""
    if not text.strip():
        return None
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    if preprocessed_input is None:
        st.write('Please enter a valid movie review.')
    else:
        try:
            prediction = model.predict(preprocessed_input)
            sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

            # Display the result
            st.write(f'Sentiment: {sentiment}')
            st.write(f'Prediction Score: {prediction[0][0]:.2f}')
        except Exception as e:
            st.error("An error occurred while making the prediction.")
            st.error(str(e))
else:
    st.write('Please enter a movie review.')
