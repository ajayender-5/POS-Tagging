import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
# Set page configuration to full width
st.set_page_config(page_title="POS Tagging Application", page_icon="🏷️")
# Load the trained model
model = tf.keras.models.load_model('C:\\Users\\madas\\Data Science 255 - Batch\\Deep_Learning\\RNN\\pos_tagger_model.keras')

# Load the word tokenizer
with open('C:\\Users\\madas\\Data Science 255 - Batch\\Deep_Learning\\RNN\\word_tokenizer.pkl', 'rb') as f:
    word_tokenizer = pickle.load(f)

# Load the tag tokenizer
with open('C:\\Users\\madas\\Data Science 255 - Batch\\Deep_Learning\\RNN\\tag_tokenizer.pkl', 'rb') as f:
    tag_tokenizer = pickle.load(f)

def predict_pos(text):
    # Tokenize and pad the input text
    words = text.split()
    word_sequences = word_tokenizer.texts_to_sequences([words])
    sequences_padded = pad_sequences(word_sequences, maxlen=271, padding="post")
    
    # Predict POS tags
    predictions = model.predict(sequences_padded)[0]
    predicted_indices = np.argmax(predictions, axis=1)
    
    # Convert predicted indices to tag names
    predicted_tags = tag_tokenizer.sequences_to_texts([predicted_indices])
    predicted_tags = predicted_tags[0].split()  # Split the tags into a list
    
    # Create the output in the "word: TAG" format
    result = [f"{word}: {tag.upper()}" for word, tag in zip(words, predicted_tags)]
    
    return result

# Streamlit app
st.title(":rainbow[POS Tagging Application]")
st.write("Enter text below to see the Parts-of-Speech tags.")

user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input:
        # Get the prediction
        prediction = predict_pos(user_input)

        # Display the result
        st.write("### Predicted POS Tags:")
        for line in prediction:
            st.write(line)  # Display each word: TAG pair on a new line
    else:
        st.warning("Please enter some text to predict.")
