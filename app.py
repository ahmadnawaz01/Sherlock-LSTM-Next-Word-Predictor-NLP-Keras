import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from st_keyup import st_keyup  # Import the new real-time component

# 1. Load assets and cache them for performance
@st.cache_resource
def load_assets():
    model = load_model('next_word_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_assets()

# The maximum sequence length used during training 
# (Extracting it dynamically from the model's input layer)
max_sequence_len = model.input_shape[1] + 1 

# 2. UI Layout
st.title("Sherlock Holmes: Next Word Predictor 🕵️")
st.write("Start typing to see real-time LSTM predictions.")

# 3. Replace st.text_input with st_keyup for live typing updates
input_text = st_keyup("Enter your text here:")

# 4. Remove the button. The app will now automatically run this block every time a key is pressed.
if input_text:
    # Preprocess user input
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    
    # We add a quick check to ensure the token list isn't empty 
    # (e.g., if the user types a word not in the vocabulary yet)
    if len(token_list) > 0:
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Predict
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        
        # Reverse lookup: Token ID to Word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
                
        st.success(f"Predicted next word: **{output_word}**")
        st.write(f"**Full sequence:** {input_text} {output_word}")