# Automated Next-Word Prediction Engine using Deep Recurrent Networks (LSTMs) 🕵️‍♂️

## Project Overview
Developed a generative, auto-regressive text forecasting system designed to predict the most statistically probable upcoming word given a variable-length text prompt sequence. The core engine frames language generation as a supervised multi-class classification problem, mimicking modern productivity autocompletion features like smartphone smart keyboards and Gmail’s Smart Compose framework.

This specific iteration of the engine is trained on the corpus of *The Adventures of Sherlock Holmes* by Arthur Conan Doyle, enabling the model to predict vocabulary, syntax, and phrasing in the style of the classic detective novels. The project features a fully interactive web interface built with Streamlit for real-time text generation.

## Technical Stack & Tools
*   **Deep Learning Framework:** TensorFlow / Keras
*   **Core Architecture:** Long Short-Term Memory (LSTM) Networks, Trainable Word Embeddings
*   **Vectorization & Parsing:** Keras Tokenizer API, NumPy, One-Hot Encoding
*   **Web Deployment:** Streamlit
*   **Development Environment:** Python, Google Colab

---

## Core Engineering Pipeline & Architecture

### 1. Contiguous N-Gram Feature Ingestion
Transformed raw, unstructured text corpora into a structured supervised dataset. Created a rolling window feature-extraction pipeline that maps a sequence of token indices as the training feature input (`X`) and explicitly isolates the immediate trailing token as the discrete categorical label target (`y`).

### 2. Sequential Data Alignment & Tokenization
Implemented a text preprocessing sequence utilizing the Keras Tokenizer to map string inputs into compact integer coordinate configurations. Standardized volatile, variable-length text inputs by computing global data length boundaries and implementing Pre-Zero Padding (`pad_sequences`) to ensure perfectly stabilized input tensor vectors.

### 3. Trainable Dense Embedding Projection
Bypassed high-memory sparse structural matrices by running inputs through an Embedding layer bottleneck. This dynamically mapped one-hot integer tokens into dense 100-dimensional continuous floating-point spaces, optimizing geometric semantic closeness across matching semantic vocabularies.

### 4. Recurrent Hidden State Memory Loop
Constructed a core sequential layer utilizing 150 LSTM hidden nodes. By separating memory tracking into volatile short-term hidden states (`h_t`) and linear addition cell states (`C_t`), the architecture effectively bypassed the catastrophic temporal vanishing gradient tracking decays common to standard SimpleRNNs, preserving deep semantic language structures over long sentences.

### 5. Multi-Class Softmax Classification Head
Mapped output tensors through a Dense matrix volume calibrated to the absolute size of the vocabulary. Utilized a Softmax activation function to project continuous network outputs into a legitimate probability distribution vector, isolating the single highest-probability token index match via sparse categorical index parsing (`np.argmax`).

### 6. Auto-Regressive Inference Engine
Engineered an auto-regressive text generation algorithm. The inference pipeline predicts a single immediate word match, appends the newly computed token string back into the historical prompt sentence buffer, and recursively re-pads and cycles the updated string through the network graph to accurately generate long, context-aware sentences word-by-word.

---

## Dataset
*   **Source:** [Next Word Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/malik12345/next-word-prediction-dataset)
*   **Corpus:** *The Adventures of Sherlock Holmes*
*   **Processing:** Converted to lowercase, stripped of special characters, and tokenized into distinct integer mappings representing the global vocabulary of the book.

---

## How to Run the Web Application

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/yourusername/Sherlock-LSTM-Next-Word-Predictor.git](https://github.com/yourusername/Sherlock-LSTM-Next-Word-Predictor.git)
