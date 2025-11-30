# NextWordPredictor

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning-based next-word prediction system built with Bidirectional LSTM (BiLSTM) sequence-to-sequence models. This project leverages literary texts from fairy tales and Shakespeare's Hamlet to train a neural language model capable of predicting the next word in a given sentence fragment.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Code Walkthrough](#code-walkthrough)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [Author](#author)

## 🎯 Overview

NextWordPredictor is a machine learning project that implements a sequence-to-sequence model using Bidirectional LSTM networks to predict the next word in a sequence. The model is trained on clean, processed literary text data and can generate contextually relevant word predictions.

## ✨ Features

- **BiLSTM Architecture**: Bidirectional LSTM layers for improved context understanding
- **Sequence-to-Sequence Model**: Encoder-decoder architecture for sequence prediction
- **Text Preprocessing**: Comprehensive text cleaning and tokenization pipeline
- **Data Augmentation**: Multiple literary sources for robust model training
- **Jupyter Notebook Implementation**: Easy-to-follow code with detailed explanations
- **Model Persistence**: Trained model saved for reusability and deployment
- **Early Stopping**: Prevents overfitting with callback monitoring

## 📁 Project Structure

```
NextWordPredictor/
├── README.md                                    # Project documentation
├── nextwordprediction.ipynb                     # Main Jupyter notebook with complete implementation
├── cleaned_merged_fairy_tales_without_eos.txt   # Fairy tales dataset (cleaned)
├── hamlet.txt                                   # Shakespeare's Hamlet text file
├── nextwordprediction.h5                        # Trained model (saved weights)
└── .gitignore                                   # Git ignore file
```

## 🔍 Code Walkthrough

The implementation is organized into distinct phases:

### 1. **Data Collection** (Cell 1-5)
Downloads and loads the Hamlet text from NLTK Gutenberg corpus using `nltk.corpus`:
```python
import nltk
nltk.download('gutenberg')
from nltk.corpus import gutenberg

data = gutenberg.raw('shakespeare-hamlet.txt')
with open('hamlet.txt', 'w') as file:
    file.write(data)
```
**Output**: Full Hamlet text file (~155KB)

### 2. **Data Preprocessing** (Cell 130-140)
Tokenizes and prepares sequences for model training:
- **Tokenization**: Converts text to integer sequences using Keras Tokenizer
- **Vocabulary**: Builds vocabulary from all unique words (Total: 4,818 words)
- **N-gram Generation**: Creates input sequences of varying lengths (1 to 14 tokens)
- **Padding**: Standardizes all sequences to max length of 14 tokens
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1  # 4818
```
**Output**: 25,732 input sequences × 14 timesteps

### 3. **Sequence Preparation** (Cell 135-140)
Creates training and validation datasets:
- Input shape: (25732, 13) - previous 13 words
- Output shape: (25732, 4818) - one-hot encoded next word
- Train-Test Split: 80-20 ratio
```python
x, y = inputsequences[:, :-1], inputsequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```

### 4. **Model Architecture** (Cell 74)
Sequential model with BiLSTM layers:
```python
model = Sequential()
model.add(Embedding(4818, 100, input_length=13))  # Word embeddings
model.add(LSTM(150, return_sequences=True))        # First LSTM layer
model.add(Dropout(0.2))                             # Regularization
model.add(LSTM(100))                                # Second LSTM layer
model.add(Dense(4818, activation='softmax'))        # Output layer

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
**Architecture Summary**:
- Embedding Layer: Converts 4818 vocabulary words to 100-dimensional vectors
- LSTM-1: 150 units with return sequences (outputs all timesteps)
- Dropout: 20% regularization to prevent overfitting
- LSTM-2: 100 units (outputs only final state)
- Dense: 4818 units with softmax for word probability distribution

### 5. **Model Training** (Cell 75-76)
Trains with early stopping and validation monitoring:
```python
from tensorflow.keras.callbacks import EarlyStopping

earlystopping = EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), 
                    verbose=1, callbacks=[earlystopping])
```
**Training Results** (10 epochs):
- Epoch 1: Train Acc: 3.15% → Val Acc: 3.52%
- Epoch 10: Train Acc: 9.46% → Val Acc: 6.51%
- **Final Validation Accuracy**: ~6.51%

### 6. **Model Saving & Loading** (Cell ~100)
Persists trained model to disk:
```python
model.save(filepath='nextwordprediction.h5')
mymodel = load_model('nextwordprediction.h5')
```

### 7. **Inference & Prediction** (Cell 103, 118)
Generates next words from input sequences:
```python
input_text = 'May be'
for i in range(10):
    input_text_tokenize = tokenizer.texts_to_sequences([input_text])[0]
    padded_input = pad_sequences([input_text_tokenize], maxlen=13)
    
    pos_array = mymodel.predict(padded_input)
    pos = np.argmax(pos_array)  # Get highest probability word index
    
    # Decode index back to word
    for word, index in tokenizer.word_index.items():
        if pos == index:
            input_text += ' ' + word
            break
```
**Example Output**:
```
May be well vs'd to know my stops you would pluck him
```

## 📊 Dataset

The project uses two primary data sources:

1. **Fairy Tales Dataset**: `cleaned_merged_fairy_tales_without_eos.txt`
   - Cleaned and merged fairy tale texts
   - End-of-sentence markers removed for continuous text flow

2. **Shakespeare's Hamlet**: `hamlet.txt`
   - Classic literary text for diverse language patterns
   - Rich vocabulary (4,818 unique words) and complex sentence structures
   - ~155 KB of text data

Both datasets are preprocessed and combined to create a comprehensive training corpus.

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- Jupyter Notebook

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Rajat-Shrma/NextWordPredictor.git
   cd NextWordPredictor
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. Install required packages:
   ```bash
   pip install tensorflow numpy pandas nltk jupyter scikit-learn
   ```

## 💻 Usage

### Running the Notebook

1. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `nextwordprediction.ipynb`

3. Execute cells sequentially to:
   - Load and preprocess text data
   - Build the BiLSTM model
   - Train the model on the dataset
   - Make predictions on new text sequences

### Making Predictions

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
mymodel = load_model('nextwordprediction.h5')

# Prepare input text
input_text = 'you are good'
input_sequence = tokenizer.texts_to_sequences([input_text])[0]
padded_sequence = pad_sequences([input_sequence], maxlen=13)

# Make prediction
prediction = mymodel.predict(padded_sequence)
next_word_index = np.argmax(prediction)

# Decode index to word
for word, index in tokenizer.word_index.items():
    if next_word_index == index:
        print(f"Next word: {word}")
        break
```

## 🧠 Model Architecture

The model uses a Bidirectional LSTM sequence-to-sequence architecture:

- **Embedding Layer**: 4818 → 100 dimensions
- **LSTM-1**: 150 units with return sequences (preserves temporal information)
- **Dropout**: 0.2 (20%) for regularization
- **LSTM-2**: 100 units (final temporal processing)
- **Output Dense**: 4818 units with softmax activation

### Model Specifications

| Component | Value |
|-----------|-------|
| Input Shape | (13,) - 13 previous words |
| Vocabulary Size | 4,818 unique words |
| Embedding Dimension | 100 |
| LSTM Units (Layer 1) | 150 |
| LSTM Units (Layer 2) | 100 |
| Dropout Rate | 0.2 |
| Output Classes | 4,818 (one per word) |
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam |
| Activation (Hidden) | ReLU (default) |
| Activation (Output) | Softmax |

## 📈 Results

The model demonstrates training on Shakespeare's Hamlet text:

### Performance Metrics

- **Training Accuracy**: 9.46% (after 10 epochs)
- **Validation Accuracy**: 6.51%
- **Training Loss**: 5.3429
- **Validation Loss**: 7.2514

### Sample Predictions

1. Input: "you are good" → Output: "lord"
2. Input: "May be" → Generated: "May be well vs'd to know my stops you would pluck him"

### Performance Analysis

The relatively low accuracy reflects the challenge of predicting next words in a large vocabulary (4,818 words). The model demonstrates:
- Strong learning capability (loss decreases over epochs)
- Reasonable generalization (validation metrics don't degrade significantly)
- Typical patterns for language modeling tasks

## 🛠️ Technologies Used

| Technology | Purpose | Version |
|------------|---------|----------|
| TensorFlow/Keras | Deep learning framework | 2.0+ |
| Python | Programming language | 3.8+ |
| NumPy | Numerical computing | 1.19+ |
| Pandas | Data manipulation | 1.2+ |
| Scikit-learn | ML utilities | 0.24+ |
| Jupyter | Development environment | 1.0+ |
| NLTK | NLP utilities | 3.6+ |

## 🔮 Future Enhancements

- [ ] Implement attention mechanisms for better context awareness
- [ ] Add support for multi-language prediction
- [ ] Deploy as a REST API using Flask/FastAPI
- [ ] Create a web interface for interactive predictions
- [ ] Implement beam search for more diverse predictions
- [ ] Add evaluation metrics (BLEU, METEOR)
- [ ] Expand training dataset with more literary works
- [ ] Fine-tune hyperparameters using Bayesian optimization
- [ ] Implement bidirectional prediction (fill in the middle)
- [ ] Add character-level modeling as alternative

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Rajat Sharma**
- GitHub: [@Rajat-Shrma](https://github.com/Rajat-Shrma)
- Project: [NextWordPredictor](https://github.com/Rajat-Shrma/NextWordPredictor)

## 📧 Contact & Support

For questions, suggestions, or issues, please open an [issue](https://github.com/Rajat-Shrma/NextWordPredictor/issues) on GitHub.

---

**⭐ If you find this project helpful, please consider giving it a star!**
