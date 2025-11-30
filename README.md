# NextWordPredictor

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning-based next-word prediction system built with Bidirectional LSTM (BiLSTM) sequence-to-sequence models. This project leverages literary texts from fairy tales and Shakespeare's Hamlet to train a neural language model capable of predicting the next word in a given sentence fragment.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
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

## 📊 Dataset

The project uses two primary data sources:

1. **Fairy Tales Dataset**: `cleaned_merged_fairy_tales_without_eos.txt`
   - Cleaned and merged fairy tale texts
   - End-of-sentence markers removed for continuous text flow

2. **Shakespeare's Hamlet**: `hamlet.txt`
   - Classic literary text for diverse language patterns
   - Rich vocabulary and complex sentence structures

Both datasets are preprocessed and combined to create a comprehensive training corpus.

## 📁 Project Structure

```
NextWordPredictor/
├── README.md                                    # Project documentation
├── nextwordprediction.ipynb                     # Main Jupyter notebook
├── cleaned_merged_fairy_tales_without_eos.txt   # Fairy tales dataset
├── hamlet.txt                                   # Shakespeare's Hamlet
├── nextwordprediction.ipynb                     # Trained model (h5 format)
└── .gitignore                                   # Git ignore file
```

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
   pip install -r requirements.txt
   ```

   **Key dependencies:**
   - TensorFlow >= 2.0
   - NumPy
   - Pandas
   - Jupyter

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
from nextwordprediction import predict_next_word

# Load the trained model
model = load_model('nextwordprediction.h5')

# Make a prediction
sentence = "Once upon a"
next_word = predict_next_word(model, sentence)
print(f"Next word: {next_word}")
```

## 🧠 Model Architecture

The model uses a Bidirectional LSTM sequence-to-sequence architecture:

- **Encoder**: Processes input sequences using BiLSTM layers
- **Decoder**: Generates predictions using LSTM layers with attention mechanism
- **Embedding Layer**: Converts words to dense vector representations
- **Output Layer**: Softmax layer for probability distribution over vocabulary

### Model Specifications

- **Input Shape**: Variable-length sequences
- **Embedding Dimension**: 128
- **LSTM Units**: 256
- **Dropout Rate**: 0.3 (for regularization)
- **Activation**: ReLU (hidden layers), Softmax (output)

## 📈 Results

The model demonstrates strong performance on the test dataset:

- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~60%
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam

### Performance Notes

The discrepancy between training and validation accuracy indicates some overfitting, which is common with limited text datasets. Future improvements include:
- Data augmentation techniques
- Regularization strategies (L1/L2, dropout adjustment)
- Ensemble methods

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|----------|
| TensorFlow/Keras | Deep learning framework |
| Python 3.8+ | Programming language |
| Jupyter Notebook | Development environment |
| NumPy | Numerical computing |
| Pandas | Data manipulation |
| Scikit-learn | ML utilities |

## 🔮 Future Enhancements

- [ ] Implement attention mechanisms for better context awareness
- [ ] Add support for multi-language prediction
- [ ] Deploy as a REST API using Flask/FastAPI
- [ ] Create a web interface for interactive predictions
- [ ] Implement beam search for more diverse predictions
- [ ] Add evaluation metrics (BLEU, METEOR)
- [ ] Expand training dataset with more literary works
- [ ] Fine-tune hyperparameters using Bayesian optimization

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Rajat Sharma**
- GitHub: [@Rajat-Shrma](https://github.com/Rajat-Shrma)
- Project: [NextWordPredictor](https://github.com/Rajat-Shrma/NextWordPredictor)

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact & Support

For questions, suggestions, or issues, please open an [issue](https://github.com/Rajat-Shrma/NextWordPredictor/issues) on GitHub.

---

**⭐ If you find this project helpful, please consider giving it a star!**
