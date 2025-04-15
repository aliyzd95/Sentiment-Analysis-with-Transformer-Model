# Sentiment Analysis with Transformer Model

This project implements a **Transformer-based model** for sentiment analysis, aiming to classify text based on the sentiment expressed in it. The model is trained using the **SST-2 dataset** from the GLUE benchmark.

### Dataset
The dataset includes:
- **67,349 training examples**
- **872 validation examples**
- **1,821 test examples**

### Model Architecture
The project focuses on building the **Encoder** of the Transformer model, which includes:
1. **Attention Mechanism**: Scaled dot product attention functions to support efficient attention calculations.
2. **Multi-Head Attention**: Captures different aspects of the input sequence.
3. **Layer Normalization**: Stabilizes training by normalizing each element in the feature space.
4. **FeedForward Block**: Implements the feedforward operations in the encoder.
5. **Positional Encoding**: Adds positional information to input sequences.
6. **Encoder Transformer**: Combines all components to process the input and predict sentiment.

### Steps Taken
1. The SST-2 dataset was preprocessed into tokenized sequences.
2. The key components of the Transformer encoder were implemented.
3. The model was trained on the SST-2 dataset.
4. The model's performance was evaluated on the test set for sentiment classification.

### Conclusion
This project demonstrates the application of Transformer models to sentiment analysis, showcasing how transformers handle sequential data and their power in NLP tasks.
