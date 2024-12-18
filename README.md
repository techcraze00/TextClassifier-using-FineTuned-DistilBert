# Active/Passive Sentence Classification using DistilBERT

This project uses **DistilBERT**, a lightweight version of the popular **BERT** model, to classify sentences as either **active** or **passive**. The model is fine-tuned on a small dataset of sentence examples and can predict the grammatical voice of any given sentence.

## Features
- **Text Classification**: Classifies sentences into two categories: Active or Passive.
- **Transformer-based Model**: Utilizes **DistilBERT** for high-performance NLP.
- **Minimal Data**: Fine-tuned with a small dataset, leveraging the power of transfer learning.
- **Fast Inference**: Thanks to the lightweight DistilBERT architecture.

## Requirements

### Libraries
You can install the required libraries using pip:

```bash
pip install transformers tensorflow datasets numpy pandas matplotlib scikit-learn
