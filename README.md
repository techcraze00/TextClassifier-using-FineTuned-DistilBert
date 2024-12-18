# Active/Passive Sentence Classification using DistilBERT

This project uses **DistilBERT**, a lightweight version of the popular **BERT** model, to classify sentences as either **active** or **passive**. The model is fine-tuned on a small dataset of sentence examples and can predict the grammatical voice of any given sentence.

## Features
- **Text Classification**: Classifies sentences into two categories: Active or Passive.
- **Transformer-based Model**: Utilizes **DistilBERT** for high-performance NLP.
- **Minimal Data**: Fine-tuned with a small dataset, leveraging the power of transfer learning.
- **Fast Inference**: Thanks to the lightweight DistilBERT architecture.

### Google Colab Link: [Active/Passive Sentence Classifier](https://colab.research.google.com/drive/1VczTW143oaEsnYjbIs5iFnl_sEx1rnqT?usp=sharing) 
### Hugging Face: [ActiveVoice_PassiveVoice_Classifier](https://huggingface.co/first-techcraze/ActiveVoice_PassiveVoice_Classifier/tree/main)

---
## Requirements

### Libraries
You can install the required libraries using pip:

```bash
pip install transformers tensorflow datasets numpy pandas matplotlib scikit-learn
```

## Model Fine-Tuning
The model is fine-tuned on a custom dataset with sentences labeled as active or passive. Below are the steps to fine-tune the DistilBERT model:

### 1. Preprocessing the Data
The input sentences are tokenized using the DistilBERT tokenizer, which converts them into a format that can be processed by the model. The tokenizer handles padding and truncation to ensure uniform input length.

```bash
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
inputs = tokenizer(texts, return_tensors="tf", padding=True, truncation=True, max_length=128)
```

### Loading and Pre-training Model
The DistilBERT model is loaded and prepared for fine-tuning. It is instantiated with a classification head for sequence classification.

```bash
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
```

The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function since this is a classification task.

```bash
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

Once the model is fine-tuned and saved, you can use it to classify new sentences as active or passive.

---
## Author
Prayas Jadhav
