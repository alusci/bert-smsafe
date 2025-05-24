# bert-smsafe

This repository contains a BERT-based NLP pipeline for detecting spam messages in SMS communications, particularly focusing on OTP and authentication flows. Built for applications in fraud prevention, message filtering, and telecom infrastructure.

## 🤗 Pre-trained Model

The trained model is available on Hugging Face Hub: [alusci/distilbert-smsafe](https://huggingface.co/alusci/distilbert-smsafe)

## 📋 Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- PEFT (Parameter Efficient Fine-tuning)
- scikit-learn
- numpy

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/alusci/bert-smsafe.git
cd bert-smsafe
```

2. **Create a conda environment:**
```bash
conda create -n smsafe-env python=3.11
conda activate smsafe-env
```

3. **Install dependencies:**
```bash
conda install pytorch transformers datasets scikit-learn numpy -c pytorch -c huggingface -c conda-forge
pip install peft
```

## 🏃‍♂️ Quick Start

### Using the Pre-trained Model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the model and tokenizer
model_name = "alusci/distilbert-smsafe"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Classify an SMS message
text = "Your OTP is 123456. Use it to complete your login."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)

# Get prediction
prediction = outputs.logits.argmax(-1).item()
label = "valid" if prediction == 1 else "not valid"
print(f"Message: {text}")
print(f"Classification: {label}")
```

### Training Your Own Model

1. **Prepare your dataset** in the format expected by the training script (with columns: `sms_text`, `label`, etc.)

2. **Activate the conda environment:**
```bash
conda activate smsafe-env
```

3. **Run the training script:**
```bash
python train_model.py
```

The training script will:
- Load the dataset from Hugging Face Hub
- Fine-tune DistilBERT with LoRA (Low-Rank Adaptation)
- Use weighted loss to handle class imbalance
- Save the trained model and push to Hugging Face Hub

### Configuration

You can modify the training parameters in `train_model.py`:

```python
# Model hyperparameters
LR = 1e-5           # Learning rate
BATCH_SIZE = 50     # Batch size
NUM_EPOCHS = 5      # Number of training epochs

# Dataset and model
DATASET_NAME = "alusci/sms-otp-spam-dataset"
MODEL_NAME = "distilbert-base-uncased"
```

## 📊 Model Performance

The model is trained to classify SMS messages as:
- **valid**: Legitimate OTP/authentication messages
- **not valid**: Spam or fraudulent messages

Performance metrics are automatically computed and displayed during training.

## 🔧 Project Structure

```
bert-smsafe/
├── train_model.py          # Main training script
├── utils/
│   ├── training.py         # Custom model classes and training utilities
│   └── evaluation.py       # Evaluation metrics and functions
├── README.md
└── requirements.txt
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.
