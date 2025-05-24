import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)


def compute_metrics(p):

    preds, labels = p

    # Ensure both are NumPy arrays
    if not isinstance(preds, np.ndarray):
        preds = preds.detach().numpy()
    if not isinstance(labels, np.ndarray):
        labels = labels.detach().numpy()

    preds = np.argmax(preds, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


def evaluate_on_subset(dataset, model, tokenizer, label2id, compute_metrics_fn=compute_metrics, n=50):
    y_true = []
    y_pred = []

    subset = dataset.select(range(n))

    for example in subset:
        # Tokenize with padding & truncation
        inputs = tokenizer(
            example["sms_text"],
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            logits = model(**inputs).logits

        y_true.append(example["label"])
        y_pred.append(logits)

    # Convert labels to integers
    y_true = [label2id[label] for label in y_true]

    # Stack predictions and evaluate
    logits_tensor = torch.cat(y_pred, dim=0)
    labels_tensor = torch.tensor(y_true)

    return compute_metrics_fn((logits_tensor, labels_tensor))
