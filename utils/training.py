import torch
from collections import Counter
from transformers import DistilBertForSequenceClassification, TrainerCallback


class WeightedLossBERT(DistilBertForSequenceClassification):
    """
    DistilBERT model for sequence classification with weighted loss support.
    
    Extends the standard DistilBertForSequenceClassification to support
    class-weighted loss computation for handling imbalanced datasets.
    
    Args:
        config: Model configuration object
        class_weights (torch.Tensor, optional): Weights for each class in loss computation
    """
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.class_weights = class_weights

    def compute_loss(self, model_outputs, labels):
        """
        Compute weighted cross-entropy loss.
        
        Args:
            model_outputs: Model output containing logits
            labels (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Computed loss value
        """
        logits = model_outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        return loss_fn(logits, labels)
    

def compute_class_weights(dataset, label2id):
    """
    Compute balanced class weights for handling imbalanced datasets.
    
    Calculates inverse frequency weights to balance the loss contribution
    of each class, giving higher weight to underrepresented classes.
    
    Args:
        dataset: HuggingFace dataset containing 'label' column
        label2id (dict): Mapping from label strings to integer IDs
        
    Returns:
        torch.Tensor: Class weights tensor ordered by label IDs
        
    Example:
        >>> weights = compute_class_weights(train_dataset, {"spam": 0, "ham": 1})
        >>> tensor([1.2000, 0.8000])  # Higher weight for underrepresented class
    """
    # Extract raw labels
    labels = dataset["label"]  # list of strings like "valid" or "not valid"
    
    # Count frequencies
    label_counts = Counter(labels)
    num_samples = len(labels)
    num_classes = len(label_counts)

    # Compute weight for each class
    class_weights = {
        label: num_samples / (num_classes * count)
        for label, count in label_counts.items()
    }

    # Convert to tensor in correct label ID order
    weights = torch.tensor([class_weights[label] for label in sorted(label2id, key=label2id.get)])

    return weights


class PrettyMetricsCallback(TrainerCallback):
    """
    Custom callback for prettier metrics logging during training.
    
    Formats and displays training metrics in a more readable format
    with proper alignment and visual separators.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when the trainer logs metrics.
        
        Args:
            args: Training arguments
            state: Current trainer state
            control: Training control object
            logs (dict, optional): Dictionary of metrics to log
            **kwargs: Additional keyword arguments
        """
        if logs:
            print("\n📊 Metrics Log (Epoch {:.1f})".format(state.epoch or 0))
            print("-" * 40)
            for k, v in logs.items():
                if isinstance(v, float):
                    print(f"{k:<25} : {v:.4f}")
                else:
                    print(f"{k:<25} : {v}")
            print("-" * 40)

