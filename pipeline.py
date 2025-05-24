from transformers import pipeline

# Load the model and tokenizer from the Hugging Face Hub
classifier = pipeline(
    "text-classification",
    model="alusci/distilbert-smsafe",
    tokenizer="alusci/distilbert-smsafe",
    return_all_scores=False  # Set to True if you want logits for all classes
)

# Sample SMS texts
samples = [
    "Your verification code is 903462.",
    "Claim your free iPhone now by visiting this link!",
    "Enter 569824 to continue logging in.",
    "URGENT: Your account is suspended. Call 123456 immediately."
]

# Run predictions
predictions = classifier(samples)

# Print results
for text, pred in zip(samples, predictions):
    print(f"\n📨 SMS: {text}\n🧠 Prediction: {pred['label']} (Score: {pred['score']:.4f})")

