import numpy as np
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoConfig
)
from peft import LoraConfig, get_peft_model
from utils.training import WeightedLossBERT, compute_class_weights, PrettyMetricsCallback
from utils.evaluation import compute_metrics, evaluate_on_subset


# Encourage deterministic behavior
random.seed(1)
np.random.seed(2)
torch.manual_seed(3)

# Define parameters
DATASET_NAME = "alusci/sms-otp-spam-dataset"
MODEL_NAME = "distilbert-base-uncased"

# Define model hyperparameters
LR = 1e-5
BATCH_SIZE = 50
NUM_EPOCHS = 5

# Define classes map
id2label = {0: "not valid", 1: "valid"}
label2id = {"not valid": 0, "valid": 1}


def tokenize_fn(example):

    text = example["sms_text"]
    tokenizer.truncation_side = "left"
    
    tokenized = tokenizer(
        text,
        truncation=True,
        return_tensors="np",
        max_length=1024
    )

    tokenized["labels"] = [label2id[label] for label in example["label"]]

    return tokenized


if __name__ == "__main__":
    
    # Load dataset

    dataset = load_dataset(DATASET_NAME, download_mode="force_redownload")
    dataset = dataset["train"].train_test_split(test_size=0.2, seed=0)

    print(dataset["train"])
    print(dataset["test"])
    
    print(dataset["train"][0])
    print(dataset["test"][0])

    # Load model
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    model = WeightedLossBERT.from_pretrained(
        MODEL_NAME,
        config=config,
        class_weights=compute_class_weights(dataset["train"], label2id)
    )

    print(model)

    # Load tokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    print(tokenizer.tokenize(dataset["train"][0]["sms_text"]))

    # Apply tokenizer

    tokenized_dataset = dataset.map(
        tokenize_fn, 
        batched=True,
        remove_columns=["phone_id", "timestamp", "status", "label"]
    )

    print(tokenized_dataset["train"][0])
    print(tokenized_dataset["test"][0])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("Untrained model predictions:")
    print("----------------------------")

    metrics = evaluate_on_subset(
        dataset["test"],
        model,
        tokenizer,
        label2id
    )
    print(metrics)
    print()

    # Define LoRa model
    peft_config = LoraConfig(
        task_type="SEQ_CLS",  
        r=8,  
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
    )

    lora_model = get_peft_model(model, peft_config)
    lora_model.print_trainable_parameters()
    print(lora_model)

    # Define trainer
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[PrettyMetricsCallback()]
    )

    # Train model
    trainer.train()

    # Get fine-tuned model
    lora_model.eval()
    tuned_model = lora_model.merge_and_unload()

    # Push to Hugging Face
    tuned_model.push_to_hub("alusci/distilbert-smsafe")
    tokenizer.push_to_hub("alusci/distilbert-smsafe")





    

    

        

     






