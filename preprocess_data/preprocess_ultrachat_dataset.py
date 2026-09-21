import os
import random

from datasets import DatasetDict, load_dataset


SEED = 42
VALIDATION_SIZE = 500

if not os.environ.get("HF_USERNAME"):
    raise RuntimeError("Set HF_USERNAME before preprocessing UltraChat.")
if not os.environ.get("HF_TOKEN"):
    raise RuntimeError("Set HF_TOKEN with write access before pushing the processed dataset.")

print("Loading dataset...")
train_ds = load_dataset("HuggingFaceH4/ultrachat_200k")["train_sft"]
test_ds = load_dataset("HuggingFaceH4/ultrachat_200k")["test_sft"]

# delete messages column
train_ds = train_ds.remove_columns(["messages"])
test_ds = test_ds.remove_columns(["messages"])

# Use a deterministic validation split so every machine gets the same dataset.
print(f"Sampling {VALIDATION_SIZE} examples from train_sft for validation (seed={SEED})...")
val_idx = random.Random(SEED).sample(range(len(train_ds)), VALIDATION_SIZE)
val_idx_set = set(val_idx)
train_idx = [i for i in range(len(train_ds)) if i not in val_idx_set]
val_ds = train_ds.select(val_idx)
train_ds = train_ds.select(train_idx)

print("Creating dataset dict...")
dataset_dict = DatasetDict({
    "train": train_ds,
    "test": test_ds,
    "val": val_ds
})
print(dataset_dict)

print("Pushing to hub...")
dataset_dict.push_to_hub(f"{os.environ['HF_USERNAME']}/UltraChat-200k", token=os.environ["HF_TOKEN"])
print("UltraChat preprocessing and upload completed.")

