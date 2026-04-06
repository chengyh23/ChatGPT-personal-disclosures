"""
Extract model information for specific conversation hashes from task_annotations.csv
Much faster than loading entire WildChat dataset

Uses raw model names - no grouping/classification
"""

import pandas as pd
from datasets import load_dataset
import time

# Load task annotations to get unique conversation hashes
print("Reading conversation hashes from task_annotations.csv...")
tasks_df = pd.read_csv('task_annotations.csv')
unique_hashes = set(tasks_df['conversation_hash'].unique())
print(f"Found {len(unique_hashes)} unique conversation hashes")

# Load WildChat and filter to only those hashes
print("\nLoading WildChat dataset and filtering for matching hashes...")
start = time.time()

dataset = load_dataset("allenai/WildChat-1M", split="train")
print(f"Total conversations in dataset: {len(dataset)}")

# Filter efficiently using map
def filter_by_hash(example):
    return example['conversation_hash'] in unique_hashes

filtered_dataset = dataset.filter(filter_by_hash)
print(f"Filtered to {len(filtered_dataset)} matching conversations")

# Extract just conversation_hash and model (raw, no classification)
# Keep only first occurrence if there are duplicates
model_mapping = []
seen_hashes = set()
for example in filtered_dataset:
    hash_val = example['conversation_hash']
    if hash_val not in seen_hashes:
        model_mapping.append({
            'conversation_hash': hash_val,
            'model': example['model']
        })
        seen_hashes.add(hash_val)

elapsed = time.time() - start
print(f"Extraction completed in {elapsed:.1f} seconds")

# Save to CSV
mapping_df = pd.DataFrame(model_mapping)
mapping_df.to_csv('conversation_hash_to_model.csv', index=False)

# Check for duplicates
if len(mapping_df) != len(mapping_df['conversation_hash'].unique()):
    print("\n⚠️  Warning: Duplicate conversation_hash values found")
    print(f"Total rows: {len(mapping_df)}")
    print(f"Unique hashes: {len(mapping_df['conversation_hash'].unique())}")
else:
    print(f"\n✓ All {len(mapping_df)} conversation hashes are unique")
print(f"\nSaved model mapping to conversation_hash_to_model.csv")

# Show summary
print("\nModel distribution in your conversations:")
print(mapping_df['model'].value_counts())

# Verify coverage
tasks_with_models = tasks_df['conversation_hash'].isin(mapping_df['conversation_hash']).sum()
print(f"\nCoverage: {tasks_with_models}/{len(tasks_df)} tasks have model info")
