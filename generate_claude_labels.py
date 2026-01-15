"""
Generate Claude-based Labels for Hate Speech Dataset

This script uses Claude to generate new labels for the entire hate speech dataset.
The generated labels can be used as ground truth for future evaluations.
Uses the same prompts and settings as the evaluation harness.
"""

import os
import csv
import asyncio
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm

# Import from the existing classifier
from hate_speech_classifier import ClaudeClassifier


def download_dataset():
    """Download the dataset using kagglehub if not present"""
    dataset_path = 'HateSpeechDataset.csv'

    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path

    print("Dataset not found. Downloading from Kaggle...")
    try:
        import kagglehub

        # Download latest version
        path = kagglehub.dataset_download("waalbannyantudre/hate-speech-detection-curated-dataset")
        print("Path to dataset files:", path)

        # Find the CSV file in the downloaded path
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    source_file = os.path.join(root, file)
                    # Copy to current directory
                    import shutil
                    shutil.copy(source_file, dataset_path)
                    print(f"Copied dataset to {dataset_path}")
                    return dataset_path

        raise FileNotFoundError("Could not find CSV file in downloaded dataset")

    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")


def load_entire_dataset(csv_path: str) -> List[Dict]:
    """Load the entire dataset"""
    print(f"Loading entire dataset from {csv_path}...")

    data = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip duplicate header rows
            if row['Label'] == 'Label':
                continue

            content = row['Content']
            original_label = int(row['Label'])

            data.append({
                'content': content,
                'original_label': original_label
            })

    print(f"Loaded {len(data)} samples from dataset")
    return data


def load_balanced_dataset(csv_path: str) -> List[Dict]:
    """Load a balanced dataset with equal numbers of label 0 and label 1"""
    print(f"Loading balanced dataset from {csv_path}...")

    label_0_samples = []
    label_1_samples = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip duplicate header rows
            if row['Label'] == 'Label':
                continue

            content = row['Content']
            original_label = int(row['Label'])

            if original_label == 0:
                label_0_samples.append({
                    'content': content,
                    'original_label': original_label
                })
            elif original_label == 1:
                label_1_samples.append({
                    'content': content,
                    'original_label': original_label
                })

    # Balance to the minimum class size, capped at 10,000
    min_size = min(len(label_0_samples), len(label_1_samples), 10000)

    print(f"Original distribution - Label 0: {len(label_0_samples)}, Label 1: {len(label_1_samples)}")
    print(f"Using {min_size} samples from each class for balanced dataset (capped at 10,000)")

    # Take equal numbers from each class
    balanced_data = label_0_samples[:min_size] + label_1_samples[:min_size]

    # Shuffle to mix the labels
    import random
    random.shuffle(balanced_data)

    print(f"Loaded {len(balanced_data)} balanced samples ({min_size} of each class)")
    return balanced_data


async def generate_labels(dataset: List[Dict], classifier: ClaudeClassifier, output_path: str):
    """Generate labels for the entire dataset"""
    print(f"\nGenerating labels using {classifier.model}...")

    results = []

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(classifier.max_concurrent)

    # Prepare all batches
    batches = []
    for i in range(0, len(dataset), classifier.batch_size):
        batch = dataset[i:i+classifier.batch_size]
        batches.append(batch)

    # Process batches concurrently with progress bar
    async def process_batch(batch):
        async with semaphore:
            texts = [item['content'] for item in batch]
            original_labels = [item['original_label'] for item in batch]

            predictions = await classifier.classify_batch(texts)

            return texts, predictions, original_labels

    # Process all batches concurrently with progress tracking
    tasks = [process_batch(batch) for batch in batches]

    for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Generating labels"):
        texts, predictions, original_labels = await coro

        # Record results
        for text, (claude_label, reasoning), original_label in zip(texts, predictions, original_labels):
            results.append({
                'Content': text,
                'Label': claude_label,
                'Reasoning': reasoning,
                'Original_Label': original_label
            })

    # Save results
    save_results(results, output_path, classifier.model)

    print(f"\n{'='*60}")
    print(f"Label Generation Complete")
    print(f"{'='*60}")
    print(f"Total samples processed: {len(results)}")
    print(f"Labels saved to: {output_path}")

    # Print comparison statistics
    matches = sum(1 for r in results if r['Label'] == r['Original_Label'])
    hate_count = sum(1 for r in results if r['Label'] == 1)
    non_hate_count = sum(1 for r in results if r['Label'] == 0)

    print(f"\nLabel Statistics:")
    print(f"  Hate Speech (Label=1): {hate_count}")
    print(f"  Non-Hate Speech (Label=0): {non_hate_count}")
    print(f"  Agreement with original labels: {matches}/{len(results)} ({matches/len(results)*100:.2f}%)")
    print(f"{'='*60}")


def save_results(results: List[Dict], output_path: str, model: str):
    """Save generated labels to CSV"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Write metadata header
        f.write(f"# Generated by Claude Model: {model}\n")
        f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"# Total Samples: {len(results)}\n")
        f.write("#\n")
        f.write("# Label Meanings:\n")
        f.write("#   1 = Hate Speech (hide)\n")
        f.write("#   0 = Non-Hate Speech (show)\n")
        f.write("#\n")

        # Write results
        writer = csv.DictWriter(f, fieldnames=['Content', 'Label', 'Reasoning', 'Original_Label'])
        writer.writeheader()
        writer.writerows(results)


async def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment. Please set it in .env file")

    # Download dataset if needed
    dataset_path = download_dataset()

    # Configuration - same as evaluation harness
    BATCH_SIZE = 20
    MAX_CONCURRENT = 40
    MODEL = "claude-sonnet-4-5-20250929"

    # Load balanced dataset (equal numbers of label 0 and 1)
    dataset = load_balanced_dataset(dataset_path)

    # Create classifier using the same class from hate_speech_classifier.py
    classifier = ClaudeClassifier(
        api_key=api_key,
        model=MODEL,
        batch_size=BATCH_SIZE,
        max_concurrent=MAX_CONCURRENT
    )

    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"claude_labeled_dataset_{timestamp}.csv"

    # Generate labels
    await generate_labels(dataset, classifier, output_path)


if __name__ == "__main__":
    asyncio.run(main())
