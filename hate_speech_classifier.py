"""
Hate Speech Classification Testing Harness

This script evaluates different classification methods for hate speech detection.
Currently supports Claude LLM with batching. Extensible for embedding-based methods.
"""

import os
import csv
import json
import time
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from datetime import datetime
import anthropic
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm


class BaseClassifier(ABC):
    """Base class for all classifiers"""

    @abstractmethod
    async def classify_batch(self, texts: List[str]) -> List[Tuple[int, str]]:
        """Classify a batch of texts. Returns list of tuples (label, reasoning)"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this classifier"""
        pass


class ClaudeClassifier(BaseClassifier):
    """Claude-based hate speech classifier with batching support"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", batch_size: int = 10, max_concurrent: int = 10):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent

        self.system_prompt = """You are a tweet filter assistant. Your task is to determine if tweets should be hidden based on user preferences.

The user wants you to hide tweets that contain the following annoying content types:
- Hate Speech

For each tweet, provide a brief sentence explaining your reasoning for whether to hide or show the tweet followed by a line including your final decision.

Respond in this exact format for each tweet where reason is your explanation and result is either hide or show:
Tweet 1: <reasoning>reason</reasoning>
<result>hide</result>

Tweet 2: <reasoning>reason</reasoning>
<result>show</result>"""

    async def classify_batch(self, texts: List[str]) -> List[Tuple[int, str]]:
        """Classify a batch of texts using Claude"""
        # Format the batch for Claude
        texts_formatted = "\n\n".join([f"Tweet {i+1}: {text}" for i, text in enumerate(texts)])

        user_message = f"""Evaluate these tweets:

{texts_formatted}

For each tweet, provide your reasoning and decision in the format specified in your instructions."""

        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            # Parse the response
            response_text = message.content[0].text.strip()

            # Extract reasoning and results from XML tags
            import re
            reasoning_matches = re.findall(r'<reasoning>(.*?)</reasoning>', response_text, re.IGNORECASE | re.DOTALL)
            result_matches = re.findall(r'<result>(hide|show)</result>', response_text, re.IGNORECASE)

            if not result_matches:
                raise ValueError(f"Could not find <result> tags in response: {response_text}")

            if not reasoning_matches:
                print(f"Warning: Could not find <reasoning> tags, using empty reasoning")
                reasoning_matches = [""] * len(result_matches)

            # Convert hide/show to 1/0 (hide=1, show=0)
            classifications = [(1 if r.lower() == 'hide' else 0, reasoning.strip())
                             for r, reasoning in zip(result_matches, reasoning_matches)]

            if len(classifications) != len(texts):
                raise ValueError(f"Expected {len(texts)} classifications, got {len(classifications)}")

            return classifications

        except Exception as e:
            print(f"\nError classifying batch: {e}")
            print(f"Falling back to individual classification...")
            # Fallback: classify one at a time
            return await asyncio.gather(*[self._classify_single(text) for text in texts])

    async def _classify_single(self, text: str) -> Tuple[int, str]:
        """Classify a single text (fallback method)"""
        try:
            message = await self.client.messages.create(
                model=self.model,
                max_tokens=200,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": f"Tweet 1: {text}"}
                ]
            )

            response = message.content[0].text.strip()
            # Extract reasoning and result
            import re
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.IGNORECASE | re.DOTALL)
            result_match = re.search(r'<result>(hide|show)</result>', response, re.IGNORECASE)

            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

            if result_match:
                label = 1 if result_match.group(1).lower() == 'hide' else 0
                return (label, reasoning)
            else:
                print(f"Warning: Could not parse response '{response}', defaulting to 0")
                return (0, reasoning)

        except Exception as e:
            print(f"Error in single classification: {e}")
            return (0, "Error during classification")

    def get_name(self) -> str:
        return f"claude_{self.model.split('-')[1]}_batch{self.batch_size}"


class DatasetLoader:
    """Handles loading and balancing the dataset"""

    @staticmethod
    def load_and_balance(csv_path: str, samples_per_class: int = 500) -> List[Dict]:
        """Load dataset and create a balanced subset"""
        print(f"Loading dataset from {csv_path}...")

        hate_speech = []
        non_hate_speech = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            # Skip comment lines starting with #
            lines = [line for line in f if not line.startswith('#')]

            import io
            reader = csv.DictReader(io.StringIO(''.join(lines)))
            for row in reader:
                # Skip duplicate header rows
                if row['Label'] == 'Label':
                    continue

                content = row['Content']
                label = int(row['Label'])
                original_reasoning = row.get('Reasoning', '')  # Get original reasoning if available

                if label == 1:
                    hate_speech.append({'content': content, 'label': label, 'original_reasoning': original_reasoning})
                else:
                    non_hate_speech.append({'content': content, 'label': label, 'original_reasoning': original_reasoning})

        print(f"Found {len(hate_speech)} hate speech samples and {len(non_hate_speech)} non-hate speech samples")

        # Balance the dataset
        import random
        random.seed(42)  # For reproducibility

        hate_speech_sample = random.sample(hate_speech, min(samples_per_class, len(hate_speech)))
        non_hate_speech_sample = random.sample(non_hate_speech, min(samples_per_class, len(non_hate_speech)))

        balanced_data = hate_speech_sample + non_hate_speech_sample
        random.shuffle(balanced_data)

        print(f"Created balanced dataset with {len(balanced_data)} samples ({len(hate_speech_sample)} hate, {len(non_hate_speech_sample)} non-hate)")

        return balanced_data


class EvaluationHarness:
    """Main evaluation harness"""

    def __init__(self, classifier: BaseClassifier):
        self.classifier = classifier

    async def evaluate(self, dataset: List[Dict], output_path: str):
        """Evaluate classifier on dataset and save results"""
        print(f"\nEvaluating {self.classifier.get_name()}...")

        results = []
        batch_times = []
        overall_start_time = time.time()

        # Process in batches
        batch_size = getattr(self.classifier, 'batch_size', 1)
        max_concurrent = getattr(self.classifier, 'max_concurrent', 10)

        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)

        # Prepare all batches
        batches = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            batches.append(batch)

        # Process batches concurrently with progress bar
        async def process_batch(batch):
            async with semaphore:
                texts = [item['content'] for item in batch]
                ground_truth = [item['label'] for item in batch]
                original_reasoning = [item.get('original_reasoning', '') for item in batch]

                batch_start_time = time.time()
                predictions = await self.classifier.classify_batch(texts)
                batch_elapsed_time = time.time() - batch_start_time

                return texts, predictions, ground_truth, original_reasoning, batch_elapsed_time

        # Process all batches concurrently with progress tracking
        tasks = [process_batch(batch) for batch in batches]

        for coro in atqdm.as_completed(tasks, total=len(tasks), desc="Processing batches"):
            texts, predictions, ground_truth, original_reasoning, batch_elapsed_time = await coro
            batch_times.append(batch_elapsed_time)

            # Record results
            for text, (pred, reasoning), truth, orig_reasoning in zip(texts, predictions, ground_truth, original_reasoning):
                results.append({
                    'message': text,
                    'predicted_label': pred,
                    'ground_truth': truth,
                    'original_reasoning': orig_reasoning,
                    'new_reasoning': reasoning,
                    'correct': pred == truth
                })

        overall_elapsed_time = time.time() - overall_start_time
        correct = sum(1 for r in results if r['correct'])
        total = len(results)

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0

        # Calculate precision, recall, F1 for hate speech class (label 1)
        tp = sum(1 for r in results if r['predicted_label'] == 1 and r['ground_truth'] == 1)
        fp = sum(1 for r in results if r['predicted_label'] == 1 and r['ground_truth'] == 0)
        fn = sum(1 for r in results if r['predicted_label'] == 0 and r['ground_truth'] == 1)
        tn = sum(1 for r in results if r['predicted_label'] == 0 and r['ground_truth'] == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate timing statistics
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0

        # Save results
        self._save_results(results, output_path, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'total_time': overall_elapsed_time,
            'avg_batch_time': avg_batch_time,
            'num_batches': len(batch_times)
        })

        # Print summary
        print(f"\n{'='*60}")
        print(f"Results for {self.classifier.get_name()}")
        print(f"{'='*60}")
        print(f"Accuracy:  {accuracy:.4f} ({correct}/{total})")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp}  FP: {fp}")
        print(f"  FN: {fn}  TN: {tn}")
        print(f"\nTiming:")
        print(f"  Total time: {overall_elapsed_time:.2f}s")
        print(f"  Avg time per batch: {avg_batch_time:.2f}s")
        print(f"  Number of batches: {len(batch_times)}")
        print(f"\nResults saved to {output_path}")
        print(f"{'='*60}")

    def _save_results(self, results: List[Dict], output_path: str, metrics: Dict):
        """Save evaluation results to CSV"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            # Write metadata header
            f.write(f"# Classifier: {self.classifier.get_name()}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"# Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"# Precision: {metrics['precision']:.4f}\n")
            f.write(f"# Recall: {metrics['recall']:.4f}\n")
            f.write(f"# F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"# TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}, TN: {metrics['true_negatives']}\n")
            f.write(f"# Total Time: {metrics['total_time']:.2f}s\n")
            f.write(f"# Avg Batch Time: {metrics['avg_batch_time']:.2f}s\n")
            f.write(f"# Number of Batches: {metrics['num_batches']}\n")
            f.write("#\n")

            # Write results
            writer = csv.DictWriter(f, fieldnames=['message', 'predicted_label', 'ground_truth', 'original_reasoning', 'new_reasoning', 'correct'])
            writer.writeheader()
            writer.writerows(results)


async def main():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment. Please set it in .env file")

    # Configuration
    DATASET_PATH = 'claude_labeled_dataset_20260115_141034.csv'
    SAMPLES_PER_CLASS = 500  # Adjust this to control dataset size
    BATCH_SIZE = 10  # Number of texts to classify in one API call
    MAX_CONCURRENT = 40  # Maximum number of concurrent API requests

    # Load and balance dataset
    dataset = DatasetLoader.load_and_balance(DATASET_PATH, samples_per_class=SAMPLES_PER_CLASS)

    # Create classifier
    classifier = ClaudeClassifier(
        api_key=api_key,
        model="claude-haiku-4-5-20251001",
        batch_size=BATCH_SIZE,
        max_concurrent=MAX_CONCURRENT
    )

    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results_{classifier.get_name()}_{timestamp}.csv"

    # Run evaluation
    harness = EvaluationHarness(classifier)
    await harness.evaluate(dataset, output_path)


if __name__ == "__main__":
    asyncio.run(main())
