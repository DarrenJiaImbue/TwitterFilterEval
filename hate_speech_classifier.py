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

# Optional: Import for HuggingFace classifier
try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    import torch
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. HuggingFace classifier will not work.")


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


class HuggingFaceZeroShotClassifier(BaseClassifier):
    """Zero-shot classifier using HuggingFace transformers"""

    def __init__(self, model_name: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0", batch_size: int = 10, device: int = -1, threshold: float = 0.5):
        """
        Initialize the zero-shot classifier.

        Args:
            model_name: HuggingFace model name
            batch_size: Number of texts to classify at once
            device: Device to run on (-1 for CPU, 0+ for GPU)
            threshold: Confidence threshold for classifying as hate speech (default: 0.5)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for HuggingFaceZeroShotClassifier. Install with: pip install transformers torch")

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.threshold = threshold

        print(f"Loading model {model_name}...")
        self.pipe = pipeline("zero-shot-classification", model=model_name, device=device)
        print("Model loaded successfully!")

        # Use only "hate speech" as the label to get direct confidence score
        self.hypothesis_template = "This text contains {}"
        self.candidate_labels = ["hate speech"]

    async def classify_batch(self, texts: List[str]) -> List[Tuple[int, str]]:
        """Classify a batch of texts using zero-shot classification"""
        # Run the classification (synchronous, but we'll wrap it for async compatibility)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._classify_sync, texts)
        return results

    def _classify_sync(self, texts: List[str]) -> List[Tuple[int, str]]:
        """Synchronous classification method"""
        try:
            # Process all texts in the batch at once
            results = self.pipe(texts, self.candidate_labels, hypothesis_template=self.hypothesis_template, multi_label=False)

            classifications = []
            for result in results:
                # Get confidence score for "hate speech"
                confidence = result['scores'][0]

                # Apply threshold to determine label
                label = 1 if confidence >= self.threshold else 0

                # Store just the confidence value as reasoning for easier extraction
                reasoning = str(confidence)

                classifications.append((label, reasoning))

            return classifications

        except Exception as e:
            print(f"Error classifying batch: {e}")
            # Fallback to one-at-a-time if batched processing fails
            print("Falling back to individual classification...")
            classifications = []
            for text in texts:
                try:
                    result = self.pipe(text, self.candidate_labels, hypothesis_template=self.hypothesis_template, multi_label=False)
                    confidence = result['scores'][0]
                    label = 1 if confidence >= self.threshold else 0
                    reasoning = str(confidence)
                    classifications.append((label, reasoning))
                except Exception as e2:
                    print(f"Error classifying individual text: {e2}")
                    classifications.append((0, f"Error during classification: {str(e2)}"))

            return classifications

    def get_name(self) -> str:
        # Extract a shorter model name
        model_short = self.model_name.split('/')[-1].replace('-', '_')
        return f"hf_{model_short}_t{self.threshold}_batch{self.batch_size}"

    def calculate_optimal_threshold(self, results: List[Dict]):
        """Calculate and display the optimal threshold for classification"""
        print(f"\n{'='*60}")
        print("OPTIMAL THRESHOLD ANALYSIS")
        print(f"{'='*60}")

        # Test thresholds from 0.0 to 1.0
        thresholds = [i / 100.0 for i in range(0, 101)]
        best_threshold = 0.5
        best_accuracy = 0.0
        best_metrics = {}

        threshold_results = []

        for threshold in thresholds:
            # Recalculate predictions with this threshold
            tp = fp = fn = tn = 0

            for r in results:
                # Extract confidence from reasoning string (just the confidence value)
                try:
                    conf = float(r.get('new_reasoning', '0'))
                except ValueError:
                    continue

                pred = 1 if conf >= threshold else 0
                truth = r['ground_truth']

                if pred == 1 and truth == 1:
                    tp += 1
                elif pred == 1 and truth == 0:
                    fp += 1
                elif pred == 0 and truth == 1:
                    fn += 1
                else:
                    tn += 1

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            threshold_results.append({
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn
                }

        # Print optimal threshold
        print(f"\nOptimal Threshold (by Accuracy): {best_threshold:.4f}")
        print(f"  Accuracy:  {best_accuracy:.4f}")
        print(f"  F1 Score:  {best_metrics['f1']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall:    {best_metrics['recall']:.4f}")
        print(f"\n  Confusion Matrix at Optimal Threshold:")
        print(f"    TP: {best_metrics['tp']}  FP: {best_metrics['fp']}")
        print(f"    FN: {best_metrics['fn']}  TN: {best_metrics['tn']}")

        # Show current threshold metrics
        current_metrics = next((t for t in threshold_results if abs(t['threshold'] - self.threshold) < 0.001), None)
        if current_metrics:
            print(f"\nCurrent Threshold: {self.threshold:.4f}")
            print(f"  F1 Score:  {current_metrics['f1']:.4f}")
            print(f"  Accuracy:  {current_metrics['accuracy']:.4f}")
            print(f"  Precision: {current_metrics['precision']:.4f}")
            print(f"  Recall:    {current_metrics['recall']:.4f}")

            improvement = best_accuracy - current_metrics['accuracy']
            if improvement > 0.001:
                print(f"\n  💡 Accuracy could improve by {improvement:.4f} ({improvement*100:.2f}%) with optimal threshold")
            else:
                print(f"\n  ✅ Current threshold is already optimal!")

        print(f"{'='*60}")


class EmbeddingSimilarityClassifier(BaseClassifier):
    """Embedding-based classifier using cosine similarity with hate speech examples.

    Compatible with both Python (transformers) and JavaScript (Transformers.js).
    Recommended model: Xenova/all-MiniLM-L6-v2
    """

    # ~50 hate speech examples for similarity comparison
    HATE_SPEECH_EXAMPLES = [
        "I hate all people from that country",
        "They should all be killed",
        "You're such a disgusting person",
        "Go back to where you came from",
        "You people are ruining everything",
        "I wish they would all die",
        "Your kind doesn't belong here",
        "You're nothing but trash",
        "They're all criminals and thugs",
        "I hope you get what you deserve",
        "You're subhuman filth",
        "Your race is inferior",
        "Women are stupid and worthless",
        "Men are all pigs and should suffer",
        "You deserve to be hurt",
        "I hope someone beats you up",
        "You're a waste of oxygen",
        "The world would be better without your kind",
        "You people are all the same - worthless",
        "I hate everyone like you",
        "Your religion is evil and you're all terrorists",
        "Gay people are disgusting and wrong",
        "You should be ashamed of who you are",
        "People like you make me sick",
        "I hope terrible things happen to you",
        "You don't deserve to live",
        "Your community is full of degenerates",
        "I wish violence upon all of you",
        "You're all parasites on society",
        "Get out of my country, you don't belong",
        "Your entire group is worthless trash",
        "I hope you all suffer",
        "You people are a disease",
        "The only good one is a dead one",
        "You deserve all the hate you get",
        "I'd be happy if you all disappeared",
        "Your kind ruins everything you touch",
        "You're animals, not humans",
        "I hate your entire race/religion/group",
        "You should all be removed from society",
        "Nobody wants you here",
        "You're all ugly and disgusting",
        "I hope you experience pain and suffering",
        "Your people are all liars and thieves",
        "The world doesn't need people like you",
        "You're all stupid and inferior",
        "I wish harm upon your entire community",
        "You're a plague on humanity",
        "Go die in a hole somewhere",
        "You're all worthless scum"
    ]

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 10, device: int = -1, threshold: float = 0.5):
        """
        Initialize the embedding similarity classifier.

        Args:
            model_name: HuggingFace model name (use models compatible with Transformers.js)
            batch_size: Number of texts to classify at once
            device: Device to run on (-1 for CPU, 0+ for GPU)
            threshold: Similarity threshold for classifying as hate speech (default: 0.5)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")

        self.model_name = model_name
        self.batch_size = batch_size
        self.device_id = device
        self.threshold = threshold

        # Set device
        if device == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

        print(f"Loading embedding model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

        # Pre-compute embeddings for hate speech examples
        print("Computing embeddings for hate speech examples...")
        self.hate_embeddings = self._compute_embeddings_sync(self.HATE_SPEECH_EXAMPLES)
        print(f"Precomputed {len(self.hate_embeddings)} hate speech example embeddings")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling to get sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _compute_embeddings_sync(self, texts: List[str]) -> torch.Tensor:
        """Compute embeddings for a list of texts (synchronous)"""
        with torch.no_grad():
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
            model_output = self.model(**encoded_input)
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def _compute_max_similarity(self, text_embedding: torch.Tensor) -> float:
        """Compute maximum cosine similarity between text and hate speech examples"""
        # Compute cosine similarities with all hate speech examples
        similarities = torch.mm(text_embedding, self.hate_embeddings.t())
        # Get maximum similarity
        max_similarity = torch.max(similarities).item()
        return max_similarity

    async def classify_batch(self, texts: List[str]) -> List[Tuple[int, str]]:
        """Classify a batch of texts using embedding similarity"""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._classify_sync, texts)
        return results

    def _classify_sync(self, texts: List[str]) -> List[Tuple[int, str]]:
        """Synchronous classification method"""
        try:
            # Compute embeddings for input texts
            text_embeddings = self._compute_embeddings_sync(texts)

            classifications = []
            for i, text_embedding in enumerate(text_embeddings):
                # Compute maximum similarity with hate speech examples
                max_similarity = self._compute_max_similarity(text_embedding.unsqueeze(0))

                # Apply threshold
                label = 1 if max_similarity >= self.threshold else 0

                # Store similarity score as reasoning
                reasoning = f"{max_similarity:.4f}"

                classifications.append((label, reasoning))

            return classifications

        except Exception as e:
            print(f"Error classifying batch: {e}")
            # Return default classifications on error
            return [(0, f"Error: {str(e)}") for _ in texts]

    def get_name(self) -> str:
        # Extract a shorter model name
        model_short = self.model_name.split('/')[-1].replace('-', '_')
        return f"embedding_{model_short}_t{self.threshold}_batch{self.batch_size}"

    def calculate_optimal_threshold(self, results: List[Dict]):
        """Calculate and display the optimal threshold for classification"""
        print(f"\n{'='*60}")
        print("OPTIMAL THRESHOLD ANALYSIS")
        print(f"{'='*60}")

        # Test thresholds from 0.0 to 1.0
        thresholds = [i / 100.0 for i in range(0, 101)]
        best_threshold = 0.5
        best_accuracy = 0.0
        best_metrics = {}

        threshold_results = []

        for threshold in thresholds:
            # Recalculate predictions with this threshold
            tp = fp = fn = tn = 0

            for r in results:
                # Extract similarity from reasoning string
                try:
                    similarity = float(r.get('new_reasoning', '0'))
                except ValueError:
                    continue

                pred = 1 if similarity >= threshold else 0
                truth = r['ground_truth']

                if pred == 1 and truth == 1:
                    tp += 1
                elif pred == 1 and truth == 0:
                    fp += 1
                elif pred == 0 and truth == 1:
                    fn += 1
                else:
                    tn += 1

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            threshold_results.append({
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn
                }

        # Print optimal threshold
        print(f"\nOptimal Threshold (by Accuracy): {best_threshold:.4f}")
        print(f"  Accuracy:  {best_accuracy:.4f}")
        print(f"  F1 Score:  {best_metrics['f1']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall:    {best_metrics['recall']:.4f}")
        print(f"\n  Confusion Matrix at Optimal Threshold:")
        print(f"    TP: {best_metrics['tp']}  FP: {best_metrics['fp']}")
        print(f"    FN: {best_metrics['fn']}  TN: {best_metrics['tn']}")

        # Show current threshold metrics
        current_metrics = next((t for t in threshold_results if abs(t['threshold'] - self.threshold) < 0.001), None)
        if current_metrics:
            print(f"\nCurrent Threshold: {self.threshold:.4f}")
            print(f"  F1 Score:  {current_metrics['f1']:.4f}")
            print(f"  Accuracy:  {current_metrics['accuracy']:.4f}")
            print(f"  Precision: {current_metrics['precision']:.4f}")
            print(f"  Recall:    {current_metrics['recall']:.4f}")

            improvement = best_accuracy - current_metrics['accuracy']
            if improvement > 0.001:
                print(f"\n  💡 Accuracy could improve by {improvement:.4f} ({improvement*100:.2f}%) with optimal threshold")
            else:
                print(f"\n  ✅ Current threshold is already optimal!")

        print(f"{'='*60}")


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
        random.seed(40)  # For reproducibility

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

        # Calculate optimal threshold for zero-shot and embedding classifiers
        if isinstance(self.classifier, (HuggingFaceZeroShotClassifier, EmbeddingSimilarityClassifier)):
            self.classifier.calculate_optimal_threshold(results)

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

    # Configuration
    DATASET_PATH = 'claude_labeled_dataset_20260115_141034.csv'
    SAMPLES_PER_CLASS = 500  # Adjust this to control dataset size
    BATCH_SIZE = 10  # Number of texts to classify in one API call
    MAX_CONCURRENT = 40  # Maximum number of concurrent API requests

    # Load and balance dataset
    dataset = DatasetLoader.load_and_balance(DATASET_PATH, samples_per_class=SAMPLES_PER_CLASS)

    # ========== CHOOSE YOUR CLASSIFIER ==========
    # Option 1: Claude Classifier (requires ANTHROPIC_API_KEY)
    # if not api_key:
    #     print("Warning: ANTHROPIC_API_KEY not found in environment. Skipping Claude classifier.")
    #     classifier = None
    # else:
    #     classifier = ClaudeClassifier(
    #         api_key=api_key,
    #         model="claude-haiku-4-5-20251001",
    #         batch_size=BATCH_SIZE,
    #         max_concurrent=MAX_CONCURRENT
    #     )

    # Option 2: HuggingFace Zero-Shot Classifier (requires transformers and torch)
    # Uncomment the following lines to use the HuggingFace classifier instead:
    # classifier = HuggingFaceZeroShotClassifier(
    #     model_name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    #     batch_size=BATCH_SIZE,
    #     device=-1,  # Use -1 for CPU, 0 for GPU
    #     threshold=0.02  # Confidence threshold (will calculate optimal threshold after evaluation)
    # )

    # Option 3: Embedding Similarity Classifier (requires transformers and torch)
    # Uses cosine similarity with hate speech examples. Compatible with Transformers.js for Chrome extensions!
    # For first-pass filtering: use LOW threshold to catch more hate speech (high recall)
    # False positives are OK since LLM will do second-pass filtering
    classifier = EmbeddingSimilarityClassifier(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Also available as Xenova/all-MiniLM-L6-v2 in Transformers.js
        batch_size=BATCH_SIZE,
        device=-1,  # Use -1 for CPU, 0 for GPU
        threshold=0.3
    )

    if classifier is None:
        raise ValueError("No classifier configured. Please set up API keys or uncomment a classifier option.")

    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results_{classifier.get_name()}_{timestamp}.csv"

    # Run evaluation
    harness = EvaluationHarness(classifier)
    await harness.evaluate(dataset, output_path)


if __name__ == "__main__":
    asyncio.run(main())
