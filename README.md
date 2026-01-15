# Hate Speech Classification Evaluation

A testing harness for evaluating different hate speech classification methods on Twitter data.

## Features

- **Balanced Dataset**: Automatically creates a balanced subset from the original dataset
- **Batching Support**: Reduces API calls by processing multiple texts in one request
- **Extensible Architecture**: Easy to add new classification methods (embeddings, other LLMs, etc.)
- **Comprehensive Metrics**: Accuracy, precision, recall, F1 score, and confusion matrix
- **Detailed Results**: CSV output with each message, prediction, and ground truth

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API key:
   - Add your Anthropic API key to `.env` (already created)
   - Get your API key from: https://console.anthropic.com/settings/keys

## Usage

Run the evaluation:
```bash
python hate_speech_classifier.py
```

### Configuration

You can modify these parameters in `hate_speech_classifier.py`:

```python
SAMPLES_PER_CLASS = 500   # Number of samples per class (hate/non-hate)
BATCH_SIZE = 5            # Number of texts to classify per API call
```

## Output

The script generates a timestamped CSV file with results:
- `results_claude_sonnet_batch5_YYYYMMDD_HHMMSS.csv`

The file includes:
- Header with classifier name, timestamp, and metrics
- Each row: message, predicted_label, ground_truth, correct

## Dataset

The original dataset (`HateSpeechDataset.csv`) contains:
- **361,594** non-hate speech samples (label 0)
- **79,305** hate speech samples (label 1)

The harness automatically balances this by sampling equal amounts from each class.

## Architecture

### BaseClassifier
Abstract base class for all classifiers. Implement this to add new methods:

```python
class MyClassifier(BaseClassifier):
    def classify_batch(self, texts: List[str]) -> List[int]:
        # Your classification logic
        pass

    def get_name(self) -> str:
        return "my_classifier"
```

### ClaudeClassifier
Current implementation using Claude Sonnet 4 with batching support.

### Future Extensions

To add embedding-based classification:

```python
class EmbeddingClassifier(BaseClassifier):
    def __init__(self, embedding_model, threshold=0.8):
        self.model = embedding_model
        self.threshold = threshold
        # Load hate speech examples and compute embeddings

    def classify_batch(self, texts: List[str]) -> List[int]:
        # Compute embeddings and cosine similarity
        # Compare to threshold
        pass

    def get_name(self) -> str:
        return "embedding_cosine"
```

Then use it in `main()`:
```python
classifier = EmbeddingClassifier(your_embedding_model)
```

## Metrics

The harness calculates:
- **Accuracy**: Overall correctness
- **Precision**: Of predicted hate speech, how many were actually hate speech
- **Recall**: Of actual hate speech, how many were detected
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: TP, FP, TN, FN

## Example Output

```
==============================================================
Results for claude_sonnet_batch5
==============================================================
Accuracy:  0.9450 (945/1000)
Precision: 0.9320
Recall:    0.9580
F1 Score:  0.9448

Confusion Matrix:
  TP: 479  FP: 35
  FN: 21  TN: 465

Results saved to results_claude_sonnet_batch5_20260115_143022.csv
==============================================================
```