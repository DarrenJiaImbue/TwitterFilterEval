# Hate Speech Classification Evaluation

A testing harness for evaluating different hate speech classification methods on Twitter data.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API key:
   - Add your Anthropic API key to `.env` (already created)
   - Get your API key from: https://console.anthropic.com/settings/keys

## Usage

Start by running 
```bash
python generate_claude_labels.py
```
This will download the data set and make sonnet generate labels for whether a message is hate speech or not. The original data set classifies all hateful language as hate speech so arguments between users and profanity often get marked as hate speech when I think they shouldn't.
Since we are taking sonnet's interpretation as the ground truth, the accuracy of the other classification methods will be a measure of how similar they are to sonnet.

After uncommenting the classification method you want to run:

```bash
python hate_speech_classifier.py
```
This will generate the results file for viewing.

## Dataset

The original dataset (`HateSpeechDataset.csv`) contains:
- **361,594** non-hate speech samples (label 0)
- **79,305** hate speech samples (label 1)

The harness automatically balances this by sampling equal amounts from each class.

## Takeaways
### Sonnet 4.5
Accuracy: 88.2

Batch Time: 10.45s

Sonnet is the slowest and takes 10.45 to classify 10 tweets.
Something to note about Sonnet 4.5 is that it has 88% accuracy when reclassifying the data points (probably because of temperature) so that is our max accuracy.

### Haiku 4.5
Accuracy: 83.30

Batch Time: 3.83s

Haiku is very fast and and almost just as accurate as Sonnet for this task. If we ant to use a Claude model this is probably it.

### Deberta
Accuracy: 69.20

Batch Time: 5.42s

Open source model run locally on my cpu. Accuracy and speed are worse than haiku but this does solve the privacy and cost issue since people can run it locally. Should work in a chrome extension via the transformers.js library. Alternatively we could spin up an AWS instance with a nice GPU and host it ourselves. The added latency might make this not worth for some users and I guess users would have to trust us not to take their data.

Deberta returns a number in [0,1] to represent the confidence for whether the text is related to a category. The optimal threshold does generalize to the rest of the hate speech dataset but I suspect that a different category/dataset would require a different threshold. This also applies to the next method.
### Embedding Vectors + Cosine similarity

Accuracy: 63.60

Batch Time: 0.07s

I had claude generate 50 sample hate speech strings and found the cosine similarity between each tweet and these 50 strings. If it's over some threshold then it's hate speech.

Blazing fast and can be run in a chrome extension via transformers.js. Kind of inaccurate and also requires finding the right threshold value. I think this would be useful as a first pass filter before passing to some LLM.
The only issue is that theres a lot of overlap in value between hate speech and non hate speech strings. Maybe I can tune the sample strings to make this better? 