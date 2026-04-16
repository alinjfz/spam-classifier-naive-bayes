# Naive Bayes Spam Classifier from Scratch

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![NLTK](https://img.shields.io/badge/NLTK-NLP-green) ![License](https://img.shields.io/badge/License-MIT-yellow)

A complete Naive Bayes text classifier built from first principles (no sklearn, no shortcuts). Implements the full probabilistic pipeline for binary spam/not-spam classification, including the zero-probability problem and Laplace smoothing.

## What This Project Covers

| Concept | Implementation |
|---|---|
| Text tokenization | NLTK `word_tokenize` |
| Bag-of-words representation | NLTK `FreqDist` |
| Prior probabilities | P(spam), P(not_spam) from corpus |
| Conditional probabilities | P(token \| class) |
| Naive Bayes prediction | Multiplicative probability scoring |
| Zero-probability problem | Demonstrated with unseen token |
| Laplace smoothing | Add-one correction |

## Results

The sentence `"blockchain proposal"` demonstrates the zero-probability problem and its fix:

| | P(not_spam) | P(spam) | Prediction |
|---|---|---|---|
| Without smoothing | 0.0 | 0.0 | Fails (tie at zero) |
| With Laplace smoothing | 0.000230 | 0.000344 | **Spam** ✓ |

The result is correct: `"blockchain"` is a strong spam signal while `"proposal"` appears in legitimate email — smoothing lets the model weigh both tokens rather than zeroing out entirely.

## Setup

```bash
pip install nltk
python -c "import nltk; nltk.download('punkt_tab')"
```

Then open `spam_classifier.ipynb` in Jupyter.

## Key Concepts

**Why "Naive"?** The model assumes all tokens are conditionally independent given the class. In practice this is rarely true, but it makes computation efficient and often performs well on text data.

**The Zero Problem:** If a token appears in spam but never in not-spam training documents, P(token | not_spam) = 0. Multiplied across all tokens, the entire not-spam score becomes zero, even if other tokens strongly suggest legitimate email.

**Laplace Fix:** Add 1 to every count and add vocabulary size V to the denominator. No probability is ever exactly zero:

$$P(f|c) = \frac{\text{freq}(f,c) + 1}{\text{freq}(c) + V}$$

## Project Structure

```
spam-classifier-naive-bayes/
├── spam_classifier.ipynb   # Full implementation with commentary
├── README.md
└── LICENSE
```

## License

MIT
