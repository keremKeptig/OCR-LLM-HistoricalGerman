# Log-Likelihood Analysis 

This notebook implements a method to detect potential OCR (Optical Character Recognition) errors by analyzing the log-likelihoods of words in a given text using a pre-trained Language Model (LLM). 


## Overview
This directory consists of two main components:

Log-Likelihood Analysis Notebook

- Extract text from PageXML OCR output

- Normalize historical characters

- Compute word-level likelihoods with a LLM

- Identify and visualize low-likelihood regions (potential OCR errors)

- Save analysis to JSON, CSV, and HTML

Prediction Evaluation Notebook

- Compare predicted OCR error regions against ground truth

- Use hybrid scoring and overlay methods

- Generate visual overlays and XML outputs


## Dependencies

Install the following Python packages:

```bash
pip install torch transformers pagexml termcolor numpy
```

---

## Configuration

Modify these parameters to adjust thresholds or model behavior:

```python
MODEL_ID = "meta-llama/Llama-3.2-1B"
CRITICAL_LL_THRESHOLD = -10.0
LOW_LL_THRESHOLD = -6.0
VISUALIZATION_THRESHOLD_RED = -5.5
VISUALIZATION_THRESHOLD_YELLOW = -3.5
MODEL_MAX_LENGTH = 1024
```

---

## Example Output

Visual outputs (in HTML or CLI) color-code the words:

* 🟢 Green: High likelihood (expected word)
* 🟡 Yellow: Unusual but acceptable
* 🔴 Red: Potential OCR error
* 🔴 Dark Red: Critically unlikely (likely OCR issue)

---


