# Chunks Analysis - OCR Error Detection

This folder contains a set of Jupyter notebooks implementing and analyzing OCR error detection using chunking-based approach. The chunking approach divides text into overlapping segments and calculates perplexity scores using LLMs.

## Notebooks

### 1. `run_analysis.ipynb` 

Core notebook for processing input XML files and generating error score predictions. Results can be saved as JSON or CSV files with folowing format:

**Output Format**:
```
word, line_id, paragraph_id, word_position, perplexity, is_error, page_id
```


### 2. `run_predictions.ipynb` 

Evaluates chunking-based predictions against ground truth annotations. Notebook loads chunking analysis results from CSV files, uses HybridOverlayGtScorer for reference annotations and generates overlay visualizations showing detected errors. 


### 3. `threshold_detection.ipynb` 

This notebook determines optimal perplexity thresholds for error detection using IQR-based outlier detection.

### 4. `visualisation_error_distribution.ipynb` 

This notebook visualizes the distribution of word-level perplexity, highlighting contiguous regions with high error rates. It also shows how the error threshold is positioned relative to the overall perplexity distribution, providing insights into how errors are identified and evaluated.

