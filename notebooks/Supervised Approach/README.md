# Supervised Approach

This directory contains scripts and notebooks for building a supervised machine learning model to predict OCR error rates at the page level by using word-level error labels.

## Files

- **label_each_word.ipynb** – Jupyter notebook for generating word-level `"correct"`/`"error"` labels by comparing OCR output to ground truth files.
- **extract_features.py** – Python script containing functions to extract handcrafted structural features from page XML files.
- **rf_regressor.ipynb** – Jupyter notebook for training and evaluating a Random Forest regressor to predict page-level OCR error rates.

---

## Workflow Overview

### 1. Generate Word-Level Labels

**Purpose:**  
Start with OCR output and ground truth files. The notebook `label_each_word.ipynb` aligns words from the OCR and ground truth for each page and labels every word in the OCR output as `"correct"` or `"error"`.

**How to use:**

- Specify the directories for:
  - `GT_DIR` (directory with ground truth XML files, *with* marginalia)
  - `INPUT_DIR` (directory with OCR output XML files, *without* marginalia)
- Run all cells in the notebook.
- Output:  
  - A JSONL file at `root/data/word_level_labels.jsonl` containing all word-level labels, required for model training.

---

### 2. Train and Evaluate the Error Rate Predictor

**Purpose:**  
Use the generated word-level labels to train a model that predicts the error rate for each page based on structural and semantic features.

**Steps (in `rf_regressor.ipynb`):**

1. **Data Preparation & Splitting**
   - Load the word-level label data (`word_level_labels.jsonl`).
   - Compute page-level error rates:  
     `error_rate = (number of "error" labels) / (total number of labels for the page)`
   - Split the pages into training (80%) and test (20%) sets.

2. **Feature Extraction**
   - For each page XML file, extract:
     - **Text Embeddings:** Using a sentence transformer model (e.g., MiniLM).
     - **Handcrafted Features:** Using `extract_filtered_features_from_page` from `extract_features.py` (see list below).
   - Concatenate embeddings and handcrafted features for model input.

3. **Model Training**
   - Train a Random Forest regressor to predict the page error rate.
   - Perform grid search for hyperparameter tuning.
   - Save the best model and evaluate on the test set (MSE, RMSE, MAE).

4. **Internal Evaluation & Visualization**
   - Compute and visualize test set error metrics and residuals.

5. **External Evaluation**
   - Load the trained regressor.
   - Apply it to manually annotated pages.
   - Compare predicted error rates with manual annotations using multiple statistics.

6. **Saving Results**
   - Save predicted XML files with error scores to disk.

7. **Error Analysis**
   - Display and inspect the top 20 pages with the largest discrepancies between predicted and manual error rates.

---

### 3. Handcrafted Feature Extraction

The script `extract_features.py` defines the following structural features for each page:

- Number of text regions (paragraphs)
- Number of lines in the page
- Number of words in the page
- Average word length (characters per word)
- Mean number of lines per region
- Variance of lines per region
- Mean number of words per region
- Variance of words per region
- Maximum number of characters per line
- Mean number of characters per line
- Variance of characters per line
- Proportion of words that are numeric
- Proportion of words that are only punctuation (.,;:!?-)
- Maximum word length
- Number of empty regions (regions with no lines)
- Proportion of empty regions
- Number of empty lines (no words or only whitespace)
- Proportion of empty lines

These features are concatenated with a semantic embedding (vector) generated from the entire page’s text which allow the model to learn from both structure and content.

---

## Getting Started

1. **Generate word-level labels**  
   Run `label_each_word.ipynb` to create `word_level_labels.jsonl`.

2. **Extract features and train the model**  
   Run `rf_regressor.ipynb` to extract features, train and evaluate the regressor, and perform error analysis.

3. **(Optional) Use/modify features**  
   Edit `extract_features.py` to add or modify handcrafted feature extraction logic.

---

For more details on model choices, evaluation methods, or directory setup, refer to comments within each notebook/script.
