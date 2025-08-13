

# OCR Web App

This folder contains Flask-based web interface for evaluating OCR quality using multiple analysis methods.
It processes scanned document pages, detects potential OCR errors, and visualizes them in a browser.

----

## 📂 Project Structure

```
project/
│
├── app.py                        # Main Flask app
├── supervised_runner.py          # Supervised model pipeline logic
├── helpers.py                    # OCR utilities 
├── data_set_visualization/       # Blueprint for dataset visualization
├── static/                       # Frontend JS, CSS, icons
├── templates/                    # HTML templates
└── ocr_project_data/             # Data folder (images, XMLs, outputs)
```

---

## Requirements

Install the dependencies:

```bash
pip install -r requirements.txt
```

```txt
Flask>=3.0.0
Flask-CORS>=4.0.0
Pillow>=10.0.0
Werkzeug>=3.0.0
pandas>=1.0
numpy>=1.18
matplotlib
scipy
tabulate
```

---

## Input Data

Data folders are configured in `app.py` via `app.config`.

### **For Supervised Method**

| Config Key                | Path                                                | Description                          |
| ------------------------- | --------------------------------------------------- | ------------------------------------ |
| `PIPELINE_GT_DIR`         | `ocr_project_data/d2_0001-0100_with_marginalia`     | Ground truth PAGE-XMLs               |
| `PIPELINE_INPUT_DIR`      | `ocr_project_data/d2_0001-0100_without_marginalia`  | OCR output PAGE-XMLs                 |
| `PIPELINE_LABELS_JSONL`   | `ocr_project_data/pipeline/word_level_labels.jsonl` | Auto-generated labels                |
| `PIPELINE_PAGE_RATES_CSV` | `ocr_project_data/pipeline/page_error_rates.csv`    | Auto-generated page error rates      |
| `PIPELINE_MODEL_PATH`     | `ocr_project_data/pipeline/rf_regressor.joblib`     | Trained supervised model             |
| `PIPELINE_DATASET_PATH`   | `data/`                                             | Dataset path for pipeline processing |
| `SUPERVISED_FOLDER`       | `ocr_project_data/supervised_results`               | Predicted XML outputs                |

---

### **For Unsupervised Methods**

| Method                | Folder                                 | Description                                             |
| --------------------- | -------------------------------------- | ------------------------------------------------------- |
| **Likelihood**        | `ocr_project_data/likelihood`          | CSV/XML outputs with likelihood-based error predictions |
| **Chunks**            | `ocr_project_data/chunks_analysis`     | CSV/XML outputs with chunk-based predictions            |
| **OCR Text**          | `ocr_project_data/html_json_*`         | OCR text HTML/JSON representations                      |
| **PNG GT Overlays**   | `ocr_project_data/text_regions_gt*`    | PNG files for ground truth text regions                 |
| **PNG Pred Overlays** | `ocr_project_data/error_overlay_pred*` | PNG files for predicted error overlays                  |

---

## Enabling / Disabling Methods

In `app.py`, methods are controlled via:

```python
ANALYSIS_METHODS = {
    'likelihood': { 'available': False, ... },
    'chunks': { 'available': False, ... },
    'supervised': { 'available': True, ... }
}
```

* Set `'available': True` to enable in the GUI
* Set `'available': False` to hide from the GUI and disable API access

---

## Running the App

### **1. First Run (Supervised)**

When starting the app for the first time:

* Generates word-level labels (if missing)
* Computes page error rates
* Trains and saves the supervised model
* Generates predicted XMLs
* Scores books and saves PNG visualizations

### **2. Start the Flask Server**

```bash
python app.py
```

### **3. Open in Browser**

```
http://localhost:5000
```

---

##  Using the Web App

* **Method Selector**: Choose between enabled methods
* **Sidebar**: Navigate through document pages
* **Main View**: Display page image, OCR text, or overlays
* **Right Panel**: See statistics 
* **Left Panel**: Search and sort based on the predicted scores

* **Evaluation**: Opens the dataset visualization interface.
Compares predicted error rates to ground truth and automatic evaluations.
Shows side-by-side plots and correlation metrics so you can see how closely each method matches the real annotated data.

* **Average Score Books**: Opens the book level statistics view. Displays per-book mean score, median score, and number of pages.

    Outputs:

    * `book_summaries.csv` → per-book average, median
    * `pagescores.csv` → all page scores


---

## API Endpoints

| Endpoint                           | Description                     |
| ---------------------------------- | ------------------------------- |
| `GET /api/pages?method=supervised` | List processed pages            |
| `POST /api/reload`                 | Reload pages for a given method |
| `GET /api/books/summaries`         | Book-level statistics           |
| `GET /api/books/page-scores`       | Page-level scores               |

---

## Notes

* For **unsupervised methods**, prepare `likelihood` or `chunks` result files in their respective folders.
* For **supervised**, keeping the model file and predicted XMLs allows skipping retraining and regeneration.
* You can toggle methods on/off via `available` flags in `ANALYSIS_METHODS`.


