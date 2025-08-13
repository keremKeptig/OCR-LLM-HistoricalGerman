from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime
import glob
import re
import csv
from pathlib import Path

from data_set_visualization.data_set_visualization import data_viz_bp
from supervised_runner import run_pipeline_supervised_model
from helpers import (
    get_file_size_formatted,
    load_ocr_text_data, load_error_predictions_csv,
    parse_page_xml, map_coordinates,
    map_errors_to_coordinates, generate_ocr_stats
)

app = Flask(__name__, static_url_path='/static')
CORS(app)
app.register_blueprint(data_viz_bp)

APP_ROOT = Path(__file__).resolve().parent

app.config['PIPELINE_GT_DIR'] = str(APP_ROOT / 'ocr_project_data' / 'd2_0001-0100_with_marginalia')
app.config['PIPELINE_INPUT_DIR'] = str(APP_ROOT / 'ocr_project_data' / 'd2_0001-0100_without_marginalia')
app.config['PIPELINE_LABELS_JSONL'] = str(APP_ROOT / 'ocr_project_data' / 'pipeline' / 'word_level_labels.jsonl')
app.config['PIPELINE_PAGE_RATES_CSV'] = str(APP_ROOT / 'ocr_project_data' / 'pipeline' / 'page_error_rates.csv')
app.config['PIPELINE_MODEL_PATH'] = str(APP_ROOT / 'ocr_project_data' / 'pipeline' / 'rf_regressor.joblib')
app.config['PIPELINE_DATASET_PATH'] = str(APP_ROOT / 'data')

app.config['BOOK_SCORES_FOLDER'] = str(APP_ROOT / 'ocr_project_data' / 'book_scores')
app.config['BOOK_PAGE_SCORES_CSV'] = str(Path(app.config['BOOK_SCORES_FOLDER']) / 'pagescores.csv')
app.config['BOOK_SUMMARIES_CSV'] = str(Path(app.config['BOOK_SCORES_FOLDER']) / 'book_summaries.csv')

app.config['IMAGES_FOLDER'] = str(APP_ROOT / 'ocr_project_data' / 'd2_0001-0100_without_marginalia')
app.config['XML_FOLDER'] = str(APP_ROOT / 'ocr_project_data' / 'd2_0001-0100_without_marginalia')
app.config['ERROR_CSV_FOLDER'] = str(APP_ROOT / 'ocr_project_data' / 'likelihood')
app.config['OCR_TEXT_FOLDER'] = str(APP_ROOT / 'ocr_project_data' / 'html_json')
app.config['OCR_TEXT_LIKELIHOOD_FOLDER'] = str(APP_ROOT / 'ocr_project_data' / 'html_json_likelihood')
app.config['OCR_TEXT_CHUNKS_FOLDER'] = str(APP_ROOT / 'ocr_project_data' / 'html_json_chunks')
app.config['OCR_TEXT_SUPERVISED_FOLDER'] = str(APP_ROOT / 'ocr_project_data' / 'html_json_supervised')
app.config['CHUNKS_FOLDER'] = str(APP_ROOT / 'ocr_project_data' / 'chunks_analysis')
app.config['SUPERVISED_FOLDER'] = str(APP_ROOT / 'ocr_project_data' / 'supervised_results')

app.config['TEXT_REGIONS_GT_FOLDER'] = str(APP_ROOT / 'ocr_project_data' / 'text_regions_gt')
app.config['ERROR_OVERLAY_PRED_FOLDER'] = str(APP_ROOT / 'ocr_project_data' / 'error_overlay_pred')

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'pdf'}

pages_data = {}  # stores data by method: {'likelihood': [], 'chunks': [], 'supervised': []}

# Available analysis methods, you can close the methods with available flag
ANALYSIS_METHODS = {
    'likelihood': {
        'name': 'Likelihood Analysis',
        'description': 'Based on detected OCR errors from analysis',
        'available': True,
        'folder_key': 'ERROR_CSV_FOLDER',
        'file_suffix': '.xml',
        'text_regions_gt_folder': 'ocr_project_data/text_regions_gt',
        'error_overlay_pred_folder': 'ocr_project_data/error_overlay_pred',
        'ocr_text_folder_key': 'OCR_TEXT_LIKELIHOOD_FOLDER'
    },
    'chunks': {
        'name': 'Chunks Analysis',
        'description': 'Analyzes text segments and contextual patterns',
        'available': True, 
        'folder_key': 'CHUNKS_FOLDER',
        'file_suffix': '.xml',
        'text_regions_gt_folder': 'ocr_project_data/text_regions_gt_chunks',
        'error_overlay_pred_folder': 'ocr_project_data/error_overlay_pred_chunks',
        'ocr_text_folder_key': 'OCR_TEXT_CHUNKS_FOLDER'
    },
    'supervised': {
        'name': 'Supervised Approach',
        'description': 'Builds a regression head on DistilBERT to predict error counts.',
        'available': True,
        'folder_key': 'SUPERVISED_FOLDER',
        'file_suffix': '.xml'
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/images/<filename>')
def serve_image(filename):
    """ images from the images folder"""
    return send_from_directory(app.config['IMAGES_FOLDER'], filename)

@app.route('/text_regions_gt/<method>/<filename>')
def serve_text_regions_gt(method, filename):
    """Serve text regions GT PNG files from method-specific folder"""
    if method == 'supervised':
        return jsonify({'error': 'No text regions GT available for supervised method'}), 404
    
    png_folders = get_method_png_folders(method)
    folder = png_folders['text_regions_gt_folder']
    return send_from_directory(folder, filename)

@app.route('/error_overlay_pred/<method>/<filename>')
def serve_error_overlay_pred(method, filename):
    """Serve error overlay pred PNG files from method-specific folder"""
    if method == 'supervised':
        return jsonify({'error': 'No error overlay pred available'}), 404
    
    png_folders = get_method_png_folders(method)
    folder = png_folders['error_overlay_pred_folder']
    return send_from_directory(folder, filename)

@app.route('/api/pages')
def get_pages():
    """Get all processed pages for a specific method"""
    method = request.args.get('method', 'likelihood')
    
    if method not in ANALYSIS_METHODS:
        return jsonify({'error': f'Invalid analysis method: {method}'}), 400
    
    if not ANALYSIS_METHODS[method]['available']:
        return jsonify({'error': f'Analysis method {method} is not available yet'}), 400
    
    if method not in pages_data or not pages_data[method]:
        load_images_from_folder(method)
    
    result_pages = pages_data.get(method, [])
    print("DEBUG supervised count:", len(result_pages))


    return jsonify(result_pages)

@app.route('/api/page/<int:page_id>')
def get_page(page_id):
    """Get specific page data"""
    method = request.args.get('method', 'likelihood')
    
    if method not in pages_data:
        return jsonify({'error': 'Method data not loaded'}), 404
    
    page = next((p for p in pages_data[method] if p['id'] == page_id), None)
    if page:
        return jsonify(page)
    return jsonify({'error': 'Page not found'}), 404

@app.route('/api/page/<int:page_id>/ocr-text')
def get_page_ocr_text(page_id):
    """Get OCR text data for a specific page"""
    method = request.args.get('method', 'likelihood')
    
    if method not in pages_data:
        return jsonify({'error': 'Method data not loaded'}), 404
    
    page = next((p for p in pages_data[method] if p['id'] == page_id), None)
    if not page:
        return jsonify({'error': 'Page not found'}), 404
    
    if not page.get('hasOcrText'):
        return jsonify({'error': 'No OCR text data available for this page'}), 404
    
    return jsonify({
        'filename': page.get('ocrText', {}).get('Filename', ''),
        'coloredTextHTML': page.get('ocrText', {}).get('ColoredTextHTML', ''),
        'hasOcrText': True
    })


@app.route('/api/reload', methods=['POST'])
def reload_images():
    """Reload images from the images folder"""
    method = request.json.get('method', 'likelihood') if request.is_json else request.args.get('method', 'likelihood')
    
    if method not in ANALYSIS_METHODS:
        return jsonify({'error': f'Invalid analysis method: {method}'}), 400
    
    if not ANALYSIS_METHODS[method]['available']:
        return jsonify({'error': f'Analysis method {method} is not available yet'}), 400
    
    load_images_from_folder(method)
    
    return jsonify({
        'message': f'Reloaded {len(pages_data.get(method, []))} images with {method} analysis',
        'count': len(pages_data.get(method, [])),
        'method': method
    })

# PNG folders, skip for supervised
for method_id, method_config in ANALYSIS_METHODS.items():
    if method_config.get('available', False) and method_id != 'supervised':
        gt_folder = method_config.get('text_regions_gt_folder')
        pred_folder = method_config.get('error_overlay_pred_folder')
        
        if gt_folder and not os.path.exists(gt_folder):
            os.makedirs(gt_folder, exist_ok=True)
        

        if pred_folder and not os.path.exists(pred_folder):
            os.makedirs(pred_folder, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_png_representations(base_name, method='likelihood'):
    """Load PNG colored representations if they exist - method-specific folders"""
    if method == 'supervised':
        return {
            'text_regions_gt': None,
            'error_overlay_pred': None
        }
        
    png_folders = get_method_png_folders(method)
    
    png_data = {
        'text_regions_gt': None,
        'error_overlay_pred': None
    }
    
    # text regions GT PNG in specific folder
    gt_folder = png_folders['text_regions_gt_folder']
    if gt_folder:
        gt_pattern = os.path.join(gt_folder, f'{base_name}*.png')
        gt_files = glob.glob(gt_pattern)
        if gt_files:
            png_data['text_regions_gt'] = os.path.basename(gt_files[0])
    
    # Error overlay pred PNG in specific folder
    pred_folder = png_folders['error_overlay_pred_folder']
    if pred_folder:
        pred_pattern = os.path.join(pred_folder, f'{base_name}*.png')
        pred_files = glob.glob(pred_pattern)
        if pred_files:
            png_data['error_overlay_pred'] = os.path.basename(pred_files[0])
            
    return png_data

def get_method_png_folders(method):
    """Get the PNG folders for a specific analysis method"""
    if method == 'supervised':
        return {
            'text_regions_gt_folder': None,
            'error_overlay_pred_folder': None
        }
        
    method_config = ANALYSIS_METHODS.get(method, {})
    return {
        'text_regions_gt_folder': method_config.get('text_regions_gt_folder', 'text_regions_gt'),
        'error_overlay_pred_folder': method_config.get('error_overlay_pred_folder', 'error_overlay_pred')
    }

def load_images_from_folder(method='likelihood'):
    global pages_data
        
    if method not in pages_data:
        pages_data[method] = []
    else:
        pages_data[method] = [] 
        
    # Load OCR text data from specific folder, skip for supervised
    if method != 'supervised':
        method_config = ANALYSIS_METHODS.get(method, {})
        ocr_text_folder_key = method_config.get('ocr_text_folder_key', 'OCR_TEXT_FOLDER')
        ocr_text_folder = app.config.get(ocr_text_folder_key, app.config['OCR_TEXT_FOLDER'])
        ocr_text_data = load_ocr_text_data(ocr_text_folder)
    else:
        # no need ocr text data for supervised
        ocr_text_data = {}
        
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    for extension in image_extensions:
        pattern = os.path.join(app.config['IMAGES_FOLDER'], extension)
        image_files.extend(glob.glob(pattern, recursive=False))
    image_files = [f for f in image_files if not f.lower().endswith(('.nrm.png'))]

    # Sort files for consistency
    image_files.sort()

    
    # xml files
    xml_pattern = os.path.join(app.config['XML_FOLDER'], '*.xml')
    xml_files = glob.glob(xml_pattern, recursive=False)
    xml_basenames = {os.path.basename(f) for f in xml_files}
    
    # specific analysis files
    method_config = ANALYSIS_METHODS.get(method, {})
    if not method_config.get('available', False):
        return
    
    analysis_folder = app.config[method_config['folder_key']]
    file_suffix = method_config['file_suffix']

    
    analysis_pattern = os.path.join(analysis_folder, f'*{file_suffix}')
    analysis_files = glob.glob(analysis_pattern, recursive=False)
    analysis_basenames = {os.path.basename(f) for f in analysis_files}
    
    successful_loads = 0
    xml_processed = 0
    analysis_processed = 0
    ocr_text_processed = 0
    mock_data_used = 0
    
    for idx, image_path in enumerate(image_files, 1):
        try:
            filename = os.path.basename(image_path)
            base_name = filename.rsplit('.', 1)[0] 
            base_name = re.split(r"\.", base_name)[0]

            xml_data = None
            xml_filename = base_name + '.xml'
            xml_path = None
            if xml_filename in xml_basenames:
                xml_path = os.path.join(app.config['XML_FOLDER'], xml_filename)
                xml_data = parse_page_xml(xml_path)
                if xml_data:
                    xml_processed += 1
        
            ocr_text = None
            if base_name in ocr_text_data:
                ocr_text = ocr_text_data[base_name]
                ocr_text_processed += 1
            
            if method != 'supervised':
                png_representations = load_png_representations(base_name, method)
            else:
                png_representations = {
                    'text_regions_gt': None,
                    'error_overlay_pred': None
                }
            
            prediction_errors = []
            analysis_filename = base_name + file_suffix

            analysis_folder_supervised = app.config['SUPERVISED_FOLDER']
            file_suffix = method_config['file_suffix']

            if analysis_filename in analysis_basenames:
                analysis_path = os.path.join(analysis_folder, analysis_filename)
                analysis_for_supervised = os.path.join(analysis_folder_supervised, analysis_filename)

                prediction_errors = load_error_predictions_csv(analysis_path, method)
                if prediction_errors:
                    analysis_processed += 1
            
            # all lines for display
            display_errors = []
            if xml_data and xml_path:
                display_errors = map_coordinates(xml_path)
                        
            prediction_mapped_errors = []
            if xml_data and prediction_errors:
                prediction_mapped_errors, total_prediction_errors = map_errors_to_coordinates(prediction_errors, xml_data, method)
                stats = generate_ocr_stats(prediction_mapped_errors, total_prediction_errors, xml_data, method)
            elif prediction_errors:
                # errors without coordinates
                prediction_mapped_errors = prediction_errors
                stats = generate_ocr_stats(prediction_errors, len(prediction_errors), xml_data, method)
            else:
                # No real prediction data, empty stats
                stats = {
                    "overallScore": 1.0,
                    "wordCount": 0,
                    "errorCount": 0,
                    "problematicAreas": 0,
                    "analysisMethod": method
                }
                mock_data_used += 1


            pages_count = 1
            if xml_data and xml_data.get('text_lines'):
                pages_count = 1

            file_size = get_file_size_formatted(image_path)
            modification_date = datetime.fromtimestamp(os.path.getmtime(image_path)).strftime('%Y-%m-%d')

            prediction_errors_supervised = load_error_predictions_csv(analysis_for_supervised, method='supervised') 
            stats_supervised = generate_ocr_stats(prediction_errors_supervised, len(prediction_errors_supervised), xml_data, method='supervised')

            page_data = {
                "id": idx,
                "filename": filename,
                "imageUrl": f"/images/{filename}",
                "filePath": image_path,
                "stats": stats,
                "displayErrors": display_errors,  
                "predictionErrors": prediction_mapped_errors,  
                "xmlData": xml_data,
                "ocrText": ocr_text,
                "pngRepresentations": png_representations,
                "hasXmlData": xml_data is not None,
                "hasOcrText": ocr_text is not None,
                "hasRealErrors": len(prediction_errors) > 0,
                "hasTextRegionsGT": png_representations['text_regions_gt'] is not None,
                "hasErrorOverlayPred": png_representations['error_overlay_pred'] is not None,
                "errorSource": "real" if prediction_errors else "none",
                "analysisMethod": method,
                "pngFolders": get_method_png_folders(method),  
                
                "fileSize": file_size,
                "pages": pages_count,
                "quality": stats_supervised["overallScore"],
                "lastModified": modification_date,
                "hasErrors": len(prediction_mapped_errors) > 0,
                "errorCount": stats_supervised["errorCount"]
            }
            
            pages_data[method].append(page_data)
            successful_loads += 1 
            
        except Exception as e:
            print(e)
            continue


def read_csv_as_dicts(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]

@app.route('/books')
def books_page():
    return render_template('books.html')


@app.route('/api/books/summaries')
def api_book_summaries():
    rows = read_csv_as_dicts(app.config['BOOK_SUMMARIES_CSV'])

    for r in rows:
        for k in ('num_pages', 'mean_score', 'median_score', 'best_score', 'worst_score'):
            v = str(r.get(k, '')).strip()
            if not v or v.lower() in ('nan', 'none'):
                r[k] = 0 if k == 'num_pages' else None
            else:
                try:
                    r[k] = int(float(v)) if k == 'num_pages' else float(v)
                except Exception:
                    r[k] = 0 if k == 'num_pages' else None

        r['best_page_id']  = r.get('best_page_id') or None
        r['worst_page_id'] = r.get('worst_page_id') or None

    return jsonify(rows)


@app.route('/api/books/page-scores')
def api_book_page_scores():
    book_id = (request.args.get('book_id') or '').strip()
    sort = request.args.get('sort', 'asc') 

    rows = read_csv_as_dicts(app.config['BOOK_PAGE_SCORES_CSV'])

    normalized_rows = []
    for row in rows:
        clean_row = { (k.strip() if isinstance(k,str) else k): (v.strip() if isinstance(v,str) else v) for k,v in row.items() }
        if 'score' in clean_row and clean_row['score'] not in (None, ''):
            try:
                clean_row['score'] = float(clean_row['score'])
            except ValueError:
                clean_row['score'] = None
        normalized_rows.append(clean_row)

    rows = [row for row in normalized_rows if row.get('score') is not None]

    if book_id:
        rows = [r for r in rows if (r.get('book_id','').strip() == book_id)]

    rows.sort(key=lambda r: r['score'], reverse=(sort == 'desc'))
    return jsonify(rows)


if __name__ == '__main__':
    # Supervised pipeline at the start
    try:
        res = run_pipeline_supervised_model(
            gt_dir=app.config['PIPELINE_GT_DIR'],
            input_dir=app.config['PIPELINE_INPUT_DIR'],
            labels_jsonl=app.config['PIPELINE_LABELS_JSONL'],
            page_error_rates_csv=app.config['PIPELINE_PAGE_RATES_CSV'],
            model_path=app.config['PIPELINE_MODEL_PATH'],
            outdir=app.config['BOOK_SCORES_FOLDER'],      
            xml_outputs=app.config['SUPERVISED_FOLDER'],  
            dataset_path=app.config['PIPELINE_DATASET_PATH'],
            score_book_dirs=[app.config['PIPELINE_INPUT_DIR']]
        )
    except Exception as e:
        app.logger.exception('Pipeline failed at the beginning')
    
    for method_id, method_info in ANALYSIS_METHODS.items():
        if method_info.get('available', False):
            load_images_from_folder(method_id)

    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
