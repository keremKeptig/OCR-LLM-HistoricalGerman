import os
import json
import glob
 
import xml.etree.ElementTree as ET

from PIL import Image

def load_ocr_text_data(text_folder):
    """Load OCR text data from JSON files"""
    ocr_text_data = {}

    try:
        json_pattern = os.path.join(text_folder, '*.json')
        json_files = glob.glob(json_pattern, recursive=False)

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, list):
                    for item in data:
                        if 'Filename' in item and 'ColoredTextHTML' in item:
                            filename = item['Filename']
                            ocr_text_data[filename] = item
                elif isinstance(data, dict) and 'Filename' in data:
                    filename = data['Filename']
                    ocr_text_data[filename] = data
                                        
            except Exception as e:
                print(f"Error loading OCR text from {json_file}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error accessing OCR text folder {text_folder}: {str(e)}")
    
    return ocr_text_data

def map_coordinates(xml_path):
    """Read the original XML file and show all line_ids as errors with their coordinates"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        namespaces = [
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2016-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'},
            {}  
        ]
        
        page = None
        ns = {}
    
        for namespace in namespaces:
            if namespace:
                page = root.find('.//page:Page', namespace)
                if page is not None:
                    ns = namespace
                    break
            else:
                page = root.find('.//Page')
                if page is not None:
                    ns = {}
                    break
        
        if page is None:
            return []
        
        errors = []
        
        # Find all TextLine elements
        text_lines = page.findall('.//page:TextLine' if ns else './/TextLine', ns)
                
        for text_line in text_lines:
            line_id = text_line.get('id', '')
            
            if not line_id:
                continue
                
            # Get line coordinates
            coords_elem = text_line.find('.//page:Coords' if ns else './/Coords', ns)
            coords = []
            if coords_elem is not None:
                points_str = coords_elem.get('points', '')
                if points_str:
                    try:
                        for point in points_str.split():
                            if ',' in point:
                                x, y = point.split(',')
                                coords.append([int(x), int(y)])
                    except ValueError as e:
                        print(f"Invalid coordinates {line_id}: {e}")
                        continue
            
            baseline_elem = text_line.find('.//page:Baseline' if ns else './/Baseline', ns)
            baseline = []
            if baseline_elem is not None:
                points_str = baseline_elem.get('points', '')
                if points_str:
                    try:
                        for point in points_str.split():
                            if ',' in point:
                                x, y = point.split(',')
                                baseline.append([int(x), int(y)])
                    except ValueError as e:
                        print(f"Invalid baseline coordinates {line_id}: {e}")
            
            # Get text content
            text_equiv = text_line.find('.//page:TextEquiv/page:Unicode' if ns else './/TextEquiv/Unicode', ns)
            text = ''
            if text_equiv is not None and text_equiv.text:
                text = text_equiv.text
            elif text_equiv is not None:
                text = ''
            
            if coords and len(coords) >= 2:
                error = {
                    'type': 'line_error',
                    'method': 'xml_all_lines',
                    'line_id': line_id,
                    'coords': coords,
                    'baseline': baseline,
                    'text': text,
                    'error_count': 1,
                    'error_words': [text] if text else [],
                    'errors': [{
                        'word': text,
                        'line_id': line_id,
                        'context': f'Full line from XML: {text[:50]}...' if len(text) > 50 else f'Full line from XML: {text}',
                        'start_char': 0,
                        'end_char': len(text),
                        'log_prob': None,
                        'type': 'xml_line',
                        'method': 'xml_all_lines'
                    }],
                    'context': f'Line {line_id} from XML',
                    'log_prob': None
                }
                errors.append(error)
            else:
                print(f"Warning: Line {line_id} has invalid or missing coordinates")
        
        return errors
        
    except ET.ParseError as e:
        print(f"Error: XML parse error in {xml_path}: {str(e)}")
        return []


# supervised functions
def load_supervised_page_error(xml_path):
    """Load page-level error score from supervised XML file"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        page_element = None
        if root.tag == 'Page':
            page_element = root
        else:
            page_element = root.find('.//Page')
        
        if page_element is None:
            return None
        
        page_error_score = page_element.get('overwritePageError')
        if page_error_score is None:
            return None
        
        try:
            error_score = float(page_error_score)
            return error_score
        except ValueError:
            return None
            
    except ET.ParseError as e:
        return None


def load_supervised_error_predictions(xml_path):
    """Load supervised predictions - returns page-level error info"""
    page_error_score = load_supervised_page_error(xml_path)
    if page_error_score is None:
        return []
        
    # Single "error" shows the page-level prediction
    return [{
        'page_error_score': page_error_score,
        'method': 'supervised',
        'type': 'page_level_error',
        'context': f'Page-level error score from supervised model: {page_error_score:.4f}'
    }]

def load_error_predictions_csv(csv_path, method='likelihood'):
    """Load OCR error predictions from CSV file based on analysis method"""
    # load_error_predictions_xml same for unsupervised approach
    # for supervised, it does not include line-level details
    try:
        if method == 'likelihood':
            return load_error_predictions_xml(xml_path=csv_path)
        elif method == 'chunks':
            return load_error_predictions_xml(xml_path=csv_path)
        elif method == 'supervised':
            return load_supervised_error_predictions(xml_path=csv_path)  
        else:
            return []
        
    except Exception as e:
        return []

def load_error_predictions_xml(xml_path, error_score_threshold=0.5):
    """Load OCR error predictions from XML file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    if "}" in root.tag:                   
        ns = {"p": root.tag.split("}")[0].strip("{")}
        tline_xpath = ".//p:TextLine"
        word_xpath  = "p:Word"
    else:                                 
        ns = {}
        tline_xpath = ".//TextLine"
        word_xpath  = "Word"

    results = []
    total_words = 0
    
    for line in root.findall(tline_xpath, ns):
        line_id = line.get("id", "")
        for w in line.findall(word_xpath, ns):
            total_words += 1
            word_text = w.get("text", "").strip()
            error_type = w.get("error_type", "")
            error_score_str = w.get("error_score", "0.0")
            
            try:
                error_score = float(error_score_str)
            except (ValueError, TypeError):
                error_score = 0.0
            
            is_error = False
            error_reason = ""
            
            if error_type == "error":
                is_error = True
                error_reason = f"error_type={error_type}"
            
            elif error_score > error_score_threshold:
                is_error = True
                error_reason = f"error_score={error_score}"
            
            if is_error:
                results.append({
                    "word": word_text,
                    "line_id": line_id,
                    "error_type": error_type,
                    "error_score": error_score,
                    "error_reason": error_reason,
                    "method": "xml_error_detection"
                })
    
    return results

def map_errors_to_coordinates(errors, xml_data, method='likelihood'):
    """Map CSV errors to XML line coordinates"""
    # load_error_predictions_xml same for unsupervised approach
    # for supervised, it does not include line-level details
    if not xml_data or not errors:
        return [], 0
    
    text_lines = xml_data.get('text_lines', [])
    lines_by_id = {line['id']: line for line in text_lines}
    
    if method == 'likelihood':
        return map_likelihood_errors_to_coordinates(errors, lines_by_id)
    elif method == 'chunks':
        return map_chunks_errors_to_coordinates(errors, lines_by_id)
    elif method == 'supervised':
        return map_supervised_errors_to_coordinates(errors, lines_by_id)
    else:
        return [], 0

def map_likelihood_errors_to_coordinates(errors, lines_by_id):
    """Map likelihood analysis errors to coordinates"""
    # Group errors by line_id
    errors_by_line = {}
    for error in errors:
        line_id = error['line_id']
        if line_id not in errors_by_line:
            errors_by_line[line_id] = []
        errors_by_line[line_id].append(error)

    mapped_errors = []
    
    for line_id, line_errors in errors_by_line.items():
        if line_id not in lines_by_id:
            continue
            
        line_data = lines_by_id[line_id]
        line_text = line_data.get('text', '')
        line_coords = line_data.get('coords', [])
        
        if not line_coords or len(line_coords) < 4:
            continue
        
        error_words = [err.get('word', '') for err in line_errors if err.get('word')]
        representative_error = line_errors[0]
        
        line_error = {
            'type': 'prediction_error',
            'method': 'likelihood',
            'line_id': line_id,
            'coords': line_coords,
            'text': line_text,
            'error_count': len(line_errors),
            'error_words': error_words,
            'errors': line_errors,
            'context': representative_error.get('context', ''),
            'log_prob': representative_error.get('log_prob', None)
        }
        mapped_errors.append(line_error)
    
    return mapped_errors, len(errors)

def map_chunks_errors_to_coordinates(errors, lines_by_id):
    """Map chunks analysis errors to coordinates - SAME LOGIC AS LIKELIHOOD"""
    # Group errors by line_id
    errors_by_line = {}
    for error in errors:
        line_id = error['line_id']
        if line_id not in errors_by_line:
            errors_by_line[line_id] = []
        errors_by_line[line_id].append(error)

    mapped_errors = []
    
    for line_id, line_errors in errors_by_line.items():
        if line_id not in lines_by_id:
            continue
            
        line_data = lines_by_id[line_id]
        line_text = line_data.get('text', '')
        line_coords = line_data.get('coords', [])
        
        if not line_coords or len(line_coords) < 4:
            continue
                
        error_chunks = [err.get('word', '') for err in line_errors if err.get('word')]
        
        representative_error = line_errors[0]
        
        line_error = {
            'type': 'prediction_error',
            'method': 'chunks',
            'line_id': line_id,
            'coords': line_coords,
            'text': line_text,
            'error_count': len(line_errors),
            'error_chunks': error_chunks,  
            'errors': line_errors,
            'context': representative_error.get('context', ''),
            'chunk_count': len(line_errors)  
        }
        mapped_errors.append(line_error)
    
    return mapped_errors, len(errors)

def map_supervised_errors_to_coordinates(errors, lines_by_id):
    """Map supervised page-level error score"""
    if not errors:
        return [], 0
    
    page_error = errors[0]
    page_error_score = page_error.get('page_error_score', 0.0)
        
    # Return the page error data
    mapped_error = {
        'type': 'page_level_prediction',
        'method': 'supervised',
        'page_error_score': page_error_score,
        'context': page_error.get('context', ''),
        'error_count': 1
    }
    
    return [mapped_error], 1  

def parse_page_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        namespaces = [
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2018-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2016-07-15'},
            {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'},
            {} 
        ]
        
        page = None
        ns = {}
        
        for namespace in namespaces:
            if namespace:
                page = root.find('.//page:Page', namespace)
                if page is not None:
                    ns = namespace
                    break
            else:
                page = root.find('.//Page')
                if page is not None:
                    ns = {}
                    break
        
        if page is None:
            return None
        
        image_width = int(page.get('imageWidth', 0))
        image_height = int(page.get('imageHeight', 0))
        image_filename = page.get('imageFilename', '')
        
        text_lines = []
        
        text_regions = page.findall('.//page:TextRegion' if ns else './/TextRegion', ns)
        
        total_lines = 0
        for text_region in text_regions:
            region_id = text_region.get('id', '')
            
            # Extract text lines
            lines_in_region = text_region.findall('.//page:TextLine' if ns else './/TextLine', ns)
            for text_line in lines_in_region:
                line_id = text_line.get('id', '')
                
                # Get line coordinates
                coords_elem = text_line.find('.//page:Coords' if ns else './/Coords', ns)
                coords = []
                if coords_elem is not None:
                    points_str = coords_elem.get('points', '')
                    if points_str:
                        try:
                            for point in points_str.split():
                                if ',' in point:
                                    x, y = point.split(',')
                                    coords.append([int(x), int(y)])
                        except ValueError as e:
                            continue
                
                baseline_elem = text_line.find('.//page:Baseline' if ns else './/Baseline', ns)
                baseline = []
                if baseline_elem is not None:
                    points_str = baseline_elem.get('points', '')
                    if points_str:
                        try:
                            for point in points_str.split():
                                if ',' in point:
                                    x, y = point.split(',')
                                    baseline.append([int(x), int(y)])
                        except ValueError as e:
                            print(f"Warning: Invalid baseline coordinates in line {line_id}: {e}")
                
                # Get text content
                text_equiv = text_line.find('.//page:TextEquiv/page:Unicode' if ns else './/TextEquiv/Unicode', ns)
                text = ''
                if text_equiv is not None and text_equiv.text:
                    text = text_equiv.text
                elif text_equiv is not None:
                    text = ''
                
                if coords and len(coords) >= 2:
                    text_lines.append({
                        'id': line_id,
                        'text': text,
                        'coords': coords,
                        'baseline': baseline
                    })
                    total_lines += 1
        
        return {
            'image_width': image_width,
            'image_height': image_height,
            'image_filename': image_filename,
            'text_lines': text_lines,
            'xml_file': os.path.basename(xml_path)
        }
    
    except ET.ParseError as e:
        print(f"Error: XML parse error in {os.path.basename(xml_path)}: {str(e)}")
        return None

def generate_ocr_stats(prediction_errors, total_prediction_errors, xml_data, method='likelihood'):
    text_lines = xml_data.get('text_lines', []) if xml_data else []
    
    total_words = 0
    for line in text_lines:
        if line.get('text'):
            total_words += len(line['text'].split())
    
    if method == 'supervised':
        if prediction_errors and len(prediction_errors) > 0:
            # Get the page error score from the mapped error
            supervised_page_error = prediction_errors[0].get('page_error_score', 0.0)
            error_rate = supervised_page_error  
            overall_score = max(0.0, min(1.0, 1.0 - error_rate))  # quality score
            error_count = 1 
            problematic_areas = 1 if error_rate > 0.5 else 0  # High error rate = problematic
            
        else:
            # No prediction available 
            supervised_page_error = 0.0
            overall_score = 0.0
            error_count = 0
            problematic_areas = 0
        
        return {
            "overallScore": overall_score,
            "wordCount": total_words,
            "errorCount": error_count,
            "problematicAreas": problematic_areas,
            "analysisMethod": method,
            "pageErrorScore": supervised_page_error 
        }
    
    error_count = total_prediction_errors
    
    if total_words > 0:
        error_rate = error_count / total_words
        
        # score calculation
        if method == 'likelihood':
            overall_score = 1.0 - (error_rate)
        elif method == 'chunks':
            overall_score = 1.0 - (error_rate)
        else:
            overall_score = 1.0 - (error_rate)
    else:
        overall_score = 1.0
        error_rate = 0.1
    
    # Count problematic areas
    lines_with_errors = {}
    for error in prediction_errors:
        line_id = error.get('line_id', '')
        if line_id:
            lines_with_errors[line_id] = lines_with_errors.get(line_id, 0) + 1
    
    problematic_areas = len([count for count in lines_with_errors.values() if count > 1])
    
    stats = {
        "overallScore": overall_score,
        "wordCount": total_words,
        "errorCount": error_count,
        "problematicAreas": problematic_areas,
        "analysisMethod": method
    }
    
    return stats

def get_image_info(image_path):
    try:
        with Image.open(image_path) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "size_mb": round(os.path.getsize(image_path) / (1024 * 1024), 2)
            }
    except Exception:
        return {}

def get_file_size_formatted(file_path):
    """Get formatted file size"""
    try:
        size_bytes = os.path.getsize(file_path)
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    except:
        return "Unknown"

