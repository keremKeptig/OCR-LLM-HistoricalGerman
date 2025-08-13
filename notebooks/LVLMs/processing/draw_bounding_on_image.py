import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import cv2
import numpy as np
import os

# === Step 1: Load XML and parse regions + text ===
def parse_pagexml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}
    page = root.find('pc:Page', ns)

    regions = []
    for region in page.findall('pc:TextRegion', ns):
        region_type = region.attrib.get('type', 'unknown')
        coords_elem = region.find('pc:Coords', ns)
        lines = region.findall('.//pc:TextLine', ns)
        if coords_elem is None:
            continue
        points_str = coords_elem.attrib['points']
        points = [tuple(map(int, p.split(','))) for p in points_str.split()]
        
        # Extract the actual text lines
        text_lines = []
        for line in lines:
            unicode_elem = line.find('.//pc:Unicode', ns)
            if unicode_elem is not None and unicode_elem.text:
                text_lines.append(unicode_elem.text.strip())
        
        regions.append({
            'type': region_type,
            'points': points,
            'text': "\n".join(text_lines)
        })
    return regions

# === Step 2: Draw boxes and save region text ===
def draw_boxes_and_save_text(image_path, regions, save_img='annotated_output.png', save_txt='region_texts.txt'):
    image_pil = Image.open(image_path).convert("RGB")
    overlay = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))  # Transparent overlay
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(image_pil)

    with open(save_txt, 'w', encoding='utf-8') as f:
        for i, region in enumerate(regions):
            points = region['points']
            polygon = [tuple(p) for p in points]
            region_type = region['type']
            text = region['text']

            # Set color based on type
            if region_type != 'paragraph':
                outline_color = (0, 0, 255)        # Blue outline
                fill_color = (0, 0, 255, 80)       # Semi-transparent blue fill
            else:
                outline_color = (255, 0, 0)        # Red outline
                fill_color = (255, 0, 0, 80)       # Semi-transparent red fill

            # Draw filled polygon on overlay (RGBA)
            draw_overlay.polygon(polygon, fill=fill_color)

            # Draw outline on original image
            draw.line(polygon + [polygon[0]], fill=outline_color, width=2)
            draw.text(polygon[0], region_type, fill=outline_color)

            # Save region text
            f.write(f"[Region {i+1}] Type: {region_type}\n")
            f.write(f"Polygon: {points}\n")
            f.write(text + "\n\n")

    # Composite overlay onto the base image
    image_combined = Image.alpha_composite(image_pil.convert("RGBA"), overlay)
    image_combined.convert("RGB").save(save_img)
    print(f"✅ Saved annotated image to: {save_img}")
    print(f"✅ Saved region text to: {save_txt}")

# === Example usage ===
project_root = os.path.abspath(os.path.join(os.getcwd(), '../../..'))

xml_path = os.path.join(project_root, 'data/d2_0001-0100_with_marginalia/1724666584_00000024.xml')
image_path = os.path.join(project_root, 'data/d2_0001-0100_with_marginalia/1724666584_00000024.bin.png')

regions = parse_pagexml(xml_path)
draw_boxes_and_save_text(image_path, regions)
