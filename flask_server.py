from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from io import BytesIO
import numpy as np
from PIL import Image
from flask import send_file
import json
import pycocotools.mask as mask_util 
from matplotlib import colormaps
import re 

def normalize_labels(raw_labels):
    suffixes_to_strip = ["-other-merged", "-merged", "-other", "-stuff"]

    def clean(label):
        label = label.lower().strip()
        for suffix in suffixes_to_strip:
            if label.endswith(suffix):
                label = label[: -len(suffix)].strip()
                break
        return label

    cleaned = [clean(label) for label in raw_labels]
    frequencies = {label: cleaned.count(label) for label in set(cleaned)}
    counts = {}

    result = []
    
    for label in cleaned:
        counts[label] = counts.get(label, 0) + 1
        if frequencies[label] > 1:
            result.append(f"{label}-{counts[label]}")
        else:
            result.append(label)
    return result

def _get_rle_segments_and_labels_for_image(annotation_map_key_for_compute, base_filename_no_ext_for_mask):
    """
    Helper function to get RLE encoded segments and their ordered normalized labels for a given image.
    - annotation_map_key_for_compute: Key for panoptic_data_store["annotations_map"] (e.g., "000000xxxxxx.png")
    - base_filename_no_ext_for_mask: Base filename (e.g., "000000xxxxxx") to locate the mask PNG.

    Returns a tuple: (list_of_segment_details, list_of_ordered_normalized_labels)
    Each item in list_of_segment_details:
    {
        "mask_segment_id": int,      // ID from the panoptic mask
        "coco_category_id": int,     // Original COCO category ID
        "raw_category_name": str,
        "normalized_label": str,
        "rle_segmentation": dict     // RLE encoded mask {'size': [h, w], 'counts': '...'}
    }
    """
    image_annotation_entry = panoptic_data_store["annotations_map"].get(annotation_map_key_for_compute)
    if not image_annotation_entry:
        print(f"[WARNING] _get_rle_segments_and_labels_for_image: No annotation entry found for key: {annotation_map_key_for_compute}.")
        return [], []

    mask_path = f'data/coco_pan_annotations/panoptic_train2017/{base_filename_no_ext_for_mask}.png'
    if not os.path.exists(mask_path):
        print(f"[ERROR] _get_rle_segments_and_labels_for_image: Mask file not found at {mask_path}")
        return [], []

    try:
        mask_pil = Image.open(mask_path).convert("RGB")
        mask_rgb_np = np.array(mask_pil)
        # Decode RGB to single segment IDs (R + G*256 + B*256*256)
        mask_ids_np = mask_rgb_np[:, :, 0].astype(np.uint32) + \
                      (mask_rgb_np[:, :, 1].astype(np.uint32) << 8) + \
                      (mask_rgb_np[:, :, 2].astype(np.uint32) << 16)
    except Exception as e:
        print(f"[ERROR] _get_rle_segments_and_labels_for_image: Failed to load or process mask {mask_path}: {e}")
        return [], []

    json_segments_info = image_annotation_entry.get("segments_info", [])
    # Map segment_info's 'id' (which is the value in the mask) to its 'category_id'
    json_mask_id_to_coco_category_id = {seg['id']: seg['category_id'] for seg in json_segments_info if 'id' in seg and 'category_id' in seg}

    category_id_map = panoptic_data_store["category_id_to_name_map"]
    
    segments_for_json = []
    raw_names_for_normalization = []

    unique_mask_ids = sorted(list(np.unique(mask_ids_np))) # Sort for consistent order

    for mask_id_val in unique_mask_ids:
        if mask_id_val == 0:  # Skip background
            continue

        coco_category_id = json_mask_id_to_coco_category_id.get(mask_id_val)
        raw_category_name = "Unknown Category"
        if coco_category_id is not None:
            raw_category_name = category_id_map.get(coco_category_id, f"Unknown COCO ID:{coco_category_id}")
        else: # mask_id_val not found in json_segments_info mapping
            raw_category_name = f"Category for Mask ID {mask_id_val} not in JSON segments_info"

        binary_mask_for_rle = np.asfortranarray((mask_ids_np == mask_id_val).astype(np.uint8))
        rle = mask_util.encode(binary_mask_for_rle)
        rle['counts'] = rle['counts'].decode('utf-8') # Decode bytes to string for JSON

        segments_for_json.append({
            "mask_segment_id": int(mask_id_val), # Ensure it's Python int
            "coco_category_id": coco_category_id, # Can be None if not found
            "raw_category_name": raw_category_name,
            "rle_segmentation": rle
        })
        raw_names_for_normalization.append(raw_category_name)

    ordered_normalized_labels = normalize_labels(raw_names_for_normalization)

    # Add normalized_label to each segment detail
    if len(ordered_normalized_labels) == len(segments_for_json):
        for i, seg_detail in enumerate(segments_for_json):
            seg_detail["normalized_label"] = ordered_normalized_labels[i]
    else: # Should not happen if logic is correct
        print(f"[ERROR] Length mismatch between normalized labels and segments for {annotation_map_key_for_compute}")
        # Add raw_name as fallback for normalized_label if lengths mismatch
        for i, seg_detail in enumerate(segments_for_json):
            seg_detail["normalized_label"] = seg_detail["raw_category_name"]

    return segments_for_json, ordered_normalized_labels

app = Flask(__name__)
CORS(app)

# Initialize a global store for panoptic annotation data
panoptic_data_store = {
    "annotations_map": {},  # Stores image annotations keyed by pan_seg_file_name
    "category_id_to_name_map": {}, # Maps COCO category_id to category_name
    "thing_class_names": [], # List of names for thing categories
    "stuff_class_names": [],  # List of names for stuff categories
    "last_displayed_image_key": None, # Annotation key of the last image whose categories were fetched
    "last_displayed_image_normalized_labels": [] # Normalized labels for the last_displayed_image_key
}

# Define the path to your panoptic annotation JSON file
# Ensure this file is in the same directory as flask_server.py or provide the correct path
PANOPTIC_JSON_PATH = 'data/coco_pan_annotations/panoptic_train2017.json'

try:
    with open(PANOPTIC_JSON_PATH, 'r') as f:
        raw_panoptic_data = json.load(f)

    # Load categories and create a direct mapping from category_id to name
    categories_list = raw_panoptic_data.get("categories", [])
    panoptic_data_store["category_id_to_name_map"] = {
        cat['id']: cat['name'] for cat in categories_list
    }
    # Populate lists of names for thing and stuff classes
    panoptic_data_store["thing_class_names"] = [
        cat['name'] for cat in categories_list if cat.get('isthing') == 1
    ]
    panoptic_data_store["stuff_class_names"] = [
        cat['name'] for cat in categories_list if cat.get('isthing') == 0
    ]
    
    # As per your description, image annotations are under the 'data' key
    image_annotations_list = raw_panoptic_data.get('annotations', [])
    
    for img_anno_entry in image_annotations_list:
        if "file_name" in img_anno_entry and "segments_info" in img_anno_entry: # Use file_name
            # Extract base filename without extension, e.g., "000000xxxxxx"
            # from "000000xxxxxx.jpg" or "path/to/000000xxxxxx.jpg"
            base_fn_no_ext = os.path.splitext(os.path.basename(img_anno_entry["file_name"]))[0]
            
            # Ensure it's 12 digits, zero-padded.
            # If base_fn_no_ext is already "000000xxxxxx", zfill(12) is harmless.
            standardized_base_fn = base_fn_no_ext.zfill(12)
            annotation_map_key = f"{standardized_base_fn}.png"
            panoptic_data_store["annotations_map"][annotation_map_key] = img_anno_entry
    
    if panoptic_data_store["annotations_map"]:
        print(f"[INFO] Panoptic annotations loaded from '{PANOPTIC_JSON_PATH}'. Found {len(panoptic_data_store['annotations_map'])} image entries.")
        if not panoptic_data_store["category_id_to_name_map"]: print(f"[WARNING] 'categories' list in '{PANOPTIC_JSON_PATH}' is empty or missing, so category_id_to_name_map is empty.")
        if not panoptic_data_store["thing_class_names"]: print(f"[WARNING] 'thing_class_names' list is empty (derived from 'categories').")
        if not panoptic_data_store["stuff_class_names"]: print(f"[WARNING] 'stuff_class_names' list is empty (derived from 'categories').")
    else:
        print(f"[ERROR] No image annotations found under 'data' key or 'file_name' missing in entries from '{PANOPTIC_JSON_PATH}'.")
except Exception as e:
    print(f"[ERROR] Failed to load or process panoptic annotations from '{PANOPTIC_JSON_PATH}': {e}")
    # Initialize to empty to prevent crashes, though functionality will be impaired
    panoptic_data_store.setdefault("annotations_map", {})
    panoptic_data_store.setdefault("category_id_to_name_map", {})
    panoptic_data_store.setdefault("thing_class_names", [])
    panoptic_data_store.setdefault("stuff_class_names", [])

@app.route('/save_edited_text', methods=['POST'])
def save_edited_text():
    """
    Route to save edited text to a file.
    Expects JSON data with 'video_id' and 'edited_text'.
    """
    try:
        data = request.json
        image_id_from_request = data['image_id'] # This is psg_id like "psg_286061"
        image_coco_path = data['image_coco_path'] # This is the original COCO path like "coco/images/val2017/000000286061.jpg"
        edited_text = data['edited_text']
        overwrite = data.get('overwrite', False) 

        # Validate image_id_from_request (used for saving the .txt file)
        if not image_id_from_request.startswith("psg_"):
            return jsonify({"error": "Invalid image_id format. Expected 'psg_...'.", "status": "failure"}), 400
        
        # Validate image_coco_path (basic check)
        if not image_coco_path or not isinstance(image_coco_path, str):
            return jsonify({"error": "Missing or invalid image_coco_path.", "status": "failure"}), 400

        # Compute base_filename_no_ext and annotation_map_key_for_labels
        # as in get_panoptic_categories, using image_coco_path.
        # Assumes base_filename_no_ext from COCO path is already 12-digit 0-padded.
        base_filename_no_ext = os.path.splitext(os.path.basename(image_coco_path))[0]
        annotation_map_key = f'{base_filename_no_ext}.png'
        
        # Get RLE segments and the ordered list of normalized labels for validation
        # base_filename_no_ext is used by the helper to find the mask PNG
        all_segment_details_with_rle, required_labels = _get_rle_segments_and_labels_for_image(
            annotation_map_key, 
            base_filename_no_ext
        )

        if not required_labels and not all_segment_details_with_rle: # If helper returned empty due to error
             print(f"[WARNING] No segment details or labels could be retrieved for {annotation_map_key}. Skipping caption validation and saving with empty segments.")
        
        # Update last displayed info, as these labels are now the ground truth for this save operation
        panoptic_data_store["last_displayed_image_key"] = annotation_map_key
        panoptic_data_store["last_displayed_image_normalized_labels"] = required_labels
        print(f"[INFO] For {image_id_from_request} (key: {annotation_map_key}), final required labels for caption: {required_labels}")

        missing_labels = []
        if required_labels: # Only validate if there are labels to check for
            for label in required_labels:
                # Case-insensitive check
                if label.lower() not in edited_text.lower():
                    missing_labels.append(label)
        
        if missing_labels:
            error_message = "Caption is missing required labels!"
            print(f"[VALIDATION_FAIL] For {annotation_map_key}: {error_message}")
            return jsonify({"error": error_message, "status": "failure", "missing_labels": missing_labels}), 400

        # Check for file existence if not overwriting
        json_filename = f'captions/{image_id_from_request}.json'
        if not overwrite and os.path.exists(json_filename):
            return jsonify({"message": "Annotation file already exists. Confirm to overwrite.", "status": "file_exists"}), 200

        
        # Validation passed or no labels to validate, or RLE generation failed (empty lists)
        output_json_data = {
            "caption": edited_text,
            "image_psg_id": image_id_from_request,
            "coco_image_path": image_coco_path,
            "annotation_map_key": annotation_map_key,
            "normalized_labels_for_caption": required_labels, # These are ordered according to RLEs
            "segments_rle_data": all_segment_details_with_rle # Contains RLEs and associated info
        }

        # Ensure the directory exists
        os.makedirs('captions', exist_ok=True)
        with open(json_filename, 'w') as f:
            json.dump(output_json_data, f, indent=4)
            
        print(f"[SUCCESS] JSON data saved to {json_filename}")
        return jsonify({"message": f"JSON file saved successfully to {json_filename}", "status": "success"})
    except Exception as e:
        print(f"[ERROR] in save_edited_text for {data.get('image_id', 'Unknown_image')}: {str(e)}")
        return jsonify({"error": f"An internal server error occurred: {str(e)}", "status": "failure"}), 500

@app.route('/get_panoptic_categories/<psg_id>/<path:image_coco_path>', methods=['GET'])
def get_panoptic_categories(psg_id, image_coco_path):
    """
    Returns the list of category names from the panoptic annotation JSON
    for a given image.
    psg_id: The PSG ID (e.g., "psg_286061")
    image_coco_path: The original COCO file path (e.g., "coco/images/val2017/000000286061.jpg")
    """
    try:
        # Extract the base numeric filename (e.g., "000000286061") from the COCO path
        base_filename_no_ext = os.path.splitext(os.path.basename(image_coco_path))[0]

        # Construct the key for annotations_map
        annotation_map_key = f'{base_filename_no_ext}.png'

        # Get RLE segments and the ordered list of normalized labels.
        # This ensures the label order is derived from the mask PNG, consistent with saving.
        all_segment_details, final_normalized_labels = _get_rle_segments_and_labels_for_image(
            annotation_map_key,
            base_filename_no_ext
        )

        normalized_category_info_list = []
        if not all_segment_details and not final_normalized_labels:
            # This case handles errors from _get_rle_segments_and_labels_for_image (e.g., no JSON annotation or mask file issue)
            # Specific warnings would have been printed by the helper function.
            print(f"[WARNING] get_panoptic_categories: No segment details or labels retrieved for {annotation_map_key}.")
            normalized_category_info_list = [{"id": -1, "name": f"No annotation data or mask found for {annotation_map_key}"}]
            # Update cache even in error case, with empty labels
            panoptic_data_store["last_displayed_image_key"] = annotation_map_key
            panoptic_data_store["last_displayed_image_normalized_labels"] = []
        else:
            # Construct the response list from the segment details
            normalized_category_info_list = [
                {"id": seg_detail["mask_segment_id"], "name": seg_detail["normalized_label"]}
                for seg_detail in all_segment_details
            ]
            
            # Store these as the "last displayed" for potential use by /save_edited_text
            panoptic_data_store["last_displayed_image_key"] = annotation_map_key
            panoptic_data_store["last_displayed_image_normalized_labels"] = final_normalized_labels
            print(f"[INFO] Stored labels for {annotation_map_key} as last displayed: {final_normalized_labels}")

        # Return the list of category names as JSON
        return jsonify({"categories": normalized_category_info_list})
    except Exception as e:
        print(f"[ERROR] Exception during category fetching for psg_id: {psg_id}, coco_path: {image_coco_path}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/check_annotation_exists/<image_psg_id>', methods=['GET'])
def check_annotation_exists(image_psg_id):
    """
    Checks if an annotation JSON file exists for the given image_psg_id.
    image_psg_id is expected to be like "psg_XXXXXX".
    """
    if not image_psg_id.startswith("psg_"):
        return jsonify({"exists": False, "error": "Invalid image_psg_id format."}), 400

    json_filename = f'captions/{image_psg_id}.json'
    if os.path.exists(json_filename):
        try:
            with open(json_filename, 'r') as f:
                data = json.load(f)
            caption = data.get("caption") # Returns None if 'caption' key doesn't exist
            if caption is not None:
                return jsonify({"exists": True, "filename": json_filename, "caption": caption}), 200
            else:
                return jsonify({"exists": True, "filename": json_filename, "caption": None, "message": "File exists but no caption found."}), 200
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to decode JSON from {json_filename}")
            return jsonify({"exists": True, "filename": json_filename, "caption": None, "error": "File exists but is not valid JSON."}), 200
        except Exception as e:
            print(f"[ERROR] Failed to read {json_filename}: {e}")
            return jsonify({"exists": True, "filename": json_filename, "caption": None, "error": f"Error reading file: {str(e)}"}), 200
    else:
        return jsonify({"exists": False}), 200


@app.route('/get_mask_overlay/<psg_id>/<path:image_coco_path>', methods=['GET'])
def get_mask_overlay_full(psg_id, image_coco_path):
    """ Route for fetching the full panoptic overlay. """
    return get_mask_overlay_impl(psg_id, image_coco_path, None)

@app.route('/get_mask_overlay/<psg_id>/<path:image_coco_path>/<int:selected_segment_id_val>', methods=['GET'])
def get_mask_overlay_selected(psg_id, image_coco_path, selected_segment_id_val):
    """ Route for fetching the panoptic overlay with a specific segment highlighted. """
    return get_mask_overlay_impl(psg_id, image_coco_path, str(selected_segment_id_val))

def get_mask_overlay_impl(psg_id, image_coco_path, selected_segment_id_str):
    """
    Common implementation for generating panoptic mask overlay.
    """
    print(f"[ROUTE_IMPL_ENTRY] psg_id: {psg_id}, image_coco_path: {image_coco_path}, selected_segment_id_str: {selected_segment_id_str}")
    try:
        # Extract the base numeric filename (e.g., "000000286061") from the COCO path
        base_filename_no_ext = os.path.splitext(os.path.basename(image_coco_path))[0]
        # The psg_id is now passed directly

        # Paths
        # image_path uses the psg_id (e.g., psg_286061.jpg)
        image_path = f'data/glamm_images/val_test/{psg_id}.jpg'
        # Assumes masks are stored based on the numeric ID
        mask_path = f'data/coco_pan_annotations/panoptic_train2017/{base_filename_no_ext}.png' # Assuming base_filename_no_ext is already 0-padded if needed by file system

        selected_segment_id = None
        if selected_segment_id_str is not None:
            try:
                selected_segment_id = int(selected_segment_id_str)
            except ValueError:
                print(f"[WARNING] Invalid selected_segment_id_str: {selected_segment_id_str}")
                return jsonify({"error": "Invalid selected_segment_id format"}), 400
       
        if not os.path.exists(image_path):
            print(f"Image not found at derived path: {image_path} (from psg_id: {psg_id})")
            return jsonify({"error": f"Image not found at derived path: {image_path} (from psg_id: {psg_id})"}), 404
        if not os.path.exists(mask_path):
            return jsonify({"error": f"Mask not found at: {mask_path} (derived from {image_coco_path})"}), 404

        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB") # Ensure mask is RGB
        mask = mask.resize(image.size, resample=Image.NEAREST)
        print(f"[DEBUG] Image: {image.size}, Mask: {mask.size}, Mask Path: {mask_path}")
        mask_np = np.array(mask)

        # Decode RGB to single ID value
        # New decoding: ID = R + G*256 + B*65536
        mask_np = mask_np[:, :, 0].astype(np.uint32) + \
                  (mask_np[:, :, 1].astype(np.uint32) << 8) + \
                  (mask_np[:, :, 2].astype(np.uint32) << 16)

        # Generate overlay
        overlay = np.array(image).copy()
        unique_ids = np.unique(mask_np)

        # Construct the key to find the image's annotation data
        # e.g., "panoptic_train2017/000000286061.png"
        annotation_map_key = f'{base_filename_no_ext}.png'
        image_annotation_entry = panoptic_data_store["annotations_map"].get(annotation_map_key)
        # print(image_annotation_entry)

        segment_id_to_category_name = {} # This maps JSON segment ID to its category name
        if image_annotation_entry:
            segments_info = image_annotation_entry.get("segments_info", [])
            category_id_map = panoptic_data_store["category_id_to_name_map"]

            for seg_info in segments_info:
                s_id = seg_info.get("id")
                cat_id_from_json = seg_info.get("category_id") # This is the category_id from the JSON

                if s_id is not None and cat_id_from_json is not None:
                    # Populate the mapping dictionary
                    category_name_from_json = category_id_map.get(cat_id_from_json, f"Unknown (ID:{cat_id_from_json})")
                    segment_id_to_category_name[s_id] = category_name_from_json
        else:
            print(f"[WARNING] No annotation entry found for key: {annotation_map_key}")
        
        # --- New Diagnostic Print Block ---
        # Print categories directly from the JSON's segments_info for the current image
        if image_annotation_entry:
            print(f"\n[DIAGNOSTIC] Categories listed in JSON for '{annotation_map_key}':")
            json_segments_info = image_annotation_entry.get("segments_info", [])
            category_id_map_diag = panoptic_data_store["category_id_to_name_map"]
            if not json_segments_info:
                print("  - No segments_info found in JSON for this image.")
            for idx, seg_info_item in enumerate(json_segments_info):
                json_s_id = seg_info_item.get("id")
                json_cat_id_val = seg_info_item.get("category_id") # Category ID value from JSON
                
                category_name_direct = "Unknown Category"
                if json_cat_id_val is not None:
                    category_name_direct = category_id_map_diag.get(json_cat_id_val, f"Unknown (ID:{json_cat_id_val})")
                # print(f"  - JSON Segment (idx {idx}): ID={json_s_id}, CategoryName='{category_name_direct}', COCO_CatID={json_cat_id_val}")
            print("[DIAGNOSTIC] End of JSON categories list.\n")
        # --- End of New Diagnostic Print Block ---

        # Print unique IDs and their category names
        print(f"[INFO] Found {len(unique_ids)} unique segment IDs in mask: {unique_ids.tolist()}")
        for obj_id in unique_ids:
            # This lookup uses the obj_id from the MASK against the s_id from the JSON used to build the map
            category_name = segment_id_to_category_name.get(obj_id, "Unknown (Mask ID not found in JSON segment IDs or no annotation entry)")
            # print(f"  - Mask Segment ID: {obj_id}, Category Name: {category_name}")

        cmap = colormaps.get_cmap('hsv')
        color_list = [(np.array(cmap(i / len(unique_ids))[:3]) * 255).astype(np.uint8)
                      for i in range(len(unique_ids))]
        
        # The `overlay` variable starts as a copy of the original image.
        # If selected_segment_id is provided, we only color that segment.
        # Otherwise, we color all segments.

        if selected_segment_id is not None and selected_segment_id != 0:
            if selected_segment_id in unique_ids:
                try:
                    # Find the index of the selected_segment_id in unique_ids to get its color
                    selected_idx_in_mask = np.where(unique_ids == selected_segment_id)[0][0]
                    color = color_list[selected_idx_in_mask]
                    
                    mask_binary = (mask_np == selected_segment_id)
                    # ---- ADD THIS DEBUG PRINT ----
                    if not np.any(mask_binary):
                        print(f"[DEBUG] For selected_segment_id {selected_segment_id}, mask_binary is all False. No pixels match this ID in the mask_np.")
                    # ---- END ADDED DEBUG PRINT ----
                    rows, cols = np.where(mask_binary)
                    if rows.size > 0: # Check if any pixels were actually selected
                        for c_channel in range(3):  # RGB channels
                            overlay[rows, cols, c_channel] = (
                                overlay[rows, cols, c_channel] * 0.4 + color[c_channel] * 0.6
                            ).astype(np.uint8)
                        print(f"[DEBUG] Applied color for selected_segment_id {selected_segment_id}. Pixels affected: {rows.size}")
                    else:
                        print(f"[DEBUG] selected_segment_id {selected_segment_id} was in unique_ids, but no pixels matched in mask_np after (mask_np == selected_segment_id) check.")
                except IndexError:
                    print(f"[WARNING] Could not find selected_segment_id {selected_segment_id} for specific coloring (IndexError during color lookup or np.where).")
            else:
                print(f"[INFO] selected_segment_id {selected_segment_id} (from JSON) not found in the actual mask's unique_ids.")
        else:
            print(f"[INFO] selected_segment_id {selected_segment_id} (from JSON) not found in the actual mask's unique_ids.")
            # Original logic: color all segments
            for idx, obj_id in enumerate(unique_ids):
                if obj_id == 0: # Skip background
                    continue
                mask_binary = mask_np == obj_id
                color = color_list[idx]
                try:
                    rows, cols = np.where(mask_binary)
                    for c_channel in range(3):  # RGB channels
                        overlay[rows, cols, c_channel] = (
                            overlay[rows, cols, c_channel] * 0.4 + color[c_channel] * 0.6
                        ).astype(np.uint8)
                except Exception as e:
                    print(f"[ERROR] Blending failed for obj_id {obj_id} with color {color}: {e}")

        # Convert to PIL and serve image
        overlay_img = Image.fromarray(overlay.astype(np.uint8))
        img_io = BytesIO()  # <-- this was missing before
        overlay_img.save(img_io, format="PNG")
        img_io.seek(0)      # move to the start

        print(f"[SUCCESS] Overlay generated for psg_id: {psg_id}, COCO path: {image_coco_path}")
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        print(f"[ERROR] Exception during overlay generation for psg_id: {psg_id}, coco_path: {image_coco_path}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=8500, debug=True)  # Set debug=False in production
