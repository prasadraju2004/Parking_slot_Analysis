import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf # Still needed for tf.lite and potentially preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import traceback
import io

app = Flask(__name__)
CORS(app)

interpreter = None # Changed from 'model' to 'interpreter'
input_details = None
output_details = None
all_spaces = {}

MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT = 224, 224

STANDARD_CSV_ORIGINAL_WIDTH = 2592
STANDARD_CSV_ORIGINAL_HEIGHT = 1944

# Load TFLite model
def load_tflite_model():
    global interpreter, input_details, output_details
    model_path = 'resnet50_dynamic_range.tflite' # <--- CHANGED MODEL PATH

    if not os.path.exists(model_path):
        print(f"ERROR: TFLite Model file not found at {os.path.abspath(model_path)}")
        return False

    try:
        print(f"Loading TFLite model from {os.path.abspath(model_path)}...")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors() # <--- IMPORTANT: Allocate tensors
        print(f"TFLite Model loaded successfully.")

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"Input details: {input_details}")
        print(f"Output details: {output_details}")

        # Test model with a dummy input
        # Expected input shape for ResNet50 is typically (1, 224, 224, 3)
        # Ensure dummy_input_shape matches input_details[0]['shape']
        dummy_input_shape_from_model = input_details[0]['shape']
        dummy_input = np.zeros(dummy_input_shape_from_model, dtype=input_details[0]['dtype'])

        # Preprocessing depends on what the TFLite model expects.
        # If the original Keras model's preprocessing is still part of the graph,
        # or if the TFLite model was converted from a model that expects raw pixels
        # and applies preprocessing internally, this might differ.
        # For a standard ResNet50 conversion, it likely still expects preprocessed input.
        # Assuming the TFLite model still expects input preprocessed by resnet_preprocess_input
        if dummy_input.shape[-1] == 3: # Check if it's an image-like input
            # ResNet preprocess expects float32 input and values typically in [0,255] or similar
            # before its specific scaling/mean subtraction.
            # Let's create a [0,255] dummy image first for preprocessing
            raw_dummy_image_for_preprocess = np.random.randint(0, 256, size=dummy_input_shape_from_model[1:], dtype=np.uint8)
            dummy_input_processed_for_model = resnet_preprocess_input(raw_dummy_image_for_preprocess.astype('float32')[np.newaxis, ...].copy())

            # Ensure the preprocessed shape matches the model's expected input shape
            if dummy_input_processed_for_model.shape != tuple(dummy_input_shape_from_model):
                 print(f"Warning: Shape mismatch after preprocessing. Preprocessed: {dummy_input_processed_for_model.shape}, Model expects: {dummy_input_shape_from_model}")
                 # Fallback to zeros if shapes don't match, but this indicates a potential issue
                 dummy_input_final = np.zeros(dummy_input_shape_from_model, dtype=input_details[0]['dtype'])
            else:
                dummy_input_final = dummy_input_processed_for_model.astype(input_details[0]['dtype'])
        else:
            dummy_input_final = dummy_input # For non-image inputs, or if preprocessing is different

        interpreter.set_tensor(input_details[0]['index'], dummy_input_final)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index']) # Get output
        print(f"TFLite Model successfully tested with a dummy input of shape {dummy_input_shape_from_model}")
        return True
    except Exception as e:
        print(f"ERROR loading TFLite model: {str(e)}")
        traceback.print_exc()
        return False

#Load CSV files (NO CHANGES HERE)
def load_camera_spaces():
    global all_spaces
    csv_dir = 'csv'
    if not os.path.exists(csv_dir):
        print(f"ERROR: CSV directory not found at {os.path.abspath(csv_dir)}")
        return False

    csv_files_found = False
    for i in range(1, 10):
        camera_csv_filename = f'camera{i}.csv'
        full_csv_path = os.path.join(csv_dir, camera_csv_filename)

        if os.path.exists(full_csv_path):
            try:
                print(f"Loading {camera_csv_filename} (original coordinates)...")
                df = pd.read_csv(full_csv_path)

                required_cols = ['SlotId', 'X', 'Y', 'W', 'H']
                if not all(col in df.columns for col in required_cols):
                    print(f"ERROR: {camera_csv_filename} is missing one or more required columns: {required_cols}. Skipping this file.")
                    continue

                all_spaces[f'camera{i}'] = df
                print(f"Loaded {len(df)} parking spaces for camera{i} (original coords).")
                csv_files_found = True
            except Exception as e:
                print(f"ERROR loading {camera_csv_filename}: {str(e)}")
                traceback.print_exc()

    if not csv_files_found:
        print(f"WARNING: No valid CSV files found in the '{os.path.abspath(csv_dir)}' directory.")
    return csv_files_found

# additional_car_check (NO CHANGES HERE)
def additional_car_check(space_img_original_patch):
    if space_img_original_patch is None or space_img_original_patch.size == 0:
        return False
    try:
        gray = cv2.cvtColor(space_img_original_patch, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]) if edges.size > 0 else 0
        texture_var = np.var(gray)

        is_car_like = (mean_intensity < 130 and edge_density > 0.07) or \
                      (edge_density > 0.15) or \
                      (texture_var > 700 and mean_intensity < 140)
        return is_car_like
    except Exception:
        return False


@app.route('/api/analyze', methods=['POST'])
def analyze():
    if interpreter is None or input_details is None or output_details is None: # <--- CHECK INTERPRETER
        return jsonify({'error': 'TFLite Model not loaded or failed to initialize'}), 500

    data = request.json
    if not data: return jsonify({'error': 'No data provided'}), 400

    image_data_b64 = data.get('imageData')
    camera_id = data.get('cameraId')

    if not image_data_b64 or not camera_id:
        return jsonify({'error': 'imageData (base64) and cameraId are required'}), 400

    if camera_id not in all_spaces:
        return jsonify({'error': f'No parking space layout data for {camera_id}'}), 404

    try:
        if ',' in image_data_b64:
            image_data_b64 = image_data_b64.split(',')[1]
        image_bytes = base64.b64decode(image_data_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_full_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_full_bgr is None:
            return jsonify({'error': 'Failed to decode image from base64 string'}), 400

        actual_img_h, actual_img_w = img_full_bgr.shape[:2]
        print(f"Received image for {camera_id}, Decoded shape: {actual_img_h}x{actual_img_w}")

        spaces_df = all_spaces[camera_id]
        results = {'totalSlots': 0, 'occupiedSlots': 0, 'availableSlots': 0, 'slotDetails': []}
        # Patches are now processed and predicted one by one for TFLite typically,
        # unless you construct a batch manually and ensure it matches input_details[0]['shape'][0]
        patch_info_for_results = []
        initial_total_slots_in_csv = len(spaces_df)

        scale_x_factor = 1.0
        scale_y_factor = 1.0
        if STANDARD_CSV_ORIGINAL_WIDTH > 0 and STANDARD_CSV_ORIGINAL_HEIGHT > 0:
            if actual_img_w == STANDARD_CSV_ORIGINAL_WIDTH and actual_img_h == STANDARD_CSV_ORIGINAL_HEIGHT:
                print(f"Image for {camera_id} matches standard original CSV dimensions ({STANDARD_CSV_ORIGINAL_WIDTH}x{STANDARD_CSV_ORIGINAL_HEIGHT}). No scaling of CSV coordinates needed.")
            else:
                scale_x_factor = actual_img_w / STANDARD_CSV_ORIGINAL_WIDTH
                scale_y_factor = actual_img_h / STANDARD_CSV_ORIGINAL_HEIGHT
                print(f"Scaling CSV coordinates for {camera_id}: From {STANDARD_CSV_ORIGINAL_WIDTH}x{STANDARD_CSV_ORIGINAL_HEIGHT} to image size {actual_img_w}x{actual_img_h}.")
                print(f"Calculated scaling factors: X={scale_x_factor:.4f}, Y={scale_y_factor:.4f}")
        else:
            print("Warning: STANDARD_CSV_ORIGINAL_WIDTH/HEIGHT not configured correctly. No scaling will be applied to CSV coordinates.")

        processed_patch_count = 0
        for _, row in spaces_df.iterrows():
            try:
                x_csv, y_csv = int(row['X']), int(row['Y'])
                w_csv, h_csv = int(row['W']), int(row['H'])
                slot_id = str(row['SlotId'])

                x, y = int(x_csv * scale_x_factor), int(y_csv * scale_y_factor)
                w, h = int(w_csv * scale_x_factor), int(h_csv * scale_y_factor)
                w = max(1, w)
                h = max(1, h)

            except ValueError:
                print(f"Warning: Non-integer coordinate or SlotId for {camera_id}. Skipping row: {row.to_dict()}")
                initial_total_slots_in_csv -=1
                continue

            if x >= actual_img_w or y >= actual_img_h or x + w > actual_img_w or y + h > actual_img_h or x < 0 or y < 0:
                print(f"Slot {slot_id} (scaled) for {camera_id} is out of bounds. (X:{x},Y:{y},W:{w},H:{h} vs Img:{actual_img_w}x{actual_img_h}). Skipping.")
                initial_total_slots_in_csv -=1
                continue

            space_img_original_patch = img_full_bgr[y:y+h, x:x+w]
            if space_img_original_patch.size == 0:
                print(f"Slot {slot_id} (scaled) for {camera_id} resulted in an empty patch. Skipping.")
                initial_total_slots_in_csv -=1
                continue

            space_img_model_input = cv2.resize(space_img_original_patch, (MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT))
            space_img_model_input_processed = resnet_preprocess_input(space_img_model_input.astype('float32').copy())

            # --- TFLite Prediction for a single patch ---
            # Ensure the input tensor is correctly shaped and typed for the TFLite model
            # Input details[0]['shape'] is e.g. [1, 224, 224, 3]
            # Input details[0]['dtype'] is e.g. <class 'numpy.float32'>
            input_tensor = np.expand_dims(space_img_model_input_processed, axis=0).astype(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            pred_value = interpreter.get_tensor(output_details[0]['index'])
            # --- End TFLite Prediction ---

            model_prediction_score = float(pred_value[0][0]) # Output is typically [[score]]

            model_says_occupied = model_prediction_score > 0.5
            heuristic_says_occupied = additional_car_check(space_img_original_patch)

            is_occupied = model_says_occupied or (heuristic_says_occupied and model_prediction_score > 0.3)
            confidence = model_prediction_score if is_occupied else (1 - model_prediction_score)

            if is_occupied: results['occupiedSlots'] += 1

            results['slotDetails'].append({
                'id': slot_id,
                'status': 'occupied' if is_occupied else 'available',
                'confidence': float(confidence),
                'coordinates': {'x': x, 'y': y, 'width': w, 'height': h},
                'model_raw_score': model_prediction_score
            })
            processed_patch_count += 1

        results['totalSlots'] = processed_patch_count

        if processed_patch_count == 0:
            if initial_total_slots_in_csv > 0 :
                 print(f"No valid patches were extracted for {camera_id} after processing its CSV.")
            elif initial_total_slots_in_csv == 0:
                 print(f"No slots were defined in the CSV for {camera_id} or all had invalid coordinate types.")
        results['availableSlots'] = results['totalSlots'] - results['occupiedSlots']
        print(f"Analysis for {camera_id} complete: {results['occupiedSlots']} occupied, {results['availableSlots']} available out of {results['totalSlots']} processable slots.")
        return jsonify(json_serialize(results))

    except base64.binascii.Error as b64e:
        print(f"Base64 decoding error for image related to {camera_id}: {str(b64e)}")
        traceback.print_exc()
        return jsonify({'error': 'Invalid base64 image data'}), 400
    except Exception as e:
        print(f"ERROR processing image request for {camera_id}: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/api/analyze_custom_image', methods=['POST'])
def analyze_custom_image():
    if interpreter is None or input_details is None or output_details is None: # <--- CHECK INTERPRETER
        return jsonify({'error': 'TFLite Model not loaded or failed to initialize'}), 500

    if 'imageFile' not in request.files or 'csvFile' not in request.files:
        return jsonify({'error': 'Both "imageFile" and "csvFile" are required in form-data'}), 400

    image_file = request.files['imageFile']
    csv_file = request.files['csvFile']

    if image_file.filename == '' or csv_file.filename == '':
        return jsonify({'error': 'No selected file for image or CSV'}), 400

    try:
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_full_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_full_bgr is None:
            return jsonify({'error': 'Failed to decode image file. Ensure it is a valid image format (JPG, PNG, etc.).'}), 400

        actual_img_h, actual_img_w = img_full_bgr.shape[:2]
        print(f"Received custom image: {image_file.filename}, Decoded shape: {actual_img_h}x{actual_img_w}")

        try:
            csv_content = io.StringIO(csv_file.read().decode('utf-8'))
            spaces_df = pd.read_csv(csv_content)
            required_cols = ['SlotId', 'X', 'Y', 'W', 'H']
            if not all(col in spaces_df.columns for col in required_cols):
                return jsonify({'error': f'CSV must contain columns: {", ".join(required_cols)}'}), 400
            print(f"Loaded custom CSV: {csv_file.filename}, with {len(spaces_df)} slots defined (raw coordinates).")
        except Exception as e:
            print(f"Error parsing CSV file: {str(e)}")
            traceback.print_exc()
            return jsonify({'error': f'Could not parse CSV file. Ensure it is valid UTF-8 CSV. Error: {str(e)}'}), 400

        results = {'totalSlots': 0, 'occupiedSlots': 0, 'availableSlots': 0, 'slotDetails': []}
        initial_total_slots_in_csv = len(spaces_df)
        processed_patch_count = 0

        scale_x_factor = 1.0
        scale_y_factor = 1.0
        if STANDARD_CSV_ORIGINAL_WIDTH > 0 and STANDARD_CSV_ORIGINAL_HEIGHT > 0:
            if actual_img_w == STANDARD_CSV_ORIGINAL_WIDTH and actual_img_h == STANDARD_CSV_ORIGINAL_HEIGHT:
                print(f"Custom image matches standard original CSV dimensions ({STANDARD_CSV_ORIGINAL_WIDTH}x{STANDARD_CSV_ORIGINAL_HEIGHT}). No scaling of CSV coordinates needed.")
            else:
                scale_x_factor = actual_img_w / STANDARD_CSV_ORIGINAL_WIDTH
                scale_y_factor = actual_img_h / STANDARD_CSV_ORIGINAL_HEIGHT
                print(f"Scaling custom CSV coordinates: From {STANDARD_CSV_ORIGINAL_WIDTH}x{STANDARD_CSV_ORIGINAL_HEIGHT} to image size {actual_img_w}x{actual_img_h}.")
                print(f"Calculated scaling factors: X={scale_x_factor:.4f}, Y={scale_y_factor:.4f}")
        else:
            print("Warning: STANDARD_CSV_ORIGINAL_WIDTH/HEIGHT not configured correctly. No scaling will be applied to custom CSV coordinates.")


        for _, row in spaces_df.iterrows():
            try:
                x_csv, y_csv = int(row['X']), int(row['Y'])
                w_csv, h_csv = int(row['W']), int(row['H'])
                slot_id = str(row['SlotId'])

                x, y = int(x_csv * scale_x_factor), int(y_csv * scale_y_factor)
                w, h = int(w_csv * scale_x_factor), int(h_csv * scale_y_factor)
                w = max(1, w)
                h = max(1, h)
            except ValueError:
                print(f"Warning: Non-integer coordinate or SlotId for a row in custom CSV. Skipping row: {row.to_dict()}")
                initial_total_slots_in_csv -=1
                continue

            if x >= actual_img_w or y >= actual_img_h or x + w > actual_img_w or y + h > actual_img_h or x < 0 or y < 0:
                print(f"Custom Slot {slot_id} (scaled) is out of bounds. (X:{x}, Y:{y}, W:{w}, H:{h} vs Img:{actual_img_w}x{actual_img_h}). Skipping.")
                initial_total_slots_in_csv -=1
                continue

            space_img_original_patch = img_full_bgr[y:y+h, x:x+w]
            if space_img_original_patch.size == 0:
                print(f"Custom Slot {slot_id} (scaled) resulted in an empty patch. Skipping.")
                initial_total_slots_in_csv -=1
                continue

            space_img_model_input = cv2.resize(space_img_original_patch, (MODEL_IMG_WIDTH, MODEL_IMG_HEIGHT))
            space_img_model_input_processed = resnet_preprocess_input(space_img_model_input.astype('float32').copy())

            # --- TFLite Prediction for a single patch ---
            input_tensor = np.expand_dims(space_img_model_input_processed, axis=0).astype(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            pred_value = interpreter.get_tensor(output_details[0]['index'])
            # --- End TFLite Prediction ---

            model_prediction_score = float(pred_value[0][0])
            model_says_occupied = model_prediction_score > 0.5
            heuristic_says_occupied = additional_car_check(space_img_original_patch)
            is_occupied = model_says_occupied or (heuristic_says_occupied and model_prediction_score > 0.3)
            confidence = model_prediction_score if is_occupied else (1 - model_prediction_score)
            if is_occupied: results['occupiedSlots'] += 1
            results['slotDetails'].append({
                'id': info['id'] if 'info' in locals() and info else slot_id, # Ensure info is defined or fallback
                'status': 'occupied' if is_occupied else 'available',
                'confidence': float(confidence),
                'coordinates': {'x': x, 'y': y, 'width': w, 'height': h},
                'model_raw_score': model_prediction_score
            })
            processed_patch_count += 1

        results['totalSlots'] = processed_patch_count
        if processed_patch_count == 0:
            if initial_total_slots_in_csv > 0 :
                 print("No valid patches were extracted from custom CSV after processing.")
            elif initial_total_slots_in_csv == 0:
                 print("No slots were defined in the custom CSV or all had invalid coordinate types.")

        results['availableSlots'] = results['totalSlots'] - results['occupiedSlots']
        print(f"Analysis for custom image '{image_file.filename}' complete: {results['occupiedSlots']} occupied, {results['availableSlots']} available out of {results['totalSlots']} processable slots.")
        return jsonify(json_serialize(results))

    except Exception as e:
        print(f"ERROR processing custom image request: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# json_serialize (NO CHANGES HERE)
def json_serialize(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {json_serialize(k): json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [json_serialize(i) for i in obj]
    else: return obj

@app.route('/api/health', methods=['GET'])
def health_check():
    model_name_str = "TFLite Model (Name N/A for interpreter)" # TFLite interpreter doesn't have a .name attribute
    return jsonify({
        'status': 'healthy',
        'model_loaded': interpreter is not None, # <--- CHECK INTERPRETER
        'model_name': model_name_str,
        'cameras_available': list(all_spaces.keys())
    })

if __name__ == '__main__':
    model_successfully_loaded = load_tflite_model() # <--- CALL NEW LOAD FUNCTION
    spaces_successfully_loaded = load_camera_spaces()

    if not model_successfully_loaded:
        print("CRITICAL WARNING: TFLite Model failed to load. Analysis endpoints will not function correctly.")
    if not spaces_successfully_loaded and not all_spaces:
        print("WARNING: No camera space layouts (CSVs) loaded. API endpoints will fail.")

    print("Starting Flask server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)