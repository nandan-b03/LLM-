"""
CNN Prediction Debugger & Correction Module
=============================================
Diagnoses and fixes issues where the model predicts the same
disease for every image. Checks softmax bias, input preprocessing,
class mapping, and supplement routing.

Usage:
    python predict_debug.py                          # run full diagnostics
    python predict_debug.py path/to/leaf_image.jpg   # debug single image
"""

import os
import sys
import cv2
import numpy as np
from collections import Counter

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_DIR  = r'C:\Users\nanda\leaf_disease\leaf_disease\train'
TEST_DIR   = r'C:\Users\nanda\leaf_disease\leaf_disease\test'
IMG_SIZE   = 50
LR         = 1e-3
MODEL_NAME = 'healthyvsunhealthynew-{}-{}.model'.format(LR, '2conv-basic')
NUM_CLASSES = 6

# ── Label & Supplement Mapping (dynamic, not hardcoded) ───────────────────────

LABEL_MAP = {
    0: 'Healthy',
    1: 'Bacterial',
    2: 'Mosaic_virus',
    3: 'Spectoria',
    4: 'Leafmod',
    5: 'curl_virus',
}

DISEASE_SUPPLEMENTS = {
    'Healthy': {
        'display':    'Healthy Leaf',
        'supplement': 'Plant Fertilizer',
        'action':     'Maintain good watering and sunlight.',
        'buy_url':    'https://www.amazon.in/s?k=plant+fertilizer',
    },
    'Bacterial': {
        'display':    'Bacterial Spot',
        'supplement': 'Copper Fungicide',
        'action':     'Apply copper-based fungicide. Avoid overhead watering.',
        'buy_url':    'https://www.amazon.in/s?k=copper+fungicide',
    },
    'Mosaic_virus': {
        'display':    'Mosaic Virus',
        'supplement': 'Neem Oil',
        'action':     'Control aphids with Neem Oil. Remove infected plants.',
        'buy_url':    'https://www.amazon.in/s?k=neem+oil',
    },
    'Spectoria': {
        'display':    'Septoria Leaf Spot',
        'supplement': 'Daconil Fungicide',
        'action':     'Apply Daconil. Mulch around plants.',
        'buy_url':    'https://www.amazon.in/s?k=daconil',
    },
    'Leafmod': {
        'display':    'Leaf Mold',
        'supplement': 'Chlorothalonil',
        'action':     'Improve air circulation. Apply Chlorothalonil.',
        'buy_url':    'https://www.amazon.in/s?k=chlorothalonil',
    },
    'curl_virus': {
        'display':    'Yellow Leaf Curl Virus',
        'supplement': 'Yellow Sticky Traps + Neem Oil',
        'action':     'Use sticky traps for whiteflies. Apply Neem oil.',
        'buy_url':    'https://www.amazon.in/s?k=yellow+sticky+traps',
    },
}

# Character prefix used in filenames -> class index
CHAR_TO_INDEX = {'h': 0, 'b': 1, 'v': 2, 's': 3, 'm': 4, 'x': 5}


# ══════════════════════════════════════════════════════════════════════════════
#  Build & Load Model
# ══════════════════════════════════════════════════════════════════════════════

def build_and_load_model():
    """Rebuild the CNN architecture and load saved weights."""
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
    convnet = conv_2d(convnet, 32,  3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
    convnet = conv_2d(convnet, 64,  3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
    convnet = conv_2d(convnet, 32,  3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
    convnet = conv_2d(convnet, 64,  3, activation='relu')
    convnet = max_pool_2d(convnet, 3)
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, NUM_CLASSES, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists(f'{MODEL_NAME}.meta'):
        model.load(MODEL_NAME)
        print('[OK] Model loaded successfully.')
    else:
        print(f'[ERROR] Model file not found: {MODEL_NAME}.meta')
        return None

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Image Preprocessing (matches training pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image(image_path):
    """Load and preprocess an image exactly as the training pipeline does."""
    print(f'\n--- Input Preprocessing Check ---')
    print(f'  Image path: {image_path}')

    if not os.path.exists(image_path):
        print(f'  [ERROR] File not found!')
        return None

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f'  [ERROR] cv2 could not read the image.')
        return None

    print(f'  Original shape : {img.shape}')
    print(f'  Dtype          : {img.dtype}')
    print(f'  Pixel range    : [{img.min()}, {img.max()}]')

    # Resize to match training size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    print(f'  Resized shape  : {img.shape}')

    # Check if normalization is needed
    # Training pipeline uses raw uint8 (0-255), so we keep it as-is
    img_array = np.array(img).reshape(IMG_SIZE, IMG_SIZE, 3)
    print(f'  Final shape    : {img_array.shape}')
    print(f'  Final range    : [{img_array.min()}, {img_array.max()}]')

    if img_array.max() <= 1.0:
        print('  [WARNING] Image appears normalized (0-1) but training uses 0-255!')
    else:
        print('  [OK] Pixel range matches training format (0-255).')

    return img_array


# ══════════════════════════════════════════════════════════════════════════════
#  Prediction with Full Debugging
# ══════════════════════════════════════════════════════════════════════════════

def predict_with_debug(model, image_path):
    """
    Predict disease from a leaf image with full diagnostic output.
    Returns top-3 predictions with probabilities and supplement info.
    """
    img_array = preprocess_image(image_path)
    if img_array is None:
        return None

    # Get raw softmax output
    softmax_output = model.predict([img_array])[0]
    softmax_output = np.array(softmax_output)

    print(f'\n--- Softmax Output (all {NUM_CLASSES} classes) ---')
    for idx in range(NUM_CLASSES):
        label = LABEL_MAP[idx]
        prob = softmax_output[idx]
        bar = '#' * int(prob * 40)
        print(f'  [{idx}] {label:<20} {prob:>7.4f}  {bar}')

    # Check for bias
    max_prob = softmax_output.max()
    min_prob = softmax_output.min()
    pred_idx = int(np.argmax(softmax_output))

    print(f'\n--- Bias Check ---')
    if max_prob - min_prob < 0.05:
        print('  [WARNING] All probabilities are nearly equal!')
        print('  The model may not have learned meaningful features.')
        print('  Possible causes:')
        print('    - Model not trained (weights are random)')
        print('    - Wrong model file loaded')
        print('    - Input preprocessing mismatch')
    elif max_prob > 0.99:
        print(f'  [INFO] Very high confidence ({max_prob:.4f}) - model is decisive.')
    else:
        print(f'  [OK] Prediction spread looks normal (max={max_prob:.4f}, min={min_prob:.4f}).')

    # Top-3 predictions
    top3_indices = np.argsort(softmax_output)[::-1][:3]
    print(f'\n--- Top-3 Predictions ---')
    for rank, idx in enumerate(top3_indices, 1):
        label = LABEL_MAP[idx]
        prob = softmax_output[idx]
        info = DISEASE_SUPPLEMENTS.get(label, {})
        display = info.get('display', label)
        supplement = info.get('supplement', 'N/A')
        print(f'  #{rank}: {display:<26} ({prob*100:>5.1f}%)  -> Supplement: {supplement}')

    # Final prediction with dynamic supplement mapping
    pred_label = LABEL_MAP[pred_idx]
    confidence = softmax_output[pred_idx]
    info = DISEASE_SUPPLEMENTS.get(pred_label, {})

    result = {
        'class_index':  pred_idx,
        'class_label':  pred_label,
        'display_name': info.get('display', pred_label),
        'confidence':   float(confidence),
        'supplement':   info.get('supplement', 'Unknown'),
        'action':       info.get('action', ''),
        'buy_url':      info.get('buy_url', ''),
        'softmax':      softmax_output.tolist(),
    }

    print(f'\n--- Final Result ---')
    print(f'  Disease    : {result["display_name"]}')
    print(f'  Confidence : {result["confidence"]*100:.1f}%')
    print(f'  Supplement : {result["supplement"]}')
    print(f'  Action     : {result["action"]}')

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Batch Diagnosis: Check if model always predicts the same class
# ══════════════════════════════════════════════════════════════════════════════

def diagnose_batch_predictions(model, data_dir=None, max_images=50):
    """Test multiple images to see if predictions vary."""
    print(f'\n{"="*60}')
    print(f'  Batch Prediction Diagnosis')
    print(f'{"="*60}')

    if data_dir is None:
        data_dir = TEST_DIR

    if not os.path.isdir(data_dir):
        print(f'  [ERROR] Directory not found: {data_dir}')
        return

    files = [f for f in os.listdir(data_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_images]

    print(f'  Testing {len(files)} images from: {data_dir}')

    predictions = []
    true_labels = []

    for f in files:
        prefix = f[0].lower()
        if prefix not in CHAR_TO_INDEX:
            continue

        path = os.path.join(data_dir, f)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_arr = np.array(img).reshape(IMG_SIZE, IMG_SIZE, 3)

        output = model.predict([img_arr])[0]
        pred_idx = int(np.argmax(output))
        true_idx = CHAR_TO_INDEX[prefix]

        predictions.append(pred_idx)
        true_labels.append(true_idx)

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # Count unique predictions
    pred_counts = Counter(predictions)
    unique_preds = len(pred_counts)

    print(f'\n  Total predictions: {len(predictions)}')
    print(f'  Unique classes predicted: {unique_preds} / {NUM_CLASSES}')

    print(f'\n  Prediction Distribution:')
    print(f'  {"Class":<26} {"Predicted":>10} {"Expected":>10}')
    print(f'  {"-"*50}')
    true_counts = Counter(true_labels)
    for idx in range(NUM_CLASSES):
        name = LABEL_MAP[idx]
        info = DISEASE_SUPPLEMENTS.get(name, {})
        display = info.get('display', name)
        pred_c = pred_counts.get(idx, 0)
        true_c = true_counts.get(idx, 0)
        print(f'  {display:<26} {pred_c:>10} {true_c:>10}')

    # Diagnose
    if unique_preds == 1:
        sole_class = LABEL_MAP[predictions[0]]
        print(f'\n  [CRITICAL] Model ALWAYS predicts: {sole_class}')
        print(f'  This means the model is stuck on one class.')
        print(f'  Fixes:')
        print(f'    1. Re-train with balanced data')
        print(f'    2. Increase epochs (current: 10)')
        print(f'    3. Check if train_data.npy has correct labels')
        print(f'    4. Reduce learning rate (try 0.0005)')
    elif unique_preds < NUM_CLASSES:
        missing = set(range(NUM_CLASSES)) - set(pred_counts.keys())
        missing_names = [LABEL_MAP[i] for i in missing]
        print(f'\n  [WARNING] Some classes never predicted:')
        for m in missing_names:
            print(f'    - {m}')
    else:
        accuracy = (predictions == true_labels).mean() * 100
        print(f'\n  [OK] All classes are being predicted.')
        print(f'  Accuracy: {accuracy:.1f}%')

    # Check supplement mapping
    print(f'\n--- Supplement Mapping Verification ---')
    for idx in range(NUM_CLASSES):
        label = LABEL_MAP[idx]
        info = DISEASE_SUPPLEMENTS.get(label)
        if info:
            print(f'  [{idx}] {label:<16} -> {info["supplement"]}')
        else:
            print(f'  [{idx}] {label:<16} -> [MISSING MAPPING!]')

    return predictions, true_labels


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset Class Balance Check
# ══════════════════════════════════════════════════════════════════════════════

def check_dataset_balance():
    """Verify training data has proper class distribution."""
    print(f'\n{"="*60}')
    print(f'  Dataset Balance Check')
    print(f'{"="*60}')

    for label, dir_path in [('TRAIN', TRAIN_DIR), ('TEST', TEST_DIR)]:
        if not os.path.isdir(dir_path):
            print(f'  [{label}] Directory not found: {dir_path}')
            continue

        files = os.listdir(dir_path)
        counts = Counter()
        for f in files:
            prefix = f[0].lower()
            if prefix in CHAR_TO_INDEX:
                counts[prefix] += 1

        print(f'\n  {label} Data ({sum(counts.values())} images):')
        for char in sorted(CHAR_TO_INDEX.keys()):
            idx = CHAR_TO_INDEX[char]
            name = LABEL_MAP[idx]
            count = counts.get(char, 0)
            bar = '#' * (count // 20)
            print(f'    {char} -> {name:<20} {count:>5}  {bar}')

        if counts:
            ratio = max(counts.values()) / max(min(counts.values()), 1)
            if ratio > 3:
                print(f'  [WARNING] Imbalance ratio: {ratio:.1f}x')
            else:
                print(f'  [OK] Balanced (ratio: {ratio:.1f}x)')


# ══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('\n' + '#'*60)
    print('#  CNN Prediction Debugger - Plant Disease Detection')
    print('#' + '='*58 + '#')

    # Step 1: Dataset check (no model needed)
    check_dataset_balance()

    # Step 2: Load model
    model = build_and_load_model()
    if model is None:
        print('\nCannot proceed without model. Exiting.')
        sys.exit(1)

    # Step 3: Single image or batch
    if len(sys.argv) > 1:
        # Single image debug
        image_path = sys.argv[1]
        predict_with_debug(model, image_path)
    else:
        # Batch diagnosis
        diagnose_batch_predictions(model)

    print('\n' + '#'*60)
    print('#  Debug Complete')
    print('#' + '='*58 + '#')
