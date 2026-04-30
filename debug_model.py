"""
CNN Model Debugging & Validation Module
========================================
Diagnoses why a multi-class CNN may only predict a subset of classes.
Checks dataset loading, label encoding, model output layer, softmax
distribution, class balance, and generates diagnostic reports.

Usage:
    python debug_model.py
"""

import os
import sys
import numpy as np
import cv2
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_DIR  = r'C:\Users\nanda\leaf_disease\leaf_disease\train'
TEST_DIR   = r'C:\Users\nanda\leaf_disease\leaf_disease\test'
IMG_SIZE   = 50
LR         = 1e-3
MODEL_NAME = 'healthyvsunhealthynew-{}-{}.model'.format(LR, '2conv-basic')
OUTPUT_DIR = 'static/debug'

# Label mapping used by cnn.py (first character of filename -> class)
CHAR_TO_LABEL = {
    'h': ('Healthy',              [1,0,0,0,0,0]),
    'b': ('Bacterial Spot',       [0,1,0,0,0,0]),
    'v': ('Mosaic Virus',         [0,0,1,0,0,0]),
    's': ('Septoria Leaf Spot',   [0,0,0,1,0,0]),
    'm': ('Leaf Mold',            [0,0,0,0,1,0]),
    'x': ('Yellow Leaf Curl Virus',[0,0,0,0,0,1]),
}

INDEX_TO_CLASS = {
    0: 'Healthy',
    1: 'Bacterial Spot',
    2: 'Mosaic Virus',
    3: 'Septoria Leaf Spot',
    4: 'Leaf Mold',
    5: 'Yellow Leaf Curl Virus',
}

NUM_CLASSES = len(INDEX_TO_CLASS)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: Dataset Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataset(data_dir, label='TRAIN'):
    """Scan a directory and count images per class prefix."""
    print(f'\n{"="*60}')
    print(f'  STEP 1: Dataset Validation ({label})')
    print(f'{"="*60}')

    if not os.path.isdir(data_dir):
        print(f'  [ERROR] Directory not found: {data_dir}')
        return None

    files = [f for f in os.listdir(data_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f'  Total image files : {len(files)}')

    # Count by first character
    prefix_counts = Counter()
    unknown_prefixes = []
    for f in files:
        prefix = f[0].lower()
        if prefix in CHAR_TO_LABEL:
            prefix_counts[prefix] += 1
        else:
            unknown_prefixes.append(f)

    print(f'  Recognized classes: {len(prefix_counts)}')
    print(f'  Expected classes  : {NUM_CLASSES}')
    print()

    # Per-class breakdown
    print(f'  {"Prefix":<8} {"Class Name":<26} {"Count":>6}  {"Status"}')
    print(f'  {"-"*55}')
    for char in sorted(CHAR_TO_LABEL.keys()):
        name = CHAR_TO_LABEL[char][0]
        count = prefix_counts.get(char, 0)
        status = '[OK]' if count > 0 else '[MISSING]'
        print(f'  {char:<8} {name:<26} {count:>6}  {status}')

    # Imbalance check
    if prefix_counts:
        max_count = max(prefix_counts.values())
        min_count = min(prefix_counts.values())
        ratio = max_count / max(min_count, 1)
        print(f'\n  Class balance ratio (max/min): {ratio:.1f}x')
        if ratio > 5:
            print('  [WARNING] Severe class imbalance detected!')
            print('            The model may ignore minority classes.')
        elif ratio > 2:
            print('  [WARNING] Moderate class imbalance.')
        else:
            print('  [OK] Class distribution is reasonably balanced.')

    # Unknown files
    if unknown_prefixes:
        print(f'\n  [WARNING] {len(unknown_prefixes)} files with unrecognized prefix:')
        for f in unknown_prefixes[:5]:
            print(f'    - {f}')
        if len(unknown_prefixes) > 5:
            print(f'    ... and {len(unknown_prefixes) - 5} more')

    # Missing classes
    missing = [CHAR_TO_LABEL[c][0] for c in CHAR_TO_LABEL if c not in prefix_counts]
    if missing:
        print(f'\n  [ERROR] Missing classes in {label} data:')
        for m in missing:
            print(f'    - {m}')
    else:
        print(f'\n  [OK] All {NUM_CLASSES} classes present in {label} data.')

    return prefix_counts


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: Saved .npy Data Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_npy_data(npy_path, label='TRAIN'):
    """Load a saved .npy dataset and verify label distribution."""
    print(f'\n{"="*60}')
    print(f'  STEP 2: Saved Data Validation ({label} .npy)')
    print(f'{"="*60}')

    if not os.path.exists(npy_path):
        print(f'  [SKIP] File not found: {npy_path}')
        return None

    data = np.load(npy_path, allow_pickle=True)
    print(f'  File             : {npy_path}')
    print(f'  Total samples    : {len(data)}')

    if len(data) == 0:
        print('  [ERROR] Dataset is empty!')
        return None

    # Check shape of first sample
    sample_img, sample_label = data[0]
    print(f'  Image shape      : {sample_img.shape}')
    print(f'  Label shape      : {np.array(sample_label).shape}')
    print(f'  Label type       : {type(sample_label)}')

    # Count labels
    label_counts = Counter()
    for _, lbl in data:
        idx = int(np.argmax(lbl))
        label_counts[idx] += 1

    print(f'\n  {"Index":<8} {"Class Name":<26} {"Count":>6}')
    print(f'  {"-"*45}')
    for idx in range(NUM_CLASSES):
        name = INDEX_TO_CLASS.get(idx, 'Unknown')
        count = label_counts.get(idx, 0)
        print(f'  {idx:<8} {name:<26} {count:>6}')

    missing_in_data = [INDEX_TO_CLASS[i] for i in range(NUM_CLASSES)
                       if label_counts.get(i, 0) == 0]
    if missing_in_data:
        print(f'\n  [ERROR] Classes with ZERO samples in .npy:')
        for m in missing_in_data:
            print(f'    - {m}')
    else:
        print(f'\n  [OK] All {NUM_CLASSES} classes have samples.')

    return label_counts


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3: Model Architecture Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_model():
    """Build the model, load weights, and verify output layer."""
    print(f'\n{"="*60}')
    print(f'  STEP 3: Model Architecture Validation')
    print(f'{"="*60}')

    import tensorflow as tf
    tf.compat.v1.reset_default_graph()

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression

    # Rebuild exact same architecture as cnn.py / app.py
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

    # Print model summary
    print(f'  Output layer size  : {NUM_CLASSES} (softmax)')
    print(f'  Expected classes   : {NUM_CLASSES}')

    if NUM_CLASSES != len(INDEX_TO_CLASS):
        print('  [ERROR] Output layer size != number of classes!')
    else:
        print('  [OK] Output layer matches expected class count.')

    # Load weights
    meta_path = f'{MODEL_NAME}.meta'
    if os.path.exists(meta_path):
        model.load(MODEL_NAME)
        print(f'  [OK] Model weights loaded from: {MODEL_NAME}')
    else:
        print(f'  [WARNING] Model file not found: {meta_path}')
        print('            Cannot run prediction tests.')
        return None

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4: Prediction Analysis on Test Data
# ══════════════════════════════════════════════════════════════════════════════

def analyze_predictions(model):
    """Run predictions on test data and analyze class coverage."""
    print(f'\n{"="*60}')
    print(f'  STEP 4: Prediction Analysis on Test Data')
    print(f'{"="*60}')

    if not os.path.isdir(TEST_DIR):
        print(f'  [ERROR] Test directory not found: {TEST_DIR}')
        return None, None

    files = [f for f in os.listdir(TEST_DIR)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not files:
        print('  [ERROR] No images found in test directory.')
        return None, None

    print(f'  Test images found: {len(files)}')

    y_true = []
    y_pred = []
    softmax_outputs = []
    errors = 0

    for f in files:
        # Determine true label from filename prefix
        prefix = f[0].lower()
        if prefix not in CHAR_TO_LABEL:
            continue

        true_idx = list(CHAR_TO_LABEL.keys()).index(prefix)

        # Load and preprocess
        path = os.path.join(TEST_DIR, f)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            errors += 1
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_arr = np.array(img).reshape(IMG_SIZE, IMG_SIZE, 3)

        # Predict
        output = model.predict([img_arr])[0]
        pred_idx = int(np.argmax(output))

        y_true.append(true_idx)
        y_pred.append(pred_idx)
        softmax_outputs.append(output)

    print(f'  Predictions made : {len(y_pred)}')
    if errors:
        print(f'  Read errors      : {errors}')

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Unique predicted classes
    unique_pred = np.unique(y_pred)
    unique_true = np.unique(y_true)
    print(f'\n  Unique TRUE classes  : {sorted(unique_true.tolist())} ({len(unique_true)} classes)')
    print(f'  Unique PRED classes  : {sorted(unique_pred.tolist())} ({len(unique_pred)} classes)')

    # Missing predictions
    all_classes = set(range(NUM_CLASSES))
    missing_pred = all_classes - set(unique_pred)
    if missing_pred:
        print(f'\n  [WARNING] Classes NEVER predicted:')
        for idx in sorted(missing_pred):
            print(f'    - {idx}: {INDEX_TO_CLASS[idx]}')
        print('  Possible causes:')
        print('    1. Class imbalance in training data')
        print('    2. Insufficient training epochs')
        print('    3. Model overfitting to dominant classes')
        print('    4. Learning rate too high/low')
    else:
        print(f'\n  [OK] All {NUM_CLASSES} classes are being predicted.')

    # Per-class accuracy
    print(f'\n  {"Class":<26} {"Total":>6} {"Correct":>8} {"Accuracy":>10}')
    print(f'  {"-"*55}')
    for idx in range(NUM_CLASSES):
        mask = y_true == idx
        total = mask.sum()
        if total > 0:
            correct = (y_pred[mask] == idx).sum()
            acc = correct / total * 100
        else:
            correct = 0
            acc = 0.0
        name = INDEX_TO_CLASS[idx]
        print(f'  {name:<26} {total:>6} {correct:>8} {acc:>9.1f}%')

    overall_acc = (y_true == y_pred).mean() * 100
    print(f'\n  Overall accuracy: {overall_acc:.1f}%')

    # Softmax distribution analysis
    print(f'\n  --- Softmax Output Analysis ---')
    softmax_arr = np.array(softmax_outputs)
    print(f'  {"Class":<26} {"Mean Prob":>10} {"Max Prob":>10} {"Min Prob":>10}')
    print(f'  {"-"*60}')
    for idx in range(NUM_CLASSES):
        col = softmax_arr[:, idx]
        print(f'  {INDEX_TO_CLASS[idx]:<26} {col.mean():>10.4f} {col.max():>10.4f} {col.min():>10.4f}')

    # Check for "dead" outputs (class probability always near 0)
    dead_classes = []
    for idx in range(NUM_CLASSES):
        if softmax_arr[:, idx].max() < 0.1:
            dead_classes.append(idx)

    if dead_classes:
        print(f'\n  [ERROR] Dead softmax outputs (max prob < 0.1):')
        for idx in dead_classes:
            print(f'    - {idx}: {INDEX_TO_CLASS[idx]}')
        print('  These classes can NEVER be predicted!')
    else:
        print(f'\n  [OK] All softmax outputs are active.')

    return y_true, y_pred


# ══════════════════════════════════════════════════════════════════════════════
#  Step 5: Generate Diagnostic Charts
# ══════════════════════════════════════════════════════════════════════════════

def generate_confusion_matrix(y_true, y_pred):
    """Plot and save a confusion matrix."""
    print(f'\n{"="*60}')
    print(f'  STEP 5: Generating Diagnostic Charts')
    print(f'{"="*60}')

    n = NUM_CLASSES
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    class_names = [INDEX_TO_CLASS[i] for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('#1e1e2f')
    im = ax.imshow(cm, interpolation='nearest', cmap='magma')
    fig.colorbar(im, ax=ax, shrink=0.8)

    for i in range(n):
        for j in range(n):
            color = 'white' if cm[i, j] < cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=12, color=color, fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=12, color='#ccc')
    ax.set_ylabel('True', fontsize=12, color='#ccc')
    ax.set_title('Debug: Confusion Matrix', fontsize=15,
                 fontweight='bold', color='white', pad=14)
    ax.set_facecolor('#2b2b3d')
    ax.tick_params(colors='#aaa')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'debug_confusion_matrix.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved >> {path}')


def generate_class_balance_chart(train_counts, test_counts):
    """Bar chart comparing class counts in train vs test sets."""
    class_names = [CHAR_TO_LABEL[c][0] for c in sorted(CHAR_TO_LABEL.keys())]
    train_vals = [train_counts.get(c, 0) for c in sorted(CHAR_TO_LABEL.keys())]
    test_vals  = [test_counts.get(c, 0) for c in sorted(CHAR_TO_LABEL.keys())]

    x = np.arange(len(class_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#2b2b3d')

    bars1 = ax.bar(x - w/2, train_vals, w, label='Train', color='#00e5ff', edgecolor='#222')
    bars2 = ax.bar(x + w/2, test_vals,  w, label='Test',  color='#ff6e40', edgecolor='#222')

    for bar, val in zip(bars1, train_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontsize=9, color='white')
    for bar, val in zip(bars2, test_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontsize=9, color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha='right', fontsize=10)
    ax.set_title('Debug: Train vs Test Class Distribution',
                 fontsize=15, fontweight='bold', color='white', pad=14)
    ax.set_ylabel('Number of Images', fontsize=12, color='#ccc')
    ax.tick_params(colors='#aaa')
    ax.legend(facecolor='#2b2b3d', edgecolor='#555', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#555')
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'debug_class_balance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved >> {path}')


def generate_classification_report(y_true, y_pred):
    """Print a text-based classification report."""
    print(f'\n{"="*60}')
    print(f'  Classification Report')
    print(f'{"="*60}')

    print(f'\n  {"Class":<26} {"Precision":>10} {"Recall":>10} {"F1-Score":>10} {"Support":>8}')
    print(f'  {"-"*68}')

    for idx in range(NUM_CLASSES):
        tp = ((y_true == idx) & (y_pred == idx)).sum()
        fp = ((y_true != idx) & (y_pred == idx)).sum()
        fn = ((y_true == idx) & (y_pred != idx)).sum()

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        support   = (y_true == idx).sum()

        name = INDEX_TO_CLASS[idx]
        print(f'  {name:<26} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>8}')

    overall_acc = (y_true == y_pred).mean()
    print(f'\n  Overall Accuracy: {overall_acc:.3f} ({overall_acc*100:.1f}%)')


# ══════════════════════════════════════════════════════════════════════════════
#  Step 6: Summary & Recommendations
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(train_counts, test_counts, y_true, y_pred):
    """Print final diagnosis and recommendations."""
    print(f'\n{"="*60}')
    print(f'  STEP 6: Diagnosis Summary')
    print(f'{"="*60}')

    issues = []

    # Check class balance
    if train_counts:
        vals = list(train_counts.values())
        if max(vals) / max(min(vals), 1) > 3:
            issues.append('Severe class imbalance in training data')

    # Check missing predictions
    if y_pred is not None:
        pred_classes = set(np.unique(y_pred))
        expected = set(range(NUM_CLASSES))
        missing = expected - pred_classes
        if missing:
            names = [INDEX_TO_CLASS[i] for i in missing]
            issues.append(f'Model never predicts: {", ".join(names)}')

    # Check low accuracy classes
    if y_true is not None and y_pred is not None:
        for idx in range(NUM_CLASSES):
            mask = y_true == idx
            if mask.sum() > 0:
                acc = (y_pred[mask] == idx).mean()
                if acc < 0.3:
                    issues.append(f'{INDEX_TO_CLASS[idx]} has very low accuracy ({acc*100:.0f}%)')

    if issues:
        print('\n  [!] Issues Found:')
        for i, issue in enumerate(issues, 1):
            print(f'      {i}. {issue}')
        print('\n  Recommendations:')
        print('    - Balance dataset using oversampling/augmentation')
        print('    - Increase training epochs (currently 10)')
        print('    - Use data augmentation (rotation, flip, brightness)')
        print('    - Try a lower learning rate (e.g. 0.0005)')
        print('    - Add more training images for underrepresented classes')
    else:
        print('\n  [OK] No critical issues found.')
        print('  The model appears to be correctly configured for all 6 classes.')

    print(f'\n  Debug charts saved to: {OUTPUT_DIR}/')
    print(f'{"="*60}')


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('\n' + '#'*60)
    print('#  CNN Model Debugger - Plant Disease Detection')
    print('#' + '='*58 + '#')

    # Step 1: Dataset validation
    train_counts = validate_dataset(TRAIN_DIR, 'TRAIN')
    test_counts  = validate_dataset(TEST_DIR, 'TEST')

    # Step 2: Saved .npy validation
    validate_npy_data('train_data.npy', 'TRAIN')
    validate_npy_data('test_data.npy', 'TEST')

    # Step 3 & 4: Model + Predictions
    y_true, y_pred = None, None
    try:
        model = validate_model()
        if model is not None:
            y_true, y_pred = analyze_predictions(model)
    except Exception as e:
        print(f'\n  [ERROR] Model loading failed: {e}')
        print('  Skipping prediction analysis.')

    # Step 5: Charts
    if train_counts and test_counts:
        generate_class_balance_chart(train_counts, test_counts)

    if y_true is not None and y_pred is not None:
        generate_confusion_matrix(y_true, y_pred)
        generate_classification_report(y_true, y_pred)

    # Step 6: Summary
    print_summary(train_counts, test_counts, y_true, y_pred)
