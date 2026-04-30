"""
Data Visualization Module for Plant Disease Detection CNN Model
================================================================
Standalone module for generating training metrics and prediction
analysis charts. Compatible with any CNN model that outputs
training history and prediction arrays.

Usage:
    from visualize import Visualizer
    viz = Visualizer(output_dir='static/charts')
    viz.plot_accuracy(history)
    viz.plot_loss(history)
    viz.plot_confusion_matrix(y_true, y_pred)
    viz.plot_prediction_distribution(predictions)
    viz.generate_all(history, y_true, y_pred, predictions)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ── Default disease classes matching the CNN label map ────────────────────────

DEFAULT_CLASSES = [
    'Healthy',
    'Bacterial Spot',
    'Mosaic Virus',
    'Septoria Leaf Spot',
    'Leaf Mold',
    'Yellow Leaf Curl Virus',
]

# Color palette – vibrant yet accessible
PALETTE = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#1abc9c']


# ── Helper: apply a consistent dark style to every figure ─────────────────────

def _style_figure(fig, ax, title, xlabel, ylabel):
    """Apply uniform dark theme styling to a matplotlib figure."""
    fig.patch.set_facecolor('#1e1e2f')
    ax.set_facecolor('#2b2b3d')
    ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=14)
    ax.set_xlabel(xlabel, fontsize=12, color='#cccccc')
    ax.set_ylabel(ylabel, fontsize=12, color='#cccccc')
    ax.tick_params(colors='#aaaaaa', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#444466')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, color='#555577')


# ══════════════════════════════════════════════════════════════════════════════
#  Visualizer class
# ══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """Reusable chart generator for CNN training & prediction analysis."""

    def __init__(self, output_dir='static/charts', class_names=None, dpi=150):
        self.output_dir = output_dir
        self.class_names = class_names or DEFAULT_CLASSES
        self.dpi = dpi
        os.makedirs(self.output_dir, exist_ok=True)

    # ── 1. Training vs Validation Accuracy ────────────────────────────────

    def plot_accuracy(self, history, filename='accuracy.png'):
        """
        Plot training & validation accuracy over epochs.

        Parameters
        ----------
        history : dict
            Must contain keys 'accuracy' (or 'acc') and
            'val_accuracy' (or 'val_acc'), each a list of floats.
        filename : str
            Output file name (saved inside output_dir).

        Returns
        -------
        str  – absolute path to the saved image.
        """
        acc = history.get('accuracy') or history.get('acc', [])
        val_acc = history.get('val_accuracy') or history.get('val_acc', [])
        epochs = range(1, len(acc) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, acc, 'o-', color='#00e5ff', linewidth=2,
                markersize=5, label='Training Accuracy')
        ax.plot(epochs, val_acc, 's--', color='#ff6e40', linewidth=2,
                markersize=5, label='Validation Accuracy')

        _style_figure(fig, ax,
                      'Training vs Validation Accuracy',
                      'Epoch', 'Accuracy')
        ax.legend(fontsize=11, facecolor='#2b2b3d', edgecolor='#555577',
                  labelcolor='white', loc='lower right')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_ylim(0, 1.05)

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'[Visualizer] Saved >> {path}')
        return os.path.abspath(path)

    # ── 2. Training vs Validation Loss ────────────────────────────────────

    def plot_loss(self, history, filename='loss.png'):
        """
        Plot training & validation loss over epochs.

        Parameters
        ----------
        history : dict
            Must contain keys 'loss' and 'val_loss'.
        """
        loss = history.get('loss', [])
        val_loss = history.get('val_loss', [])
        epochs = range(1, len(loss) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, loss, 'o-', color='#69f0ae', linewidth=2,
                markersize=5, label='Training Loss')
        ax.plot(epochs, val_loss, 's--', color='#ff5252', linewidth=2,
                markersize=5, label='Validation Loss')

        _style_figure(fig, ax,
                      'Training vs Validation Loss',
                      'Epoch', 'Loss')
        ax.legend(fontsize=11, facecolor='#2b2b3d', edgecolor='#555577',
                  labelcolor='white', loc='upper right')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'[Visualizer] Saved >> {path}')
        return os.path.abspath(path)

    # ── 3. Confusion Matrix ───────────────────────────────────────────────

    def plot_confusion_matrix(self, y_true, y_pred, filename='confusion_matrix.png'):
        """
        Plot a confusion matrix heatmap.

        Parameters
        ----------
        y_true : array-like   – ground-truth class indices (ints).
        y_pred : array-like   – predicted class indices (ints).
        """
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        n = len(self.class_names)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t][p] += 1

        fig, ax = plt.subplots(figsize=(9, 8))
        fig.patch.set_facecolor('#1e1e2f')

        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='magma',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names, ax=ax,
                        linewidths=0.5, linecolor='#333355',
                        cbar_kws={'shrink': 0.8})
        else:
            im = ax.imshow(cm, interpolation='nearest', cmap='magma')
            fig.colorbar(im, ax=ax, shrink=0.8)
            for i in range(n):
                for j in range(n):
                    color = 'white' if cm[i, j] < cm.max() / 2 else 'black'
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=11, color=color, fontweight='bold')
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax.set_yticklabels(self.class_names)

        ax.set_facecolor('#2b2b3d')
        ax.set_title('Confusion Matrix – Disease Classification',
                      fontsize=15, fontweight='bold', color='white', pad=14)
        ax.set_xlabel('Predicted Label', fontsize=12, color='#cccccc')
        ax.set_ylabel('True Label', fontsize=12, color='#cccccc')
        ax.tick_params(colors='#aaaaaa', labelsize=9)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'[Visualizer] Saved >> {path}')
        return os.path.abspath(path)

    # ── 4. Prediction Distribution Bar Chart ──────────────────────────────

    def plot_prediction_distribution(self, predictions, filename='prediction_distribution.png'):
        """
        Bar chart showing how many samples were predicted per class.

        Parameters
        ----------
        predictions : array-like
            List / array of predicted class indices (ints).
        """
        predictions = np.asarray(predictions)
        counts = np.zeros(len(self.class_names), dtype=int)
        for p in predictions:
            counts[p] += 1

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(self.class_names, counts, color=PALETTE[:len(self.class_names)],
                      edgecolor='#222244', linewidth=1.2, width=0.6)

        # Value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color='white')

        _style_figure(fig, ax,
                      'Prediction Distribution Across Disease Classes',
                      'Disease Class', 'Number of Predictions')
        ax.set_xticklabels(self.class_names, rotation=30, ha='right', fontsize=10)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f'[Visualizer] Saved >> {path}')
        return os.path.abspath(path)

    # ── Convenience: generate all four charts at once ─────────────────────

    def generate_all(self, history, y_true, y_pred, predictions=None):
        """
        Generate all four visualizations in one call.

        Parameters
        ----------
        history     : dict        – training history with acc / loss keys.
        y_true      : array-like  – ground-truth class indices.
        y_pred      : array-like  – predicted class indices.
        predictions : array-like  – (optional) if None, y_pred is reused.

        Returns
        -------
        dict  – mapping chart name → saved file path.
        """
        if predictions is None:
            predictions = y_pred

        paths = {
            'accuracy':               self.plot_accuracy(history),
            'loss':                   self.plot_loss(history),
            'confusion_matrix':       self.plot_confusion_matrix(y_true, y_pred),
            'prediction_distribution': self.plot_prediction_distribution(predictions),
        }
        print(f'\n[Visualizer] All charts saved to: {self.output_dir}/')
        return paths


# ══════════════════════════════════════════════════════════════════════════════
#  Demo / standalone execution
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Simulated training history (10 epochs)
    np.random.seed(42)
    demo_history = {
        'accuracy':     [0.55, 0.62, 0.70, 0.76, 0.81, 0.85, 0.88, 0.90, 0.92, 0.94],
        'val_accuracy': [0.50, 0.58, 0.64, 0.70, 0.74, 0.77, 0.80, 0.82, 0.83, 0.84],
        'loss':         [1.60, 1.30, 1.05, 0.85, 0.70, 0.58, 0.48, 0.40, 0.34, 0.28],
        'val_loss':     [1.70, 1.45, 1.25, 1.08, 0.95, 0.86, 0.80, 0.76, 0.74, 0.73],
    }

    # Simulated predictions (200 samples across 6 classes)
    demo_y_true = np.random.randint(0, 6, size=200)
    demo_y_pred = demo_y_true.copy()
    # Introduce ~15 % noise to make the confusion matrix realistic
    noise_idx = np.random.choice(200, size=30, replace=False)
    demo_y_pred[noise_idx] = np.random.randint(0, 6, size=30)

    viz = Visualizer(output_dir='static/charts')
    viz.generate_all(demo_history, demo_y_true, demo_y_pred)

    print('\n[OK] Demo complete - open static/charts/ to view the PNGs.')
