import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Custom F1 metric
class F1Score(keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
        
def plot_metrics(history):
    # Plot Accuracy
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot AUC
    plt.plot(history.history['auc'], label='train AUC')
    plt.plot(history.history['val_auc'], label='val AUC')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()

    # Plot F1
    plt.plot(history.history['f1'], label='train F1')
    plt.plot(history.history['val_f1'], label='val F1')
    plt.title('Model F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
plt.show()