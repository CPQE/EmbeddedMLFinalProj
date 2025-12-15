import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score

import tensorflow as tf
from tensorflow import keras

# Custom F1 metric
@tf.keras.utils.register_keras_serializable(package="metrics_tracking")
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
        
def plot_metrics(history, model_name="best_model_road_128_f1Fixed.keras", save_plots=False):
    # Plot AUC
    plt.plot(history.history['auc'], label='train AUC')
    plt.plot(history.history['val_auc'], label='val AUC')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.show()
    if save_plots:
        plt.savefig(f"plots/{model_name}_auc.png")
    # Plot F1
    plt.plot(history.history['f1'], label='train F1')
    plt.plot(history.history['val_f1'], label='val F1')
    plt.title('Model F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.show()
    if save_plots:
        plt.savefig(f"plots/{model_name}_f1.png")

    # Plot precision
    plt.plot(history.history['precision'], label='train Prec')
    plt.plot(history.history['val_precision'], label='val Prec')
    plt.title('Model Precision Score')
    plt.xlabel('Epoch')
    plt.ylabel('Prec')
    plt.legend()
    plt.show()
    if save_plots:
        plt.savefig(f"plots/{model_name}_precision.png")

    # Plot recall
    plt.plot(history.history['recall'], label='train recall')
    plt.plot(history.history['val_recall'], label='val recall')
    plt.title('Model Recall Score')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()
    if save_plots:
        plt.savefig(f"plots/{model_name}_Recall.png")

    # Plot Accuracy
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    if save_plots:
        plt.savefig(f"plots/{model_name}_accuracy.png")
