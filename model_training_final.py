import numpy as np
import pandas as pd
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
from keras import layers
from sklearn.preprocessing import StandardScaler
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from metrics_tracking import F1Score, plot_metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve

#import the datasets to test
def load_data(data_file_name):
    data = np.load(f"Preprocessed_Data/{data_file_name}")
    # Access arrays by their keys
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]
    X_train = X_train[..., :-1] #TEMP FIX REMOVE LATER - FIX PREPROCESSING TO GET RID OF STRING COLUMN
    X_test  = X_test[..., :-1] #TEMP FIX REMOVE LATER
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    data.close()
    return X_train, y_train, X_test, y_test

def standardize(X_train, y_train, X_test, y_test):
    # Clip outliers 
    X_train = np.clip(X_train, -1e6, 1e6)
    X_test  = np.clip(X_test,  -1e6, 1e6)
    # Standardize features
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat  = X_test.reshape(-1,  X_test.shape[-1])
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled  = scaler.transform(X_test_flat)
    X_train = X_train_scaled.reshape(X_train.shape)
    X_test  = X_test_scaled.reshape(X_test.shape)
    # y_train = y_train.astype("int32")
    # y_test  = y_test.astype("int32")
    y_train_oh = keras.utils.to_categorical(y_train, num_classes=2)
    y_test_oh  = keras.utils.to_categorical(y_test, num_classes=2)
    return X_train, y_train_oh, X_test, y_test_oh

def create_model_road(): #this is the same model we'll always use for all. 
    model = keras.Sequential()
    model.add(layers.Input(shape=(600, 23)))
    model.add(layers.Conv1D(32, 4, activation='relu'))
    model.add(layers.GlobalAveragePooling1D()) #was GlobalMaxPooling1D
    model.add(layers.Dense(2, activation='softmax'))  # ← change
    return model

def train_model(model, X_train, y_train): 
        #train data 
    b_size = 32
    callbacks = [
        ModelCheckpoint("saved_models/model_road_32cnn_stm32_chkpt.keras", monitor='val_auc', mode='max', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5, patience=8, min_lr=1e-11, verbose=1),
        EarlyStopping(monitor='val_auc', mode='max',  patience=15, verbose=1, restore_best_weights=True)
    ]
    model.compile(optimizer=keras.optimizers.Adam(1e-3), 
                loss = 'categorical_crossentropy', 
                metrics = ['accuracy', 'auc'
                    # SoftmaxAUC(name='auc')
                            #  keras.metrics.AUC(name='auc')# 
                #   keras.metrics.Precision(name='precision'),
                #   keras.metrics.Recall(name='recall')#,
                #   F1Score(name="f1")
    ])
    class_weight = {0: 1.0, 1: 9.581613508442777}
    history = model.fit(X_train, y_train, batch_size = b_size, epochs = 50, validation_split=0.1, callbacks = callbacks, verbose = 1,
                        class_weight=class_weight,
                        )
    return history 

def examine_weights(model, X_test, y_test): 
    attack_idx  = np.where(y_test == 1)[0]
    normal_idx  = np.where(y_test == 0)[0]
    # force selection
    attack_i = attack_idx[0]
    normal_i = normal_idx[0]
    X_sel = np.stack([
        X_test[attack_i],
        X_test[normal_i]
    ])
    y_sel = np.array([1, 0])
    print("Selected labels:", y_sel)
    probs = model.predict(X_sel, verbose=0)
    for i in range(len(probs)):
        print(f"Sample {i} | true label={y_sel[i]}")
        # prediction = 
        print(f"  normal = {probs[i][0]:.10e}")
        print(f"  attack = {probs[i][1]:.10e}")


def print_test_metrics(X_test, y_test, model):
    testing_acc = model.evaluate(X_test,y_test, verbose=1)
    print(f"Test loss: {testing_acc[0]}")
    print(f"Test accuracy: {testing_acc[1]}")
    print(f"Test AUC: {testing_acc[2]}")
    # print(f"Test Precision: {testing_acc[3]}")
    # print(f"Test Recall: {testing_acc[4]}")
    # print(f"Test F1: {testing_acc[5]}")
def print_report_and_score(model, X_test, y_test):
    y_pred_probs = model.predict(X_test).ravel()     # shape: (N,)
    y_pred = (y_pred_probs >= 0.5).astype(int)       # threshold
    print(classification_report(y_test, y_pred, target_names=["Ambient", "Attack"]))
    # --- Correct ROC-AUC for binary classifier ---
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    return y_pred

def display_confusion_matrix(y_test, y_pred, model_name="best_ROAD_model128.keras"):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["Attack", "Ambient"],
        yticklabels=["Attack", "Ambient"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    plt.savefig(f"{model_name}_confusion_matrix.png")



def main(): 
    window_frequency = 200
    window_sec = 3
    data_file_name = f"roads_canids_windows_{window_frequency}hz_{window_sec}s.npz"
    X_train, y_train, X_test, y_test = load_data(data_file_name)
    X_train, y_train, X_test, y_test= standardize(X_train, y_train, X_test, y_test)
    model = create_model_road()
    history = train_model(model, X_train, y_train) #train model
    print_test_metrics(X_test, y_test, model) #create metrics and plots
    examine_weights(model, X_test, y_test)
    model_save_name = "stm32_ROAD_model32_2Output.keras"
    model.save(f"saved_models/{model_save_name}")
    plot_metrics(history, model_save_name)
    y_pred = print_report_and_score(model, X_test, y_test)
    display_confusion_matrix(y_test, y_pred, model_save_name)

if __name__ == '__main__':
    main()