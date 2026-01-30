# ==========================================
# RERNN with 5-Fold Cross-Validation + LIME & SHAP
# ==========================================

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# For explainability
import shap
from lime import lime_tabular

# -------------------------------
# Step 1: Load dataset
# -------------------------------
input_csv = "/content/drive/MyDrive/diabetic_features_selected.csv"
df = pd.read_csv(input_csv)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

feature_names = df.columns[:-1]

# Encode labels if categorical
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 2: Define RBF Layer
# -------------------------------
class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, units, gamma=None):
        super(RBFLayer, self).__init__()
        self.units = units
        self.gamma = gamma
    
    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.units, input_shape[-1]),
                                       initializer='random_normal',
                                       trainable=True)
        if self.gamma is None:
            self.gamma = 1.0 / input_shape[-1]
    
    def call(self, inputs):
        diff = tf.expand_dims(inputs, axis=1) - tf.expand_dims(self.centers, axis=0)
        l2 = tf.reduce_sum(tf.square(diff), axis=-1)
        return tf.exp(-self.gamma * l2)

# -------------------------------
# Step 3: K-Fold Cross-Validation
# -------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
metrics_list = []

fold = 1
for train_index, test_index in kf.split(X_scaled):
    print(f"==== Fold {fold} ====")
    
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Reshape for RNN: (samples, timesteps=1, features)
    X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # Build RERNN model
    input_layer = tf.keras.Input(shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]))
    rbf_layer = RBFLayer(units=32)(input_layer)
    rnn_layer = tf.keras.layers.SimpleRNN(16, activation='tanh')(rbf_layer)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(rnn_layer)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train_rnn, y_train, validation_split=0.1, epochs=50, batch_size=16, verbose=0)
    
    # Predict
    y_pred_prob = model.predict(X_test_rnn)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    npv = tn / (tn + fn)
    error_rate = 1 - accuracy
    
    print(f"Accuracy: {accuracy*100:.2f}%, Precision: {precision*100:.2f}%, Sensitivity: {sensitivity*100:.2f}%, Specificity: {specificity*100:.2f}%")
    
    metrics_list.append([accuracy, precision, sensitivity, specificity, f1, fpr, fnr, npv, error_rate])
    
    # -------------------------------
    # LIME Explanation
    # -------------------------------
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['No Diabetic', 'Diabetic'],
        mode='classification'
    )
    
    # Explain first sample in test set
    i = 0
    def predict_fn(input_data):
        input_data_rnn = input_data.reshape((input_data.shape[0], 1, input_data.shape[1]))
        return np.hstack([1 - model.predict(input_data_rnn), model.predict(input_data_rnn)])
    
    exp = explainer.explain_instance(X_test[i], predict_fn, num_features=5)
    print("LIME explanation for first test sample:")
    print(exp.as_list())
    
    # -------------------------------
    # SHAP Explanation
    # -------------------------------
    # Use a subset to save computation
    shap_sample = X_train[np.random.choice(X_train.shape[0], min(50, X_train.shape[0]), replace=False)]
    
    # SHAP Kernel Explainer
    def model_predict(x):
        x_rnn = x.reshape((x.shape[0], 1, x.shape[1]))
        return model.predict(x_rnn).flatten()
    
    shap_explainer = shap.KernelExplainer(model_predict, shap_sample)
    shap_values = shap_explainer.shap_values(X_test[:5], nsamples=100)
    
    print("SHAP values for first 5 test samples:")
    print(shap_values)
    
    fold += 1

# -------------------------------
# Step 4: Average metrics across folds
# -------------------------------
metrics_array = np.array(metrics_list)
metrics_mean = metrics_array.mean(axis=0)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy","Precision","Sensitivity","Specificity","F1-score","FPR","FNR","NPV","Error_rate"],
    "Mean": metrics_mean
})

metrics_csv = "/content/drive/MyDrive/RERNN_5fold_metrics.csv"
metrics_df.to_csv(metrics_csv, index=False)
print(f"\nAverage metrics saved to: {metrics_csv}")
print(metrics_df)
