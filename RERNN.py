# ==========================================
# Diabetic Detection using RERNN (RBF + RNN)
# ==========================================

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------
# Step 1: Load dataset
# -------------------------------
input_csv = "/content/drive/MyDrive/diabetic_features_selected.csv"
df = pd.read_csv(input_csv)

# Separate features and label
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode labels if categorical
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Reshape for RNN: (samples, timesteps=1, features)
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# -------------------------------
# Step 2: Define RBF Layer
# -------------------------------
class RBFLayer(tf.keras.layers.Layer):
    def __init__(self, units, gamma=None):
        super(RBFLayer, self).__init__()
        self.units = units
        self.gamma = gamma
    
    def build(self, input_shape):
        # Centers of RBF neurons
        self.centers = self.add_weight(name='centers',
                                       shape=(self.units, input_shape[-1]),
                                       initializer='random_normal',
                                       trainable=True)
        # Gamma (width)
        if self.gamma is None:
            self.gamma = 1.0 / input_shape[-1]
    
    def call(self, inputs):
        # Compute squared distance to centers
        diff = tf.expand_dims(inputs, axis=1) - tf.expand_dims(self.centers, axis=0)
        l2 = tf.reduce_sum(tf.square(diff), axis=-1)
        return tf.exp(-self.gamma * l2)

# -------------------------------
# Step 3: Build RERNN model
# -------------------------------
input_layer = tf.keras.Input(shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]))
rbf_layer = RBFLayer(units=32)(input_layer)             # 32 RBF neurons
rnn_layer = tf.keras.layers.SimpleRNN(16, activation='tanh')(rbf_layer)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(rnn_layer)  # binary classification

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# Step 4: Train model
# -------------------------------
history = model.fit(
    X_train_rnn, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    verbose=1
)

# -------------------------------
# Step 5: Predict on test data
# -------------------------------
y_pred_prob = model.predict(X_test_rnn)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# -------------------------------
# Step 6: Compute classification metrics
# -------------------------------
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = precision_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)  # Recall
specificity = tn / (tn + fp)
f1 = f1_score(y_test, y_pred)
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)
npv = tn / (tn + fn)
error_rate = 1 - accuracy

# -------------------------------
# Step 7: Print metrics
# -------------------------------
print("==== RERNN Classification Metrics ====")
print(f"Accuracy       : {accuracy*100:.2f}%")
print(f"Precision      : {precision*100:.2f}%")
print(f"Sensitivity    : {sensitivity*100:.2f}%")
print(f"Specificity    : {specificity*100:.2f}%")
print(f"F1-score       : {f1*100:.2f}%")
print(f"FPR            : {fpr*100:.2f}%")
print(f"FNR            : {fnr*100:.2f}%")
print(f"NPV            : {npv*100:.2f}%")
print(f"Error rate     : {error_rate*100:.2f}%")

# -------------------------------
# Step 8: Save metrics to CSV
# -------------------------------
metrics_df = pd.DataFrame({
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Sensitivity": [sensitivity],
    "Specificity": [specificity],
    "F1-score": [f1],
    "FPR": [fpr],
    "FNR": [fnr],
    "NPV": [npv],
    "Error_rate": [error_rate]
})

metrics_csv = "/content/drive/MyDrive/RERNN_classification_metrics.csv"
metrics_df.to_csv(metrics_csv, index=False)
print(f"Metrics saved to: {metrics_csv}")
