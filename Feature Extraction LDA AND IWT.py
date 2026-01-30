# ===========================================
# Feature Extraction from EHR Data
# Using LDA and Improved Wavelet Transform (IWT)
# ===========================================

import pandas as pd
import numpy as np
import pywt  # Wavelet Transform
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# -------------------------------
# Step 1: Load EHR dataset
# -------------------------------
# Assumes last column is label/class for LDA
input_csv = "/content/drive/MyDrive/EHR_data.csv"
ehr_df = pd.read_csv(input_csv)

# Separate features and labels
X = ehr_df.iloc[:, :-1].values  # all columns except last
y = ehr_df.iloc[:, -1].values   # last column as label

# Encode labels if they are categorical
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 2: Linear Discriminant Analysis (LDA)
# -------------------------------
lda = LDA(n_components=None)  # max possible components = n_classes - 1
X_lda = lda.fit_transform(X_scaled, y)
print("LDA-transformed shape:", X_lda.shape)

# -------------------------------
# Step 3: Improved Wavelet Transform (IWT)
# -------------------------------
# Here we use Discrete Wavelet Transform with multiple levels
def iwt_features(signal, wavelet='db4', level=3):
    """
    signal: 1D array
    Returns statistical features of wavelet coefficients
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    features = []
    for c in coeffs:
        features.append(np.mean(c))        # mean
        features.append(np.std(c))         # std deviation
        features.append(np.max(c))         # max
        features.append(np.min(c))         # min
    return np.array(features)

# Apply IWT to each row of X_scaled (treat each patient record as 1D signal)
iwt_feature_list = []
for i in range(X_scaled.shape[0]):
    feats = iwt_features(X_scaled[i, :])
    iwt_feature_list.append(feats)

X_iwt = np.array(iwt_feature_list)
print("IWT feature shape:", X_iwt.shape)

# -------------------------------
# Step 4: Combine LDA + IWT features
# -------------------------------
X_features = np.hstack((X_lda, X_iwt))
print("Combined feature shape:", X_features.shape)

# -------------------------------
# Step 5: Save features to CSV
# -------------------------------
output_csv = "/content/drive/MyDrive/EHR_features.csv"
features_df = pd.DataFrame(X_features)
features_df['Label'] = y  # add original labels
features_df.to_csv(output_csv, index=False)
print(f"Features saved to: {output_csv}")
