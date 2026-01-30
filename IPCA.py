# ===============================
# Improved PCA (IPCA) for EHR data
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# -------------------------------
# Step 1: Load EHR dataset
# -------------------------------
# Example CSV path; replace with your actual EHR dataset
ehr_path = ""
ehr_df = pd.read_csv(ehr_path)

# -------------------------------
# Step 2: Handle missing values
# -------------------------------
# Use median imputation (common for EHR data)
imputer = SimpleImputer(strategy='median')
ehr_data_imputed = imputer.fit_transform(ehr_df)

# -------------------------------
# Step 3: Feature scaling
# -------------------------------
scaler = StandardScaler()
ehr_scaled = scaler.fit_transform(ehr_data_imputed)

# -------------------------------
# Step 4: Apply PCA
# -------------------------------
# Choose number of components or explained variance
n_components = 0.95  # retain 95% variance
pca = PCA(n_components=n_components)
ehr_pca = pca.fit_transform(ehr_scaled)

# -------------------------------
# Step 5: Output results
# -------------------------------
# Explained variance ratio for each component
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_variance)
print("Number of components selected:", pca.n_components_)

# PCA-transformed data
ehr_pca_df = pd.DataFrame(ehr_pca, columns=[f'PC{i+1}' for i in range(ehr_pca.shape[1])])
print(ehr_pca_df.head())

# Optional: Save transformed data
ehr_pca_df.to_csv("/content/drive/MyDrive/EHR_PCA_transformed.csv", index=False)
