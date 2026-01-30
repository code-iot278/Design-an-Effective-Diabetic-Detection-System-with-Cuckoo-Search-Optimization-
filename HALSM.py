# =========================================
# Feature Selection using HALSM (Diabetic Detection)
# =========================================

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# Step 1: Load dataset
# -------------------------------
input_csv = "/content/drive/MyDrive/diabetic_data.csv"
df = pd.read_csv(input_csv)

# Assume last column is label
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

n_features = X.shape[1]

# -------------------------------
# Step 2: Fitness function
# -------------------------------
def fitness_function(solution, X, y):
    """
    solution: binary array of selected features (1 = selected)
    """
    if np.sum(solution) == 0:
        return 0  # avoid zero features
    
    X_sel = X[:, solution == 1]
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    score = cross_val_score(clf, X_sel, y, cv=3).mean()
    return score

# -------------------------------
# Step 3: HALSM parameters
# -------------------------------
n_agents = 10      # population size
n_iter = 20        # number of iterations
threshold = 0.5    # binary threshold

# Initialize agents randomly (0 = not selected, 1 = selected)
agents = np.random.randint(0, 2, (n_agents, n_features))

# -------------------------------
# Step 4: HALSM Optimization loop
# -------------------------------
best_agent = None
best_score = -1

for iteration in range(n_iter):
    fitness_scores = np.array([fitness_function(a, X, y) for a in agents])
    
    # Update best agent
    max_idx = np.argmax(fitness_scores)
    if fitness_scores[max_idx] > best_score:
        best_score = fitness_scores[max_idx]
        best_agent = agents[max_idx].copy()
    
    # ---------------------------
    # Ant Lion operator (exploration)
    # ---------------------------
    for i in range(n_agents):
        rand_agent = agents[np.random.randint(0, n_agents)]
        # Move towards best agent
        agents[i] = agents[i] + 0.1 * (best_agent - agents[i]) + 0.1 * (rand_agent - agents[i])
    
    # ---------------------------
    # Spider Monkey operator (social interaction)
    # ---------------------------
    for i in range(n_agents):
        agents[i] = agents[i] + 0.05 * (np.mean(agents, axis=0) - agents[i])
    
    # Binary threshold
    agents = np.where(agents > threshold, 1, 0)
    
    print(f"Iteration {iteration+1}/{n_iter}, Best fitness: {best_score:.4f}")

# -------------------------------
# Step 5: Selected features
# -------------------------------
selected_features = np.where(best_agent == 1)[0]
print("Selected feature indices:", selected_features)

# Save reduced dataset
X_selected = X[:, selected_features]
reduced_df = pd.DataFrame(X_selected)
reduced_df['Label'] = y
output_csv = "/content/drive/MyDrive/diabetic_features_selected.csv"
reduced_df.to_csv(output_csv, index=False)
print(f"Reduced dataset saved to: {output_csv}")
