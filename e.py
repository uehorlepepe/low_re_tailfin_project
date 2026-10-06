import os
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Load your DoE matrix parameters
doe = pd.read_csv("openfoam_template/doe_matrix.csv")

X_list = []
y_list = []

print("--- PARSING CASES ---")
# 2. Loop through completed cases 1 to 21
for i in range(1, 22):
    case_name = f"case_{i:02d}"
    force_coeffs_dir = os.path.join("runs", case_name, "postProcessing", "forceCoeffs")
    
    if not os.path.exists(force_coeffs_dir):
        continue
        
    subdirs = os.listdir(force_coeffs_dir)
    if not subdirs:
        continue
    latest_subdir = sorted(subdirs, key=lambda x: float(x) if x.replace('.','',1).isdigit() else 0)[-1]
    coef_file = os.path.join(force_coeffs_dir, latest_subdir, "coefficient.dat")
    
    if os.path.exists(coef_file):
        data = pd.read_csv(coef_file, comment='#', sep=r'\s+', header=None)
        
        # Directly grab Cd (index 1) and Cl (index 4) from the final converged row
        final_row = data.iloc[-1]
        cd = float(final_row[1])
        cl = float(final_row[4])
        
        target = cl / cd if cd != 0 else 0.0
        print(f"Parsed {case_name}: Cd={cd:.6f}, Cl={cl:.6f}, Cl/Cd={target:.6f}")
        
        if i - 1 < len(doe):
            row_data = doe.iloc[i - 1]
            numeric_features = pd.to_numeric(row_data, errors='coerce').dropna().values.astype(float)
            
            X_list.append(numeric_features)
            y_list.append(target)

print("---------------------")

if len(X_list) > 0:
    X = np.array(X_list)
    y = np.array(y_list)

    print("--- DEBUG INFO ---")
    print("Features X shape:", X.shape)
    print("Targets y values:", y)
    print("------------------")

    # Wrap GPR in a pipeline with StandardScaler
    kernel = 1.0 * Matern(nu=1.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10, random_state=42)
    model = make_pipeline(StandardScaler(), gpr)
    
    scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring='r2')

    print("Scaled CV R^2 Scores:", scores)
    print("Scaled Mean R^2:", scores.mean())
else:
    print("No valid case data parsed yet. Check your folder paths!")

