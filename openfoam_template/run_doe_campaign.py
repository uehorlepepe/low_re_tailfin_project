import csv
import subprocess
import os
import re
import numpy as np
from scipy.stats import qmc

def ensure_doe_matrix():
    """Generates doe_matrix.csv if it does not already exist."""
    matrix_file = "doe_matrix.csv"
    if not os.path.exists(matrix_file):
        print("Generating 30-sample Latin Hypercube Design Matrix...")
        sampler = qmc.LatinHypercube(d=3, seed=42)
        sample = sampler.random(n=30)
        
        # Bounds: [Sweep (15-35°), Taper (0.4-0.8), Aspect Ratio (1.5-3.0)]
        l_bounds = [15.0, 0.4, 1.5]
        u_bounds = [35.0, 0.8, 3.0]
        scaled = qmc.scale(sample, l_bounds, u_bounds)
        
        s_ref = 0.05
        headers = ["run_id", "sweep_deg", "taper_ratio", "aspect_ratio", "span_b_m", "c_root_m", "c_tip_m", "Cd", "Cy", "Cl", "status"]
        rows = []
        
        for i, (sweep, taper, ar) in enumerate(scaled, start=1):
            span_b = np.sqrt(ar * s_ref)
            c_root = (2.0 * s_ref) / (span_b * (1.0 + taper))
            c_tip = taper * c_root
            rows.append([
                f"run_{i:02d}", round(float(sweep), 2), round(float(taper), 3), round(float(ar), 2),
                round(float(span_b), 4), round(float(c_root), 4), round(float(c_tip), 4),
                "", "", "", "PENDING"
            ])
            
        with open(matrix_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print("✓ Created doe_matrix.csv\n")

def update_control_dict_reference(c_mac, s_ref):
    """Updates lRef and Aref in system/controlDict for current geometry."""
    path = "system/controlDict"
    if os.path.exists(path):
        with open(path, "r") as f:
            txt = f.read()
        txt = re.sub(r"lRef\s+[\d\.\-+eE]+;", f"lRef            {c_mac:.6f};", txt)
        txt = re.sub(r"Aref\s+[\d\.\-+eE]+;", f"Aref            {s_ref:.6f};", txt)
        with open(path, "w") as f:
            f.write(txt)

def extract_force_coefficients():
    """Parses OpenFOAM postProcessing output for Cd, Cy, and Cl."""
    base_path = "postProcessing/forceCoeffs"
    if not os.path.exists(base_path):
        return None, None, None
        
    time_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))], key=lambda x: float(x) if x.replace('.','',1).isdigit() else -1)
    if not time_dirs:
        return None, None, None
        
    coeffs_file = os.path.join(base_path, time_dirs[-1], "coefficient.dat")
    if os.path.exists(coeffs_file):
        with open(coeffs_file, "r") as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            if lines:
                parts = lines[-1].split()
                # OpenFOAM forceCoeffs format: Time, Cd, Cs (Cy), Cl ...
                return float(parts[1]), float(parts[2]), float(parts[3])
    return None, None, None

def run_campaign():
    ensure_doe_matrix()
    
    matrix_file = "doe_matrix.csv"
    with open(matrix_file, "r") as f:
        rows = list(csv.DictReader(f))

    print(f"==========================================================")
    print(f" LAUNCHING DOE CAMPAIGN ({len(rows)} Iterations)")
    print(f"==========================================================\n")

    for idx, row in enumerate(rows):
        if row.get("status") == "SUCCESS":
            print(f"--> Skipping {row['run_id']} (Already Completed)")
            continue

        run_id = row["run_id"]
        sweep = float(row["sweep_deg"])
        taper = float(row["taper_ratio"])
        ar = float(row["aspect_ratio"])
        c_root = float(row["c_root_m"])
        
        # Calculate Mean Aerodynamic Chord (MAC) & Reference Area
        c_mac = (2.0 / 3.0) * c_root * ((1.0 + taper + taper**2) / (1.0 + taper))
        s_ref = 0.05

        print(f"----------------------------------------------------------")
        print(f" EXECUTING {run_id} [{idx+1}/{len(rows)}]: Sweep={sweep}°, Taper={taper}, AR={ar}")
        print(f"----------------------------------------------------------")

        try:
            # 1. Regenerate parametric STL
            subprocess.run(["python3", "generate_tailfin.py", str(sweep), str(taper), str(ar)], check=True)
            
            # 2. Extract surface features
            subprocess.run(["surfaceFeatureExtract"], check=True)
            
            # 3. Update controlDict reference dimensions
            update_control_dict_reference(c_mac, s_ref)
            
            # 4. Run OpenFOAM mesh and solver
            subprocess.run(["bash", "setup_and_run.sh"], check=True)
            
            # 5. Extract forces
            cd, cy, cl = extract_force_coefficients()
            
            # 6. Update row status in memory & CSV
            row["Cd"] = f"{cd:.6f}" if cd is not None else "N/A"
            row["Cy"] = f"{cy:.6f}" if cy is not None else "N/A"
            row["Cl"] = f"{cl:.6f}" if cl is not None else "N/A"
            row["status"] = "SUCCESS"
            
            print(f"\n✓ {run_id} Finished: Cd={row['Cd']} | Cy={row['Cy']} | Cl={row['Cl']}\n")

        except Exception as e:
            print(f"x Error executing {run_id}: {e}")
            row["status"] = "FAILED"

        # Write updated progress back to CSV after every run
        fieldnames = rows[0].keys()
        with open(matrix_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

if __name__ == "__main__":
    run_campaign()
