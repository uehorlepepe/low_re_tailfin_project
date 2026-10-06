import os
import shutil
import subprocess
import numpy as np
import trimesh

TEMPLATE_DIR = os.path.expanduser("~/low_re_tailfin_project/openfoam_template")
RUNS_DIR = os.path.expanduser("~/low_re_tailfin_project/runs")
CAD_SCRIPT = os.path.expanduser("~/low_re_tailfin_project/cad_generator/build_tailfin.py")

def generate_parameter_space(num_cases=30):
    """Generates 30 parameter variations across sweep, AR, and taper."""
    np.random.seed(42)  # For reproducible parameter combinations
    sweeps = np.linspace(10.0, 35.0, num_cases)
    ars = np.linspace(1.5, 3.5, num_cases)
    tapers = np.linspace(0.4, 0.8, num_cases)
    
    # Shuffle AR and tapers to sample space effectively
    np.random.shuffle(ars)
    np.random.shuffle(tapers)
    
    return list(zip(sweeps, ars, tapers))

def clean_stl(stl_path):
    """Ensures each generated tailfin geometry is watertight before meshing."""
    mesh = trimesh.load(stl_path, force='mesh')
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    mesh.export(stl_path, file_type='stl')


def execute_case(case_dir):
    """Runs OpenFOAM execution sequence using the native OpenFOAM app wrapper."""

    # Prefix your commands with the openfoam environment launcher
    commands = [
        "openfoam -c 'surfaceFeatureExtract'",
        "openfoam -c 'blockMesh'",
        "openfoam -c 'snappyHexMesh -overwrite'",
        "openfoam -c 'potentialFoam -writep'",
        "openfoam -c 'simpleFoam'"
    ]

    for cmd in commands:
        res = subprocess.run(cmd, shell=True, cwd=case_dir, executable="/bin/zsh")
        if res.returncode != 0:
            print(f"[!] Command failed in {case_dir}")
            return False

    # Purge all numbered time folders automatically to save space (keep 0/)
    for tdir in os.listdir(case_dir):
        if tdir.isdigit() and tdir != "0":
            shutil.rmtree(os.path.join(case_dir, tdir), ignore_errors=True)

    return True

if __name__ == "__main__":
    os.makedirs(RUNS_DIR, exist_ok=True)
    param_list = generate_parameter_space(30)
    
    for i, (sweep, ar, taper) in enumerate(param_list, start=1):
        case_name = f"case_{i:02d}"
        case_path = os.path.join(RUNS_DIR, case_name)
        stl_path = os.path.join(case_path, "constant", "triSurface", "tailfin.stl")
        
        print(f"\n================ Case {i}/30: {case_name} (Sweep={sweep:.1f}°, AR={ar:.2f}, Taper={taper:.2f}) ================")
        
        # 1. Copy template case
        if os.path.exists(case_path):
            shutil.rmtree(case_path)
        shutil.copytree(TEMPLATE_DIR, case_path)
        
        # 2. Generate CAD variation
        cad_cmd = [
            "python3", CAD_SCRIPT,
            "--sweep", str(sweep),
            "--ar", str(ar),
            "--taper", str(taper),
            "--out", stl_path
        ]
        res_cad = subprocess.run(cad_cmd, capture_output=True, text=True)
        if res_cad.returncode != 0:
            print(f"[!] CAD generation failed for {case_name}: {res_cad.stderr}")
            continue
            
        # 3. Clean & Repair STL
        clean_stl(stl_path)
        
        # 4. Mesh & Solve
        success = execute_case(case_path)
        if success:
            print(f"[✓] {case_name} meshed and solved successfully.")
