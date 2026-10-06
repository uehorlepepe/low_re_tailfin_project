import os
import subprocess
import glob
import shutil

runs_dir = os.path.expanduser("~/low_re_tailfin_project/runs")
cases = sorted([d for d in os.listdir(runs_dir) if d.startswith("case_")])

print(f"=== Starting Batch CFD Sweep ({len(cases)} cases total) ===\n")

for i, case_name in enumerate(cases, 1):
    case_path = os.path.join(runs_dir, case_name)
    header = f"[{i}/{len(cases)}] {case_name}"
    
    print(f"{header} ... running", flush=True)
    
    try:
        cmds = [
            "openfoam -c 'blockMesh'",
            "openfoam -c 'surfaceFeatureExtract'",
            "openfoam -c 'snappyHexMesh -overwrite'",
            "openfoam -c 'simpleFoam'"
        ]
        
        for cmd in cmds:
            subprocess.run(
                cmd,
                shell=True,
                cwd=case_path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
        # Smart Cleanup: Keep '0' and ONLY the final timestep (e.g., '1000'), delete all intermediate folders
        subdirs = [d for d in os.listdir(case_path) if os.path.isdir(os.path.join(case_path, d)) and d.isdigit()]
        numeric_dirs = sorted([int(d) for d in subdirs])
        if numeric_dirs:
            final_time = str(max(numeric_dirs))
            for d in subdirs:
                if d != "0" and d != final_time:
                    shutil.rmtree(os.path.join(case_path, d))
                
        print(f"\033[A\033[2K{header} ... [✓] PASSED (Kept timestep {final_time})")

    except subprocess.CalledProcessError as e:
        print(f"\033[A\033[2K{header} ... [✗] FAILED")
        err_log_path = os.path.join(case_path, "fatal_error.log")
        with open(err_log_path, "w") as f:
            f.write(e.stderr)

print("\n=== Batch Sweep Execution Finished! ===")
