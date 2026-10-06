import os, glob, re

runs_dir = os.path.expanduser("~/low_re_tailfin_project/runs")
print(f"{'Case':<10} | {'Min y+':<10} | {'Max y+':<10} | {'Avg y+':<10}")
print("-" * 46)

for i in range(1, 31):
    case_name = f"case_{i:02d}"
    case_path = os.path.join(runs_dir, case_name)
    
    if not os.path.exists(case_path):
        continue
        
    min_y, max_y, avg_y = "N/A", "N/A", "N/A"
    log_files = glob.glob(os.path.join(case_path, "*.log")) + glob.glob(os.path.join(case_path, "log.*"))
    
    for log in log_files:
        try:
            with open(log, "r", errors="ignore") as f:
                for line in f:
                    if "patch tailfin y+" in line:
                        match = re.search(r"min\s*=\s*([0-9\.e\-\+]+),\s*max\s*=\s*([0-9\.e\-\+]+),\s*average\s*=\s*([0-9\.e\-\+]+)", line)
                        if match:
                            min_y, max_y, avg_y = match.groups()
        except:
            pass
            
    print(f"{case_name:<10} | {min_y:<10} | {max_y:<10} | {avg_y:<10}")
