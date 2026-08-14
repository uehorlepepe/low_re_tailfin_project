import sys
import numpy as np

def generate_sd8020_coords(n_points=80):
    """Generates normalized SD8020 airfoil coordinates (symmetric 10% t/c)."""
    x = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n_points)))
    
    # SD8020 thickness distribution polynomial fit
    yt = 5.0 * 0.10 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    )
    
    xu = x
    yu = yt
    xl = x[::-1]
    yl = -yt[::-1]
    
    x_coords = np.concatenate([xu, xl[1:]])
    y_coords = np.concatenate([yu, yl[1:]])
    return x_coords, y_coords

def export_stl(filename, sweep_deg, taper_ratio, aspect_ratio, s_ref=0.05):
    # Compute planform geometry
    span_b = np.sqrt(aspect_ratio * s_ref)
    c_root = (2.0 * s_ref) / (span_b * (1.0 + taper_ratio))
    c_tip = taper_ratio * c_root
    dx_sweep = span_b * np.tan(np.radians(sweep_deg))
    
    x_raw, y_raw = generate_sd8020_coords(80)
    n_pts = len(x_raw)
    
    # Root section (z = 0)
    root_x = x_raw * c_root
    root_y = y_raw * c_root
    root_z = np.zeros(n_pts)
    
    # Tip section (z = span_b)
    tip_x = x_raw * c_tip + dx_sweep
    tip_y = y_raw * c_tip
    tip_z = np.full(n_pts, span_b)
    
    facets = []
    
    # 1. Lofted side faces
    for i in range(n_pts - 1):
        p1 = [root_x[i], root_y[i], root_z[i]]
        p2 = [root_x[i+1], root_y[i+1], root_z[i+1]]
        p3 = [tip_x[i+1], tip_y[i+1], tip_z[i+1]]
        p4 = [tip_x[i], tip_y[i], tip_z[i]]
        
        facets.append((p1, p2, p3))
        facets.append((p1, p3, p4))
        
    p1 = [root_x[-1], root_y[-1], root_z[-1]]
    p2 = [root_x[0], root_y[0], root_z[0]]
    p3 = [tip_x[0], tip_y[0], tip_z[0]]
    p4 = [tip_x[-1], tip_y[-1], tip_z[-1]]
    facets.append((p1, p2, p3))
    facets.append((p1, p3, p4))

    # 2. Root Cap (z = 0)
    cx_root, cy_root = np.mean(root_x), np.mean(root_y)
    for i in range(n_pts - 1):
        p1 = [cx_root, cy_root, 0.0]
        p2 = [root_x[i+1], root_y[i+1], 0.0]
        p3 = [root_x[i], root_y[i], 0.0]
        facets.append((p1, p2, p3))
    facets.append(([cx_root, cy_root, 0.0], [root_x[0], root_y[0], 0.0], [root_x[-1], root_y[-1], 0.0]))

    # 3. Tip Cap (z = span_b)
    cx_tip, cy_tip = np.mean(tip_x), np.mean(tip_y)
    for i in range(n_pts - 1):
        p1 = [cx_tip, cy_tip, span_b]
        p2 = [tip_x[i], tip_y[i], span_b]
        p3 = [tip_x[i+1], tip_y[i+1], span_b]
        facets.append((p1, p2, p3))
    facets.append(([cx_tip, cy_tip, span_b], [tip_x[-1], tip_y[-1], span_b], [tip_x[0], tip_y[0], span_b]))

    # Write ASCII STL
    with open(filename, "w") as f:
        f.write("solid tailfin\n")
        for p1, p2, p3 in facets:
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p3) - np.array(p1)
            norm = np.cross(v1, v2)
            norm_val = np.linalg.norm(norm)
            if norm_val > 0:
                norm = norm / norm_val
            else:
                norm = [0, 0, 0]
                
            f.write(f"  facet normal {norm[0]:.6e} {norm[1]:.6e} {norm[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {p1[0]:.6e} {p1[1]:.6e} {p1[2]:.6e}\n")
            f.write(f"      vertex {p2[0]:.6e} {p2[1]:.6e} {p2[2]:.6e}\n")
            f.write(f"      vertex {p3[0]:.6e} {p3[1]:.6e} {p3[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid tailfin\n")
    
    print(f"✓ Generated Watertight STL: {filename} (Sweep={sweep_deg}°, Taper={taper_ratio}, AR={aspect_ratio})")

if __name__ == "__main__":
    sweep = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    taper = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
    ar = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    export_stl("constant/triSurface/tailfin.stl", sweep, taper, ar)
