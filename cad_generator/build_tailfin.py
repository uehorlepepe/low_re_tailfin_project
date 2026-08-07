import argparse
import math
import os
import cadquery as cq

def load_airfoil_points(dat_filepath, chord_length):
    points = []
    with open(dat_filepath, 'r') as f:
        lines = f.readlines()[1:]  # Skip header line
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                x = float(parts[0]) * chord_length
                y = float(parts[1]) * chord_length
                points.append((x, y))
    return points

def generate_tailfin(sweep_deg, ar, taper, output_stl):
    root_chord = 0.15 
    tip_chord = root_chord * taper
    mean_chord = 0.5 * (root_chord + tip_chord)
    span = ar * mean_chord
    sweep_offset = span * math.tan(math.radians(sweep_deg))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dat_path = os.path.join(script_dir, 'sd8020.dat')
    
    root_pts = load_airfoil_points(dat_path, root_chord)
    tip_pts = load_airfoil_points(dat_path, tip_chord)

    root_wire = cq.Workplane("XY").polyline(root_pts).close()
    tip_wire = (
        cq.Workplane("XY")
        .transformed(offset=(sweep_offset, 0, span))
        .polyline(tip_pts)
        .close()
    )

    tailfin = root_wire.loft(tip_wire)

    os.makedirs(os.path.dirname(os.path.abspath(output_stl)), exist_ok=True)
    cq.exporters.export(tailfin, output_stl)
    print(f"✅ Successfully generated 3D CAD: {output_stl}")
    print(f"   Metrics: Sweep={sweep_deg}°, AR={ar}, Taper={taper} | Span={span:.3f}m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parametric 3D Tailfin CAD Generator")
    parser.add_argument("--sweep", type=float, required=True)
    parser.add_argument("--ar", type=float, required=True)
    parser.add_argument("--taper", type=float, required=True)
    parser.add_argument("--out", type=str, default="tailfin.stl")

    args = parser.parse_args()
    generate_tailfin(args.sweep, args.ar, args.taper, args.out)
