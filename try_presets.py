from src.loader import load_circuit
from src.routing import compare_presets

c = load_circuit("circuits/circuit1.json")
for a, b in [(3, 4), (8, 9)]:
    print(f"Route {a} -> {b}")
    for name, r in compare_presets(c.graph, a, b).items():
        print(f"  {name:9} {r.vertices}  cost={r.cost:.3f}  "
              f"dist={r.distance:.0f}  time={r.time:.1f}  fuel={r.fuel:.1f}")