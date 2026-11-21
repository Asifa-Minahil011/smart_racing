import matplotlib.pyplot as plt
from src.loader import load_circuit

c = load_circuit("circuits/circuit1.json")
g = c.graph

colors = {"Start": "green", "Checkpoint": "blue", "Intersection": "gray",
          "FuelStation": "red", "NitroPickup": "orange",
          "DeadEnd": "black", "Finish": "purple"}

for e in g.edges.values():
    a, b = g.vertices[e.from_v], g.vertices[e.to_v]
    style = "r--" if e.is_shortcut else "k-"
    plt.plot([a.x, b.x], [a.y, b.y], style, linewidth=1)

for v in g.vertices.values():
    plt.scatter(v.x, v.y, c=colors[v.type], s=120, zorder=3)
    label = v.checkpoint_order if v.checkpoint_order else v.id
    plt.text(v.x + 8, v.y + 8, str(label), fontsize=8)

plt.gca().invert_yaxis()   # screen coordinates: y grows downward
plt.title(c.name)
plt.show()