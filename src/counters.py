"""Operation counters and the observer hook. Owner: Asifa Minahil.
Every algorithm accepts optional counters= and observer= arguments.
The Algorithm Lab passes an observer to step through the real algorithm."""

from dataclasses import dataclass, asdict


@dataclass
class Counters:
    visited: int = 0          # vertices discovered
    edges_examined: int = 0   # edges looked at
    expanded: int = 0         # vertices taken from queue/stack/heap and processed
    relaxed: int = 0          # successful distance improvements (Dijkstra/A*)
    heap_push: int = 0
    heap_pop: int = 0

    def as_dict(self):
        return asdict(self)


def notify(observer, name, **data):
    """Call observer(name, data) if one was given.
    Event names: visit, expand, relax, push, pop, path_found."""
    if observer is not None:
        observer(name, data)