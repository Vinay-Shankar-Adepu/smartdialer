"""The state diagrams in ARCHITECTURE.md show exactly the transitions the code allows."""
import os
import re
import unittest

from core.state_machines import AGENT_TRANSITIONS, CALL_TRANSITIONS

DOC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ARCHITECTURE.md")


def diagram_edges(heading: str) -> set:
    with open(DOC, encoding="utf-8") as fh:
        text = fh.read()
    section = text.split(f"### {heading}", 1)[1]
    block = re.search(r"```mermaid\s*\nstateDiagram-v2\n(.*?)```", section, re.S).group(1)
    edges = set()
    for src, dst in re.findall(r"^\s*(\w+)\s*-->\s*(\w+)", block, re.M):
        if "[*]" not in (src, dst):
            edges.add((src, dst))
    return edges


def code_edges(table) -> set:
    return {(s.value, t.value) for s, targets in table.items() for t in targets}


class StateDiagramsMatchCodeTest(unittest.TestCase):
    def test_agent_diagram(self):
        self.assertEqual(diagram_edges("Agent"), code_edges(AGENT_TRANSITIONS))

    def test_call_diagram(self):
        self.assertEqual(diagram_edges("Call"), code_edges(CALL_TRANSITIONS))


if __name__ == "__main__":
    unittest.main()
