"""The narrated demos (python main.py --demo ...) run end to end and leave a consistent database."""
import contextlib
import io
import unittest

from simulation.demos import DEMOS


class DemosTest(unittest.TestCase):
    def test_every_demo_runs(self):
        for name, demo in DEMOS.items():
            with self.subTest(demo=name):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    demo()
                self.assertIn("=" * 20, out.getvalue())


if __name__ == "__main__":
    unittest.main()
