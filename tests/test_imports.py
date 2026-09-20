import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ImportTests(unittest.TestCase):
    def test_pimnet_import_does_not_load_models_or_modify_environment(self):
        root = str(Path(__file__).resolve().parents[1])
        script = """
import os, sys
before = dict(os.environ)
import rptcr
import rptcr.pimnet
import rptcr.pimnet_graph
assert dict(os.environ) == before
assert 'tensorflow' not in sys.modules
assert 'torch' not in sys.modules
assert os.listdir('.') == []
"""
        env = dict(os.environ)
        env['PYTHONPATH'] = root
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        with tempfile.TemporaryDirectory() as directory:
            subprocess.check_call([sys.executable, '-c', script], cwd=directory, env=env)


if __name__ == '__main__':
    unittest.main()
