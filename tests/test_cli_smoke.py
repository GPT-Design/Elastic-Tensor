import subprocess
import sys
from pathlib import Path


def test_help():
    result = subprocess.run([
        sys.executable,
        str(Path("src/elastic_ligo_fit.py")),
        "--help",
    ])
    assert result.returncode == 0
