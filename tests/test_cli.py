import subprocess
import sys

def test_help():
    result = subprocess.run(
        [sys.executable, "src/elastic_ligo_fit.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "constraint tool" in result.stdout
