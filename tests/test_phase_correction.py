import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from elastic_ligo_fit import phase_correction

def test_phase_correction_zero_mu():
    freqs = np.linspace(0, 200, 201)
    ph = phase_correction(freqs, {"mu": 0})
    assert np.all(ph == 0)

def test_phase_correction_linear_mu():
    freqs = np.linspace(0, 200, 201)
    ph1 = phase_correction(freqs, {"mu": 1e-20})
    ph2 = phase_correction(freqs, {"mu": 2e-20})
    idx = np.where(freqs == 100)[0][0]
    assert ph2[idx] == 2 * ph1[idx]
