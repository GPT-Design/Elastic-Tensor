"""
Elastic-Shear Constraint from LIGO/Virgo Strain Files
====================================================

Standalone script to derive a joint Bayesian upper bound on the dimensionless
elastic-shear propagation parameter

    mu = G / (rho + P)

using public compact‑binary events.  A non‑zero shear modulus adds a small
frequency‑dependent phase delay to GW waveforms; we incorporate that via a
custom phase correction inside Bilby's likelihood and then merge posteriors
across events.

Usage
-----
    python src/elastic_ligo_fit.py \
        --events_dir /path/to/strain \
        --event_list  GW150914 GW190521 \
        --outdir      results/elast \
        --nlive       1024

Dependencies
------------
• bilby[gw] ≥ 2.2     (pip install bilby[gw])
• gwpy, gwosc         (for strain I/O)
• numpy, scipy, pandas, tqdm

"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import bilby
import numpy as np
from bilby.core.prior import PriorDict
from bilby.gw.detector import InterferometerList
from bilby.gw.waveform_generator import WaveformGenerator
from bilby.gw.conversion import generate_all_bbh_parameters
from bilby.gw.prior import BBHPriorDict
from gwpy.timeseries import TimeSeries
from tqdm import tqdm

################################################################################
# Helper functions
################################################################################

def load_strain(event: str, events_dir: Path) -> Dict[str, TimeSeries]:
    """Return a dict {ifo: strain_timeseries} for *event*.

    Looks for local FITS/HDF5 strain files; if missing downloads via GWOSC.
    """
    strain_files = list(events_dir.glob(f"{event}_*.*"))
    if not strain_files:
        from gwosc import datasets, event_gps, download

        gps = event_gps(event)
        metadata = datasets.alertjson(event)
        ifos = metadata["instruments"]
        local: Dict[str, TimeSeries] = {}
        for ifo in ifos:
            fname = download(event, ifo, type="strain", path=str(events_dir))
            local[ifo] = TimeSeries.read(fname)
        return local

    out: Dict[str, TimeSeries] = {}
    for fp in strain_files:
        if fp.suffix == ".fits":
            out[fp.stem.split("_")[-1]] = TimeSeries.read(str(fp), format="fits")
        else:
            out[fp.stem.split("_")[-1]] = TimeSeries.read(str(fp))
    return out


def make_interferometers(strain_dict: Dict[str, TimeSeries]) -> InterferometerList:
    """Build a Bilby InterferometerList with on‑the‑fly PSD estimation."""
    ifos = InterferometerList([])
    for ifo_name, ts in strain_dict.items():
        det = bilby.gw.detector.get_interferometer_with_fake_noise_and_injection(
            ifo_name,
            sampling_frequency=ts.sample_rate.value,
            duration=len(ts) / ts.sample_rate.value,
            start_time=ts.t0.value,
            strain_data=ts.value,
        )
        ifos.append(det)
    return ifos

################################################################################
# Phase‑correction helpers
################################################################################

def phase_correction(frequencies: np.ndarray, parameters: dict) -> np.ndarray:
    """Return extra GW phase (radians) for elastic shear (small‑mu limit)."""
    mu = parameters.get("mu", 0.0)
    distance = parameters.get("luminosity_distance", 1.0)  # Mpc
    c_km_s = 2.99792458e5
    D_km = distance * 3.085677581491367e19 / 1000.0
    return 2 * np.pi * D_km / c_km_s * mu * frequencies


def modified_waveform(parameters: dict, wfgen: WaveformGenerator):
    """Inject phase shift into standard frequency‑domain waveform."""
    hp, hc = wfgen.frequency_domain_strain(parameters)
    phase = phase_correction(wfgen.frequency_array, parameters)
    hp *= np.exp(-1j * phase)
    hc *= np.exp(-1j * phase)
    return hp, hc

################################################################################
# Per‑event run
################################################################################

def run_single_event(event: str, events_dir: Path, outdir: Path, nlive: int):
    strain = load_strain(event, events_dir)
    ifos = make_interferometers(strain)

    priors: PriorDict = BBHPriorDict(aligned_spin=False)
    priors["mu"] = bilby.core.prior.LogUniform(name="mu", minimum=1e-25, maximum=1e-12)

    wfgen = WaveformGenerator(
        duration=4,
        sampling_frequency=2048,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=generate_all_bbh_parameters,
    )
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=wfgen,
        phase_marginalization=True,
        distance_marginalization=True,
        time_marginalization=True,
    )
    likelihood.waveform_generator.frequency_domain_source_model = (
        lambda p: modified_waveform(p, wfgen)
    )

    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler="dynesty",
        nlive=nlive,
        outdir=str(outdir / event),
        label=event,
        resume=False,
        verbose=True,
    )
    result.save_to_file()
    return result

################################################################################
# Posterior combination
################################################################################

def combine_results(result_files: List[Path], outfile: Path):
    import pandas as pd
    dfs = [bilby.result.read_in_result(str(rf)).posterior[["mu"]] for rf in result_files]
    pd.concat(dfs, ignore_index=True).to_csv(outfile, index=False)
    print(f"[elastic] joint posterior → {outfile}")

################################################################################
# CLI
################################################################################

def main():
    p = argparse.ArgumentParser(description="Elastic‑Shear GW constraint tool")
    p.add_argument("--events_dir", type=Path, required=True)
    p.add_argument("--event_list", nargs="*", default=None,
                   help="Space‑separated list of event names; defaults to all files in --events_dir")
    p.add_argument("--outdir", type=Path, default=Path("results/elast"))
    p.add_argument("--nlive", type=int, default=1024)
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.event_list is None:
        args.event_list = sorted({fp.stem.split("_")[0] for fp in args.events_dir.iterdir()})

    result_files = []
    for ev in tqdm(args.event_list, desc="Events"):
        res = run_single_event(ev, args.events_dir, args.outdir, args.nlive)
        result_files.append(Path(res.outdir) / f"{ev}_result.json")

    combine_results(result_files, args.outdir / "joint_mu_posterior.csv")


if __name__ == "__main__":
    main()
