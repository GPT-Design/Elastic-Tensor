# Elastic-Tensor

Elastic-Tensor is a small demo that fits gravitational-wave data for a possible elastic shear component. The main script `elastic_ligo_fit.py` downloads or reads strain files for a list of LIGO/Virgo events and performs Bayesian inference on the shear modulus parameter using the `bilby` framework. Results from multiple events are combined into a single posterior sample set.

## Setup

Create a Python environment with all required packages:

```bash
pip install -r requirements.txt
```

## Example

Suppose your strain files live in `~/gw_data`. You can run the fit on two events and store results in `results/elast` with:

```bash
python src/elastic_ligo_fit.py \
    --events_dir ~/gw_data \
    --event_list GW150914 GW190521 \
    --outdir results/elast \
    --nlive 1024
```

This will create subdirectories for each event inside `results/elast` and a `joint_mu_posterior.csv` file containing the combined posterior.

