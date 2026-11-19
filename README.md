# SatForensics: scripted evaluation on SatNOGS observations

This repository contains the complete pipeline behind the revised manuscript: metadata collection from the public SatNOGS Network API, waterfall download and conversion, SGP4 geometry, the attack-injection model, the SatForensics model and its baselines, the hash-chained forensic ledger and its benchmark, and the scripts that produce every table, macro and figure of the paper.

## Layout

| Path | Content |
| --- | --- |
| `satforensics/` | library: API client, waterfall parsing, SGP4 geometry, injection, features, model, metrics, ledger, classical baselines, experiment helpers |
| `scripts/` | pipeline stages (`collect_metadata.py`, `fetch_waterfalls.py`, `build_dataset.py`, `geometry_check.py`, `data_summary.py`, `run_experiments.py`, `triage_rules.py`, `make_tables.py`) |
| `figures/` | figure scripts |
| `run_all.sh` | end-to-end driver |
| `data/` | `meta.jsonl` (collected observation records), `dataset.npz` (extracted matrices and labels), `dataset_stats.json` |
| `results/` | all result files, generated LaTeX tables and macros, and figures of the paper |

## Requirements

Python 3.10 or later and the packages in `requirements.txt`. Training uses the CPU; the full run takes several hours on a multi-core machine.

## Reproducing the paper

```
pip install -r requirements.txt
bash run_all.sh data results
```

The driver skips collection and download when `data/dataset.npz` exists, so the command above re-runs all experiments on the released dataset. `EPOCHS` (default 40) and `RUNS` (default 5) control training length and the number of independent runs, for example `EPOCHS=2 RUNS=1 bash run_all.sh data out_test` for a quick check. `SF_THREADS` sets the number of CPU threads.

To collect new data instead, remove `data/dataset.npz`; the driver then queries the SatNOGS API (`scripts/collect_metadata.py`, 600 random 15-minute windows with seed 7 by default), downloads the waterfall images (`scripts/fetch_waterfalls.py`) and builds the dataset. Because SatNOGS is a live archive, a new collection will contain different records. The collection reported in the paper stopped after 372 windows (`data/meta.jsonl.windows`).

## Protocol summary

* Natural label: SatNOGS status `bad`; normal class: status `good`; `failed` and `unknown` are excluded.
* Attacks (broadband, continuous-wave, swept, pulsed, Doppler-following spoofing) are injected into half of the good observations of each partition after the split.
* Splits are grouped by satellite (70/15/15) for each of the five runs; additional chronological, station and leave-one-satellite-out evaluations are included.
* Baseline hyperparameters are selected on validation data (`--stage search` for the neural models on split 0; per run for SVM and Random Forest).
* The ledger benchmark measures in-memory hash-chain appends only; it does not include durable storage, signatures or network transport.

## Data source

Observation records come from the SatNOGS Network (https://network.satnogs.org), operated by the Libre Space Foundation and the SatNOGS community. Waterfall images are processed in memory; only extracted matrices and metadata are released.

## Citation

Kaushik, K., Al-Zyoud, M., Gaur, P. *et al.* SatForensics: a transformer-based multi-modal framework for cyber-attack detection, attribution, and chain-of-custody forensics in low-earth-orbit satellite communication networks. *Sci Rep* (2026). [https://doi.org/10.1038/s41598-026-81042-7](https://doi.org/10.1038/s41598-026-81042-7)
