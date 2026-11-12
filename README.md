# Random-Effects Centroids for Domain Generalization

Official implementation of **Random-Effects Centroids for Domain Generalization** (NeurIPS 2026).

Jinyong Jeong, Hyungu Kahng, Seoung Bum Kim  
Korea University

[[Paper]](https://openreview.net/forum?id=GFYiUzBsqE)

## Repository structure

```
REC/
└── WILDS/   # Official WILDS codebase (v2.0.1) with REC added
```

REC is implemented in [`WILDS/examples/algorithms/REC.py`](WILDS/examples/algorithms/REC.py).
The experiments in the paper used WILDS v2.0.0, which has the same training code as v2.0.1.

## Installation

```bash
git clone https://github.com/jyjeongku/REC.git
cd REC/WILDS
pip install -e .
```

See [`WILDS/README.md`](WILDS/README.md) for the dependencies and dataset downloads.

## Usage

Run from the `WILDS` directory:

```bash
python examples/run_expt.py --dataset [DATASET] --algorithm REC --lambda_avg [LAMBDA] --root_dir [DATA_DIR]
```

## Citation

```bibtex
@inproceedings{jeong2026randomeffects,
  title     = {Random-Effects Centroids for Domain Generalization},
  author    = {Jeong, Jinyong and Kahng, Hyungu and Kim, Seoung Bum},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=GFYiUzBsqE}
}
```

## Acknowledgements

This code is built upon [WILDS](https://github.com/p-lambda/wilds).
