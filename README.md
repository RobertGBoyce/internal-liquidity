# Internal Liquidity

This repository contains Python code for simulating the internal liquidity model from our paper *FX Market Making with Internal Liquidity* by Alexander Barzykin, Eyal Neuman, and myself. The paper is published in Risk cutting edge. 

The code is designed to be used through a Google Colab notebook, so readers can run the simulations without installing anything locally.

## Open in Colab

You can run the notebook here:

```text
https://colab.research.google.com/github/RobertGBoyce/internal-liquidity/blob/main/notebooks/internal_liquidity.ipynb
```

## Installation

Inside the Colab notebook, install the package directly from GitHub:

```python
!pip install git+https://github.com/RobertGBoyce/internal-liquidity.git
```

Then import the package:

```python
import internal_liquidity as il
```

## Repository structure

```text
internal-liquidity/
├── README.md
├── pyproject.toml
├── notebooks/
│   └── internal-liquidity.ipynb
└── src/
    └── internal_liquidity/
        ├── __init__.py
        └── internal_liquidity.py
```

## Citation

If you use this code, please cite our paper:

**FX Market Making with Internal Liquidity**  
Alexander Barzykin, Robert Boyce, and Eyal Neuman  
*Risk*, 2026

```bibtex
@article{barzykin2026fx,
  title={FX Market Making with Internal Liquidity},
  author={Barzykin, Alexander and Boyce, Robert and Neuman, Eyal},
  journal={Risk},
  year={2026}
}
```
