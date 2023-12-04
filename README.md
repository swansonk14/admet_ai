# ADMET-AI

This git repo contains the code for ADMET-AI, an ADMET prediction platform that uses [Chemprop-RDKit]((https://github.com/chemprop/chemprop)) models trained on ADMET datasets from the Therapeutics Data Commons ([TDC](https://tdcommons.ai/)). ADMET-AI can be used to make ADMET predictions on new molecules via the command line, via the Python API, or via a web server. A live web server hosting ADMET-AI is at [admet.ai.greenstonebio.com](https://admet.ai.greenstonebio.com)

ADMET-AI will be described in a forthcoming paper.

Instructions to reproduce the results in our paper are in [docs/reproduce.md](docs/reproduce.md).

- [Installation](#installation)
- [Predicting ADMET properties](#predicting-admet-properties)
  * [Command line tool](#command-line-tool)
  * [Python module](#python-module)
  * [Web server](#web-server)

## Installation

ADMET-AI can be installed in a few minutes on any operating system using pip (optionally within a conda environment). If a GPU is available, it will be used by default, but the code can also run on CPUs only.

Optionally, create a conda environment.

```bash
conda create -y -n admet_ai python=3.10
conda activate admet_ai
```

Install ADMET-AI via pip.

```bash
pip install admet_ai
```

Alternatively, clone the repo and install ADMET-AI locally.

```bash
git clone https://github.com/swansonk14/admet_ai.git
cd admet_ai
pip install -e .
```

By default, the pip installation only includes dependencies required for making ADMET predictions, either via the command line or via the Python API. To install dependencies required for processing TDC data or plotting TDC results, run `pip install admet_ai[tdc]`. To install dependencies required for hosting the ADMET-AI web server, run `pip install admet_ai[web]`.

If there are version issues with the required packages, create a conda environment with specific working versions of the packages as follows.

```bash
pip install -r requirements.txt
pip install -e .
```

Note: If you get the issue `ImportError: libXrender.so.1: cannot open shared object file: No such file or directory`, run `conda install -c conda-forge xorg-libxrender`.

## Predicting ADMET properties

ADMET-AI can be used to make ADMET predictions in three ways: (1) as a command line tool, (2) as a Python module, or (3) as a web server.

### Command line tool

ADMET predictions can be made on the command line with the `admet_predict` command, as illustrated below.

```bash
admet_predict \
    --data_path data.csv \
    --save_path preds.csv \
    --smiles_column smiles
```

This command assumes that there exists a file called `data.csv` with SMILES strings in the column `smiles`. The predictions will be saved to a file called `preds.csv`.

### Python module

ADMET predictions can be made using the `predict` function in the `admet_ai` Python module, as illustrated below.

```python
from admet_ai import ADMETModel

model = ADMETModel()
preds = model.predict(smiles="O(c1ccc(cc1)CCOC)CC(O)CNC(C)C")
```

If a SMILES string is provided, then `preds` is a dictionary mapping property names to values. If a list of SMILES strings is provided, then `preds` is a Pandas DataFrame where the index is the SMILES and the columns are the properties.

### Web server

ADMET predictions can be made using the ADMET-AI web server, as illustrated below. Note: Running the following command requires additional web dependencies (i.e., `pip install admet_ai[web]`).

```bash
admet_web
```

Then navigate to http://127.0.0.1:5000 to view the website.
