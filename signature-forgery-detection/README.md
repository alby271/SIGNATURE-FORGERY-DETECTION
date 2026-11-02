# Signature Forgery Detection

Read the [complete project guide](../README.md) for the problem's importance and use cases, methods, measured results, prerequisites, download instructions, metrics, and limitations.

## Quick start

From this directory, create a Python 3.13 environment and install dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open [signature_forgery_detection.ipynb](signature_forgery_detection.ipynb) in VS Code, select this folder's `.venv` kernel, and run all cells in order. The first run trains models locally. Saved notebook outputs and [exported metrics](results/notebook_run/metrics.csv) can also be viewed without training.

Run the lightweight checks:

```powershell
.\.venv\Scripts\python.exe test_notebook.py
```

The original `signature_verify.py` baseline is retained. Its historical results use a different evaluation protocol; consult the main guide before comparing tables.
