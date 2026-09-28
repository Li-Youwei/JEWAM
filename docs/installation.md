# Installation

[Documentation index](README.md) · [Project overview](../README.md)

After cloning the repository, run setup commands from its root.

Use **Python 3.10**. Full training and rollout evaluation target Linux with an
NVIDIA GPU; CPU tests and CLI checks can run on macOS. Activate your environment
before running scripts; they do not assume a particular conda installation.

```bash
git clone https://github.com/Li-Youwei/JEWAM.git
cd JEWAM
conda env create -f environment.yml
conda activate jewam
```

Alternatively, in an existing Python 3.10 environment:

```bash
python -m pip install -e '.[train]'
```

The conda environment installs the package with the `train` extra. Both setup
options use editable installation so changes in the checkout are available to
Python immediately. Run the documented commands from the repository root.

Install a matching PyTorch/torchvision CUDA build for your machine. For core
CPU checks only, use `python -m pip install -e '.[dev]'`.
`stable-pretraining==0.1.6` is an API compatibility pin: later releases change
`Manager.ckpt_path` semantics. The constraints in
[`pyproject.toml`](../pyproject.toml) specify release compatibility;
they are **not an archived lockfile from the original training machine**.

| Installation | Included dependencies |
|---|---|
| `python -m pip install -e .` | Core model, preprocessing, configuration, and CPU test dependencies |
| `python -m pip install -e '.[train]'` | Core plus stable-pretraining, Lightning, and TensorBoard |
| `python -m pip install -e '.[train,eval]'` | Training plus LIBERO simulation and optional video dependencies |
| `python -m pip install -e '.[dev]'` | Core plus Ruff for linting and core CPU checks |
| `python -m pip install -e '.[train,dev]'` | Training plus Ruff for the full local test suite and CLI checks |

For rollout evaluation, install the simulation dependencies and
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO):

```bash
python -m pip install -e '.[train,eval]'
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
python -m pip install -e third_party/LIBERO --no-deps
```

The `eval` extra constrains the legacy simulator's NumPy/PyTorch compatibility.
Do not install LIBERO's historical dependency set over this
project's Transformers dependency. Configure LIBERO's asset, BDDL, and initial
state paths as described upstream. For headless Linux rendering, the evaluator
defaults to `MUJOCO_GL=egl`; it respects an existing value.
