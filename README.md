# AME556-Final-Project

## Environment Setup

Choose one of the two workflows below. Run commands from this folder (`AME556-Final-Project`).

### Option A: Conda env (recommended if you already use Anaconda/Miniconda)
1) Create env:
```
conda create -n ame556 python=3.10 -y
conda activate ame556
```
2) Install requirements:
```
pip install -r requirements.txt
```

### Option B: Python venv (no Conda)
1) Create and activate venv (Windows PowerShell):
```
python -m venv .venv
& .venv\Scripts\Activate.ps1
```
	On macOS/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```
2) Install requirements:
```
pip install -r requirements.txt
```

### Notes
- `mujoco` wheel requires Python 3.8–3.12; ensure your env uses a supported version.
- If `cvxopt` fails to build on Windows, use a prebuilt wheel for your Python version and install via `pip install <wheel-file>.whl`.
- After activating the env, you can run scripts like `python task1_sim.py --interactive`.