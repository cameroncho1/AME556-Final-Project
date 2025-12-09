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
- Interactive runs require GPU/GL support; fall back to headless mode if viewer fails to open.

## Running the tasks

- **Task 1 (constraint enforcement + PD joints):**
	```
	python task1.py --interactive --sim-time 3.0 --perturb
	```
	Omitting `--interactive` runs headless; add `--controller zero` to demonstrate passive violation detection.

- **Task 2 (standing QP controller):**
	```
	python task2.py --interactive --sim-time 3.0
	```
	The controller tracks the prescribed vertical CoM trajectory (0.45→0.55→0.4 m) using a contact-force QP similar to HW4. Use `--perturb` to test robustness.

- Append `--ignore-violations` to either script to keep the simulation running after a constraint break (useful for debugging while still logging warnings).