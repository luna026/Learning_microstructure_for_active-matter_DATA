import os
os.environ["PYTHON_JULIAPKG_ENV"] = "/home/wdasgupta/miniconda3/envs/pysr_env"
os.environ["JULIA_CONDAPKG_BACKEND"] = "System"
from pysr import PySRRegressor
import numpy as np
#from scipy.special import erf
import joblib
import sympy
import pickle
np.random.seed(2001)

# --- load dataset ---
X = np.load("inp_PCF_ABP_pephir.npy")
y = np.load("op_PCF_ABP_pephir.npy")

save_dir = "my_saved_pysr_new"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "new_model_notheta_pysr_for_paper_ssh.pkl")

if os.path.exists(model_path):
    print(">> Found existing model, loading for warm start...")
    model = joblib.load(model_path)
    model.warm_start = True
else:
    print(">> Starting new PySR run...")
    model = PySRRegressor(
        niterations=10000,             # first stage
        populations=30,
        procs=4,
        parallelism="multithreading",
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["exp", "cos", "sqrt","log"],
        constraints={"^": (-1, 1)},
        nested_constraints={"exp": {"exp": 0}, "cos": {"cos": 0},  "log": {"log": 0},  "sqrt":{"sqrt":0}},
        early_stop_condition="stop_if(loss, complexity) = (loss < 5e-3) & (complexity < 45)",
        maxsize=50,
        maxdepth=10,
        model_selection="accuracy",
        parsimony=0.001075,
        denoise=False,
        verbosity=1,
        progress=False,
        random_state=43,
        tempdir=os.path.join(save_dir, "new_tmp_notheta_01"),
        warm_start=False,
    )

# --- Fit or continue fitting ---
print(f">> Running symbolic regression up to {model.niterations} iterations...")
model.fit(X, y, variable_names=["phi", "Pe", "r"])

# --- Save progress ---
joblib.dump(model, model_path)
model.equations_.to_csv(os.path.join(save_dir, "notheta_equations_latest.csv"))
print(">> Model and equations saved.")

# --- Auto-increase niterations for next run ---
next_iters = model.niterations + 10000
if next_iters <= 50000:      # stop threshold
    model.niterations = next_iters
    model.warm_start = True
    model.fit(X, y, variable_names=["phi", "Pe", "r"])
    joblib.dump(model, model_path)
    model.equations_.to_csv(os.path.join(save_dir, "notheta_equations_latest.csv"))
    print(f">> Prepared for next warm start: niterations={next_iters}")
else:
    print(">> Reached final iteration target.")
