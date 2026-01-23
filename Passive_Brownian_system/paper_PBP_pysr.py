from pysr import PySRRegressor
import numpy as np
import os
import joblib
import sympy
import pickle
# --- Reproducibility setup ---
np.random.seed(2002)
#os.environ["JULIA_NUM_THREADS"] = str(os.cpu_count())  
os.environ["JULIA_NUM_THREADS"] = "1"
X = np.load("PBP_inp.npy")   ##Pe/100,phi, cos_theta
Y = np.load("PBP_op.npy")
save_dir = "my_saved_pysr_new"
os.makedirs(save_dir, exist_ok=True)  #my_saved_models_new
model_path = os.path.join(save_dir, "new_1_pysr_model_PBP.pkl")

default_pysr_params = dict(
    model_selection="accuracy",#"best",accuracy
)

if os.path.exists(model_path):
    print(">> Found existing model, loading for warm start...")
    model = joblib.load(model_path)
    model.warm_start = True
else:
    print(">> Starting new PySR run...")
    model = PySRRegressor(
                    verbosity=1,                    
                    niterations=20000,  
                    populations=20,
                    procs=1,
                    random_state=44, 
                    parallelism="serial", #"multithreading", 
                    deterministic=True, 
                    early_stop_condition=(
                        "stop_if(loss, complexity) = loss < 1e-4 " # Stop early if we find a good and simple equation
                    ), 
                    parsimony=0.001075,
                    maxsize=50,
                    # ^ Allow greater complexity.
                    maxdepth = 8, 
                    binary_operators=["+", "-", "*", "/","^"],
                    constraints={"^": (-1, 1)},  
                    unary_operators=["exp", "sin", "cos"],
                    nested_constraints={ "exp": {"exp": 0}, "cos": {"cos": 0}, "sin": {"sin": 0}},
                    denoise = False,
                    warm_start = False,
                    tempdir=os.path.join(save_dir, "new1_tmp_PBP"),
                    )
print("Started fitting model...")
print(f">> Running symbolic regression up to {model.niterations} iterations...")
model.fit(X, Y, variable_names=["r", "phi"])
print("Model fitting done!")

joblib.dump(model, model_path)
model.equations_.to_csv(os.path.join(save_dir, "new_1_PBP_from_NN.csv"))
print(">> Model and equations saved.")

# --- Auto-increase niterations for next run ---
next_iters = model.niterations + 20000
if next_iters <= 100000:      # stop threshold
    model.niterations = next_iters
    model.warm_start = True
    joblib.dump(model, model_path)
    print(f">> Prepared for next warm start: niterations={next_iters}")
else:
    print(">> Reached final iteration target.")
