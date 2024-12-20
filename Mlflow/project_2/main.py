import sys
import mlflow
from steps.download_data import load_raw_data
from steps.preprocess_data import preprocess_data
from steps.tune_model import tune_model
from steps.train_final_model import train_model
import argparse
import bcolors

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=str)
    parser.add_argument("n_trials", type=int)
    return parser.parse_args()

def main():
    args = parse_arguments()
    mlflow.set_experiment("fraud")
    file_location = load_raw_data(args.data_file)
    print(f"{bcolors.OKCYAN}Data is loaded{bcolors.ENDC}")
    file_dirs = preprocess_data(file_location, missing_thr=0.95)
    print(f"{bcolors.OKCYAN}Data is preprocessed{bcolors.ENDC}")
    best_params = tune_model(
        train_path=file_dirs["train-data-dir"],
        val_path=file_dirs["val-data-dir"],
        n_trials=args.n_trials,
    )
    print(f"{bcolors.OKCYAN}HP tuning is finished{bcolors.ENDC}")
    best_params["n_estimators"] = 1000
    best_params["objective"] = "Logloss"
    roc, pr = train_model(
        best_params,
        train_path=file_dirs["train-data-dir"],
        val_path=file_dirs["val-data-dir"],
        test_path=file_dirs["test-data-dir"],
    )
    print(f"{bcolors.OKGREEN}Final model is trained. \nTestset ROC AUC: {roc}\nTestset PR AUC: {pr}{bcolors.ENDC}")

if __name__ == "__main__":
    main()