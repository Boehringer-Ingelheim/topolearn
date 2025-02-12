import pandas as pd
import numpy as np
import logging
import itertools
import random
import argparse
from ripser import ripser
from tqdm import tqdm
from pqdm.threads import pqdm
from multiprocessing import set_start_method
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, ParameterGrid

from topolearn.train_utils import scaffold_split, random_split, flatten_list, chunker
from topolearn.datasets import *
from topolearn.scores import *
from topolearn.config import get_configs
set_start_method("spawn", force=True)


# Dataset args
parser = argparse.ArgumentParser(
    prog='TopoLearn Analysis',
    description='Runs the training and topological computations.')
parser.add_argument('-d', '--dataset')
args = parser.parse_args()

if args.dataset:
    dataset_list = [args.dataset]
else:
    print("No dataset passed, processing all datasets.")
    dataset_list = ["ADRA1A", "ALOX5AP", "ATR", "JAK1",
                    "JAK2", "MUSC1", "MUSC2", "KOR",
                    "LIPO", "HLMC", "SOL", "DPP4"]

# Configs
config, global_params = get_configs(dataset_list)
split = global_params["split_type"]
scale = global_params["scale_y"]


def prepare_dataset(dataset, representation, n_samples):
    """
    Loads the features, filters nans and performs test-train splitting.
    """
    # Load processed dataset
    data = pd.read_pickle(
        f"data/features/{dataset}/{dataset}_{representation}.pkl")
    if data.shape[0] < n_samples:
        return None

    # Sample dataset
    data = data.sample(n=n_samples, random_state=2024).reset_index()
    data = data[~data["smiles"].isna()].reset_index()

    # Split data
    if global_params["split_type"] == "random":
        splits = random_split(data, test_size=0.2)
    else:
        splits = scaffold_split(data, n_splits=global_params["n_splits"])

    # Apply splitting
    train_mask = splits[0][0]
    test_mask = splits[0][1]
    data_train = data.loc[train_mask]
    data_test = data.loc[test_mask]

    # Get data
    feature_cols = [col for col in data.columns if col not in [
        "smiles", "target", "index", "MurckoScaffold"]]
    X = data[feature_cols].values
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = data_train["target"].values
    y_test = data_test["target"].values

    return data, X, X_train, y_train, X_test, y_test, train_mask, test_mask


def train_cv_model(model, X_train, y_train, X_test, y_test):
    """
    Performs a grid search cross-validation using as many cores as available 
    and trains a model with the best parameters for multiple seeds provided 
    in global_params.
    """
    # Run Grid Search
    m = model["model"]()
    n_fits = len(ParameterGrid(model["param_grid"]))
    search = GridSearchCV(estimator=m, param_grid=model["param_grid"],
                          scoring="neg_mean_absolute_error",
                          n_jobs=-1, verbose=2,
                          error_score="raise")
    search.fit(X_train, y_train)

    # Train with random seeds and best params
    eval_errs = {}
    eval_errs["best_cv_score"] = search.best_score_
    eval_errs["best_params"] = search.best_params_
    for seed in global_params["seeds"]:
        np.random.seed(seed)
        random.seed(seed)
        params = search.best_params_
        params["random_state"] = seed
        m = model["model"](**params)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        eval_metrics = global_params["eval_metrics"]
        for eval_func in eval_metrics.keys():
            eval_errs = eval_errs | {
                f"{eval_func}_{seed}": eval_metrics[eval_func](y_test, y_pred)}
    return eval_errs, n_fits + len(global_params["seeds"]), y_pred


def compute_persistent_homology_summary(X, sim_metric, suffix=""):
    """
    Computes various PH statistics and descriptors
    """
    summary = {}
    dgms = ripser.ripser(X, metric=sim_metric)['dgms']
    dgms = [dgm[dgm[:, 1] < np.inf] for dgm in dgms]
    summary[f"dgms{suffix}"] = [dgm.tolist() for dgm in dgms]
    summary[f"b_0{suffix}"] = compute_betti(dgms[0])
    summary[f"b_1{suffix}"] = compute_betti(dgms[1])
    summary[f"b_0_norm{suffix}"] = compute_betti_norm(X, dgms[0])
    summary[f"b_1_norm{suffix}"] = compute_betti_norm(X, dgms[1])
    if len(dgms[0]) > 0:
        ph_entr_0 = compute_persistence_entropy(dgms, idx=0)[0][0]
    else:
        ph_entr_0 = np.nan
    if len(dgms[1]) > 0:
        ph_entr_1 = compute_persistence_entropy(dgms, idx=1)[0][0]
    else:
        ph_entr_1 = np.nan
    summary[f"ph_entr_0{suffix}"] = ph_entr_0
    summary[f"ph_entr_1{suffix}"] = ph_entr_1
    summary = summary | compute_lifetime_stats(dgms[0], h_dim=0, suffix=suffix)
    summary = summary | compute_lifetime_stats(dgms[1], h_dim=1, suffix=suffix)
    phdim_0, logspace, logedges = compute_persistent_homology_dim(
        X, h_dim=0, metric=sim_metric)
    summary[f"ph_dim_0{suffix}"] = phdim_0
    summary[f"ph_dim_logspace0{suffix}"] = logspace.tolist()
    summary[f"ph_dim_logedges0{suffix}"] = logedges.tolist()
    phdim_1, logspace1, logedges1 = compute_persistent_homology_dim(
        X, h_dim=1, metric=sim_metric)
    summary[f"ph_dim_1{suffix}"] = phdim_1
    summary[f"ph_dim_logspace1{suffix}"] = logspace1.tolist()
    summary[f"ph_dim_logedges1{suffix}"] = logedges1.tolist()
    return summary


def compute_scores(X, y, metric, suffix=""):
    """
    Computes different scores related to intrinsic dimensionality
    and QSAR.
    """
    summary = {}
    summary[f"sample_size{suffix}"] = compute_train_dataset_size(X)
    summary[f"sample_dim{suffix}"] = compute_dimensionality(X)
    summary[f"pca_dim{suffix}"] = compute_global_int_dim_pca(X)
    summary[f"twonn_dim{suffix}"] = compute_global_int_dim_twonn(X)
    summary[f"rogi{suffix}"] = compute_rogi(X, y, metric)
    summary[f"rogi_xd{suffix}"] = compute_rogi_xd(X, y, metric)
    summary[f"rmodi{suffix}"] = compute_rmodi(X, y, metric)
    summary[f"sari{suffix}"] = compute_sari(X, y, metric)
    return summary


def compute_results(args):
    """
    One single execution of training models, computing topological properties
    and creating a results file for the configuration provided as args.
    """
    representation, dataset, n_samples, model = args
    model_name = model["model"].__name__
    logging.info(
        10*"#" + f" Computing: {representation} | model: {model_name} | dataset: {dataset} | samplesize: {n_samples} " + 10*"#")

    # Load featurized data and split
    data_ret = prepare_dataset(dataset, representation, n_samples)
    if data_ret is None:
        return None
    else:
        data, X, X_train, y_train, X_test, y_test, train_mask, test_mask = data_ret

    druglikeness, mw = compute_dataset_descriptors(data["smiles"])

    # Scale targets to [0, 1]
    if scale:
        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).squeeze()
        y_test = scaler.transform(y_test.reshape(-1, 1)).squeeze()

    # Compute evaluation metrics and run grid search
    model_errors, n_fits, y_pred = train_cv_model(
        model, X_train, y_train, X_test, y_test)

    # Store test predictions for later analysis
    if global_params["store_preds"]:
        csv_name = f"data/predictions/{representation}_{model_name}_{dataset}_{n_samples}_{scale}_{split}_test_preds.csv"
        dict_data = {"pred": y_pred,
                     "true": y_test,
                     "smiles": data.loc[test_mask]["smiles"]}
        pd.DataFrame().from_dict(dict_data).to_csv(csv_name)

    # Scores / Descriptors are computed on the whole dataset
    metrics = global_params["sim_metrics"]
    results = []
    for metric in metrics:
        # Compute all scores on all data (not just train/test)
        ph_summary_all = compute_persistent_homology_summary(
            X, metric, suffix="")
        ph_summary_train = compute_persistent_homology_summary(
            X_train, metric, suffix="_train")
        scores_all = compute_scores(X, data["target"], metric, suffix="")
        scores_train = compute_scores(
            X_train, y_train, metric, suffix="_train")

        result = {
            "representation": representation,
            "model": model["model"].__name__,
            "dataset": dataset,
            "samples": n_samples,
            "qed": druglikeness,
            "mw": mw,
            "distance_metric": metric,
            "param_grid": model["param_grid"],
            "dimensionality": X_train.shape[1]
        }

        result = result | model_errors
        result = result | ph_summary_all
        result = result | ph_summary_train
        result = result | scores_all
        result = result | scores_train
        result["n_fits"] = n_fits
        results.append(result)

    # Save intermediate results
    f_name = f"topolearn/processed_data/intermediate/{representation}_{model_name}_{dataset}_{n_samples}_{scale}_{split}.csv"
    pd.DataFrame(results).to_csv(f_name)
    return results


def run_analysis(config):
    """
    Run end-to-end analysis computing topological properties,
    training models and saving the results as files.
    """
    configurations = [config["representations"],
                      config["datasets"],
                      config["n_samples"],
                      config["models"]]

    # Permutation of possible configurations
    product_params = list(itertools.product(*configurations))

    # Run analysis in parallel
    if not global_params["debug"]:
        # Run in chunks as pool seemed to freeze every now and then
        results = []
        for eval_group in chunker(product_params, 16):
            results += pqdm(eval_group, compute_results, n_jobs=16)
    else:
        results = []
        for params in tqdm(product_params):
            results.append(compute_results(params))

    # Flatten results
    results = [result for result in results if result is not None]
    results = flatten_list(results)
    results = [r for r in results if isinstance(r, dict)]
    comparison = pd.DataFrame(results)
    file_ext = args.dataset if args.dataset else ""
    comparison.to_csv(
        f"topolearn/processed_data/comparison_final_{split}_{scale}_{file_ext}.csv", index=False)
    logging.info("Done.")


run_analysis(config)
