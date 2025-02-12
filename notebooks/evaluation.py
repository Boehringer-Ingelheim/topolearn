import numpy as np
import pandas as pd
import math
import shap
import pickle
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error
from matplotlib import rc
from utils import filter_df, create_subplots
from plotting import load_plot_config


class Validation():
    def __init__(self, data, loo_group, feature_cols, target_col, samples, split, distance_type, representation, rep_type={"all": None}) -> None:
        self.filter = {"samples": samples,
                       "split": split,
                       "dataset": {"all": None},
                       "representation_type": rep_type,
                       "distance_metric": distance_type,
                       "representation": representation
                       }
        self.data = data
        self.cross_validation_results = []
        self.baselines = ["rogi_xd", "rmodi", "sari"]
        self.loo_group = loo_group
        self.feature_cols = feature_cols
        self.target_col = target_col

    def process_data(self, y_pred, y_test, X_train, subset, test_index, method,
                     plot_intermediate, plot_hue, model=None):

        if "Relative" in self.target_col:
            y_pred = np.clip(y_pred, a_min=0.0, a_max=1.0)
            y_test = np.clip(y_test, a_min=0.0, a_max=1.0)

        # Compute correlation
        pearson_res = sp.stats.pearsonr(y_pred, y_test)
        pearson = pearson_res.statistic
        pep = pearson_res.pvalue
        spearman_res = sp.stats.spearmanr(y_pred, y_test)
        spearman = spearman_res.statistic
        spp = spearman_res.pvalue
        mae = mean_absolute_error(y_pred, y_test)

        importances = None
        if model:
            # Feature importance
            forest_importances = pd.Series(
                model.feature_importances_, index=self.feature_cols)
            importances = pd.DataFrame(forest_importances, columns=[
                                       "importances"]).reset_index().sort_values(by="importances", ascending=False)

            # SHAP
            explainer = shap.TreeExplainer(
                model, feature_names=self.feature_cols)
            explanation = explainer(X_train)
            if plot_intermediate:
                shap.plots.beeswarm(explanation, s=10,
                                    color=plt.get_cmap('RdBu'))

        # Collect all data
        plot_data = pd.DataFrame({
            "pred": [y_pred],
            "test": [y_test.values],
            "representation": [subset.iloc[test_index]["representation"].values],
            "dataset": [subset.iloc[test_index]["dataset"].values],
            "representation_type": [subset.iloc[test_index]["representation_type"].values],
            "samples": [subset.iloc[test_index]["samples"].values],
            "spearman_corr": spearman,
            "spearman_p": spp,
            "pearson_corr": pearson,
            "pearson_p": pep,
            "mae": mae,
            "method": method,
        })

        if plot_intermediate:
            self.plot_predictions(plot_data, hue=plot_hue, custom_legend=True)
            plt.show()

        return plot_data, importances

    def run_LOGO_CV(self, plot_hue, plot_intermediate=True, print_best_params=False):
        subset = filter_df(self.data, self.filter).reset_index()
        subset["group_id"] = subset.groupby(self.loo_group).ngroup()
        logo = LeaveOneGroupOut()
        X, y = subset[self.feature_cols], subset[self.target_col]
        for i, (train_index, test_index) in tqdm(enumerate(logo.split(X, y, subset['group_id'])), total=subset["group_id"].nunique()):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            print(
                f"Fold {i}, X train: {X_train.shape[0]}, X test: {X_test.shape[0]}")

            if X_test.shape[0] > 1:
                # Run training
                model, y_pred = self.run_cv(
                    X_train, y_train, X_test, print_best_params=print_best_params)
                topolearn_data, importances = self.process_data(y_pred, y_test, X_train, subset, test_index, "TopoLearn",
                                                                plot_intermediate, plot_hue, model)

                # Compute baseline predictions
                baselines_data = []
                for baseline in self.baselines:
                    b_pred = subset.iloc[test_index][baseline].values
                    baselines_data.append(self.process_data(b_pred, y_test, X_train, subset, test_index,
                                                            baseline.upper().replace("_", "-"),
                                                            plot_intermediate, plot_hue))

                results = {
                    "topolearn_data": topolearn_data,
                    "baselines_data": baselines_data,
                    "importances": importances,
                    "model": model,
                    "X_train": X_train
                }
                self.cross_validation_results.append(results)

    def run_cv(self, X_train, y_train, X_test, print_best_params):
        # Hyperparameter search
        model = RandomForestRegressor()
        parameter_grid = {'n_estimators': [100, 200, 300],
                          'max_depth': [5, 10, 20, 50],
                          'min_samples_leaf': [1, 5, 10]}

        search = GridSearchCV(estimator=model, param_grid=parameter_grid,
                              scoring="neg_mean_absolute_error",
                              n_jobs=-1, verbose=0, error_score="raise")
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)

        if print_best_params:
            print("Parameters: ", search.best_params_)

        return search.best_estimator_, y_pred

    def plot_predictions(self, row, hue=None, ax=None, legend=True, custom_legend=False,
                         palette=None, size_div=15, alpha=0.7, markeredgecolor="black", markeredgewidth=1, regplot=True):
        load_plot_config()
        row["samples"] = pd.Series(row["samples"])
        sizes = row["samples"].apply(lambda x: float(x) / size_div)
        plot = sns.scatterplot(y=row["pred"], x=row["test"], hue=row[hue],
                               palette=palette,  linewidth=markeredgewidth, alpha=alpha,
                               edgecolors=markeredgecolor, s=sizes.values, legend=legend, ax=ax)
        if isinstance(row["spearman_corr"], float):
            corr = row["spearman_corr"]
            p = row["spearman_p"]
            plot.text(0.05, 0.95, f"R = {corr:.2f} \np < {(math.ceil(p * 1000) / 1000):.3f}", ha="left", va="top",
                      weight='bold', fontsize=9, transform=plot.transAxes,
                      bbox=dict(facecolor='lightgrey', alpha=0.7, edgecolor='lightgrey'))

        # Add regression line
        if regplot:
            sns.regplot(y=row["pred"], x=row["test"], ci=None, scatter=False, line_kws=dict(
                color="grey", linestyle='dashed'), ax=ax)

        # Adjust legend (remove sizes labels)
        if custom_legend:
            h, l = plot.get_legend_handles_labels()
            h = [i for i, j in zip(h, l) if not j.isnumeric()]
            l = [i for i in l if not i.isnumeric()]
            cols = 2 if len(h) > 10 else 1
            plot.legend(h, l, loc="upper left", bbox_to_anchor=(
                1, 1), title=hue.capitalize(), ncol=cols)
            plt.setp(plot.get_legend().get_texts(), fontsize='9')
            plt.setp(plot.get_legend().get_title(), fontsize='10')

        title = ""
        if not isinstance(row["method"], list):
            for element in self.loo_group:
                np_group = np.array(row[element])
                assert (np_group[0] == np_group).all(0)
                title += f"{np_group[0]}"
            method = row['method']
        else:
            method = row['method'][0]
        plot.set(xlabel=self.target_col, ylabel=method)
        plot.set_title(title, fontsize=12, fontweight='bold')

        sns.despine(offset=5)

    def plot_predictions_grid(self, method="TopoLearn", hue="dataset", palette=None, n_cols=4, save_file=None,
                              alpha=0.7, markeredgecolor="black", markeredgewidth=1, size_div=15, bbox_to_anchor=(1, 0.3),
                              figsize=(10, 3)):
        plot_data_list = []
        for data in self.cross_validation_results:
            if method == "TopoLearn":
                plot_data_list.append(data["topolearn_data"])
            elif method == "ROGI-XD":
                plot_data_list.append(data["baselines_data"][0][0])
            elif method == "RMODI":
                plot_data_list.append(data["baselines_data"][1][0])
            elif method == "SARI":
                plot_data_list.append(data["baselines_data"][2][0])

        plot_data = pd.concat(plot_data_list)
        create_subplots(plot_data,
                        self.plot_predictions,
                        n_cols=n_cols,
                        figsize=figsize,
                        sharex=False, sharey=False,
                        ncols_legend=1,
                        save_file=save_file,
                        bbox_to_anchor=bbox_to_anchor,
                        hue_title=hue.capitalize(),
                        marker_legend=True,
                        grid=True,
                        kwargs={"hue": hue, "palette": palette, "alpha": alpha,
                                "markeredgecolor": markeredgecolor, "markeredgewidth": markeredgewidth,
                                "size_div": size_div})

    def plot_summary(self, methods=["TopoLearn", "ROGI-XD", "RMODI", "SARI"], hue="dataset", palette=None,
                     alpha=0.7, size_div=50, figsize=(10, 3), ncols=2, save_file=None, markeredgecolor="black",
                     markeredgewidth=1, bbox_to_anchor=(1, 0.3)):
        plot_data = []
        for method in methods:
            method_data = []
            for data in self.cross_validation_results:
                if method == "TopoLearn":
                    results_data = data["topolearn_data"].copy()
                elif method == "ROGI-XD":
                    results_data = data["baselines_data"][0][0].copy()
                elif method == "RMODI":
                    results_data = data["baselines_data"][1][0].copy()
                elif method == "SARI":
                    results_data = data["baselines_data"][2][0].copy()

                # Padding
                list_length = len(results_data['pred'][0])
                for col in results_data.columns:
                    if isinstance(results_data[col][0], np.ndarray):
                        results_data[col][0] = results_data[col][0].tolist()
                    if not isinstance(results_data[col][0], list):
                        results_data[col] = [
                            [results_data[col][0]] * list_length]
                method_data.append(results_data)

            # Concatenate lists using sum and convert back to single-row DF
            plot_data.append(pd.DataFrame(
                pd.concat(method_data).sum()).transpose())

        plot_data = pd.concat(plot_data).reset_index(drop=True)
        create_subplots(plot_data,
                        self.plot_predictions,
                        n_cols=len(methods),
                        col_labels=methods,
                        figsize=figsize,
                        sharex=False, sharey=False,
                        ncols_legend=ncols,
                        save_file=save_file,
                        marker_legend=True,
                        grid=True,
                        bbox_to_anchor=bbox_to_anchor,
                        hue_title=hue.capitalize(),
                        kwargs={"hue": hue, "size_div": size_div, "palette": palette, "alpha": alpha,
                                "markeredgecolor": markeredgecolor, "markeredgewidth": markeredgewidth,
                                "regplot": False})

    def plot_shap_swarm(self, row, ax, legend, col_name_mapping, color_bar_label):
        load_plot_config()
        explainer = shap.TreeExplainer(
            row["model"], feature_names=self.feature_cols)
        explanation = explainer(row["X_train"])
        plt.sca(ax)

        shap.plots.beeswarm(explanation, max_display=6, s=20, color=plt.get_cmap('RdBu'),
                            axis_color='#333333', alpha=0.8, group_remaining_features=False,
                            show=False, color_bar=legend, color_bar_label=color_bar_label)

        labels = [t.get_text() for t in ax.get_yticklabels()]
        ax.set_yticklabels(map(
            lambda yy: col_name_mapping[yy] if yy in col_name_mapping else yy, labels), fontsize='small')
        ax.set_xlabel("SHAP value", fontsize=10)
        ax.set_title(row["dataset"], fontweight="bold", fontsize=12)

    def plot_shap_grid(self, figsize=(10, 3), save_file=None, n_cols=3, bbox_inches="tight", col_name_mapping=None,
                       color_bar_label="Feature value"):
        load_plot_config()

        payload = {
            "model": [],
            "X_train": [],
            "dataset": [],
        }
        for result in self.cross_validation_results:
            payload["model"].append(result["model"])
            payload["X_train"].append(result["X_train"])
            payload["dataset"].append(
                result["topolearn_data"]["dataset"][0][0])

        plot_data = pd.DataFrame().from_dict(payload)
        create_subplots(plot_data,
                        self.plot_shap_swarm,
                        n_cols=n_cols,
                        figsize=figsize,
                        sharex=False, sharey=False,
                        ncols_legend=1,
                        grid=True,
                        bbox_inches=bbox_inches,
                        save_file=save_file,
                        kwargs={"col_name_mapping": col_name_mapping, "color_bar_label": color_bar_label})

    def plot_feature_importances(self, hue, top_n=10, palette=None, save_file=None, alpha=0.7, s=10,
                                 bar_fill=False, figsize=(10, 6), bar_color="blue", bar_line_color="black",
                                 bar_linewidth=0.75, col_name_mapping=None):
        load_plot_config()
        all_importances = []
        for r in self.cross_validation_results:
            imp_df = r["importances"]
            imp_df[hue] = r["topolearn_data"][hue].iloc[0][0]
            all_importances.append(imp_df.head(top_n))

        importances = pd.concat(all_importances)
        importances = importances.sort_values(
            by="importances", ascending=False)
        p = sns.stripplot(x='importances', y='index', hue=hue,
                          data=importances, alpha=alpha, s=s, palette=palette)
        sns.boxplot(x="importances", y="index", data=importances, fill=bar_fill, color=bar_color,
                    linecolor=bar_line_color, linewidth=bar_linewidth, showfliers=False)
        sns.move_legend(p, "upper left", bbox_to_anchor=(1, 1))
        plt.gcf().set_size_inches(figsize)
        p.set(xlabel="Importance", ylabel="Features")
        labels = [t.get_text() for t in p.get_yticklabels()]
        p.set_yticklabels(
            map(lambda yy: col_name_mapping[yy] if yy in col_name_mapping else yy, labels))

        if save_file:
            p.figure.savefig(f'plots/{save_file}.png',
                             dpi=300, bbox_inches='tight')
        plt.show()

    def evaluation_table(self, filename, methods=["TopoLearn", "ROGI-XD", "RMODI", "SARI"]):
        summary = {}
        for method in methods:
            all_spearmans = []
            all_spearman_ps = []
            all_pearsons = []
            all_pearson_ps = []
            all_maes = []
            all_preds = []
            all_tests = []
            for result in self.cross_validation_results:
                if method == "TopoLearn":
                    results_data = result["topolearn_data"].copy()
                elif method == "ROGI-XD":
                    results_data = result["baselines_data"][0][0].copy()
                elif method == "RMODI":
                    results_data = result["baselines_data"][1][0].copy()
                elif method == "SARI":
                    results_data = result["baselines_data"][2][0].copy()

                # Extract data
                all_spearmans.append(results_data["spearman_corr"].iloc[0])
                all_spearman_ps.append(results_data["spearman_p"].iloc[0])
                all_pearsons.append(results_data["pearson_corr"].iloc[0])
                all_pearson_ps.append(results_data["pearson_p"].iloc[0])
                all_maes.append(results_data["mae"].iloc[0])
                all_preds.append(results_data["pred"].iloc[0])
                all_tests.append(results_data["test"].iloc[0])

            table_data = {}
            # Individual data per CV fold
            table_data["spearman_ps"] = np.array(all_spearman_ps)
            table_data["pearson_ps"] = np.array(all_pearson_ps)
            table_data["spearman_mean"] = np.array(all_spearmans).mean()
            table_data["spearman_std"] = np.array(all_spearmans).std()
            table_data["pearson_mean"] = np.array(all_pearsons).mean()
            table_data["pearson_std"] = np.array(all_pearsons).std()
            table_data["mae_mean"] = np.array(all_maes).mean()
            table_data["mae_std"] = np.array(all_maes).std()

            # Aggregate data across all datasets
            all_pred = np.concatenate(all_preds)
            all_test = np.concatenate(all_tests)
            spearman = sp.stats.spearmanr(all_pred, all_test)
            pearson = sp.stats.pearsonr(all_pred, all_test)
            table_data["spearman_all"] = spearman.statistic
            table_data["spearman_p_all"] = spearman.pvalue
            table_data["pearson_all"] = pearson.statistic
            table_data["pearson_p_all"] = pearson.pvalue
            table_data["mae_all"] = mean_absolute_error(all_pred, all_test)

            # Formatted table
            table_data["Spearman"] = f'{table_data["spearman_mean"]:.2f}\u00B1{table_data["spearman_std"]:.2f}'
            table_data["Pearson"] = f'{table_data["pearson_mean"]:.2f}\u00B1{table_data["pearson_std"]:.2f}'
            table_data["MAE"] = f'{table_data["mae_mean"]:.2f}\u00B1{table_data["mae_std"]:.2f}'
            summary[method] = table_data

        self.summary = summary
        transposed = pd.DataFrame(summary).transpose()[
            ["Spearman", "Pearson", "MAE", "spearman_all", "spearman_p_all", "pearson_all", "pearson_p_all", "mae_all"]]
        transposed.to_csv(filename, sep="\t")
        return transposed

    def export_models(self):
        for i, result in enumerate(self.cross_validation_results):
            with open(f'models/model_{i}.pkl', 'wb') as f:
                pickle.dump(result["model"], f)
