import math
import logging
import itertools
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval
from persim import plot_diagrams
from matplotlib import rc
from matplotlib.pyplot import gcf
from matplotlib.lines import Line2D
from tqdm import tqdm

logger = logging.getLogger(__name__)
rc('text', usetex=False) 

def create_subplots(data, plt_func, n_cols=4, row_labels=[], col_labels=[], title=None, 
                    figsize=(12, 8), x_range=[], y_range=[], kwargs={}, hue_title="",
                    legend_text_fs = '8', legend_title_fs = '11', sharex=False, sharey=False, ncols_legend=2,
                    save_file=None, marker_legend=False, bbox_to_anchor=None, bbox_inches="tight", grid=False):
    data = data.reset_index()
    n_plots = data.shape[0]
    n_rows = math.ceil(n_plots / n_cols)
    figsize = tuple([figsize[0], figsize[1] * n_rows])
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, constrained_layout=True, sharex=sharex, sharey=sharey) 
    

    # Correct legend positioning
    if not bbox_to_anchor:
        bbox_to_anchor = (1, n_rows*0.1)

    all_sizes = np.array([])    
    for row, axis in tqdm(itertools.zip_longest(data.iterrows(), ax.ravel()), total=len(ax.ravel())):
        if row is None:
            fig.delaxes(axis)
            continue
        idx, row = row
        legend = False
        if ((idx + 1) % n_cols == 0 and ((idx + 1) / n_cols) == n_rows) or (idx + 1 == data.shape[0]):
            # For now disabled (no legend on last plot)
            if not "palette" in kwargs:
                legend = True
            

        # Plot subfigure
        plt_func(row, ax=axis, legend=legend, **kwargs)     

        if grid:
            axis.set_axisbelow(True)
            axis.yaxis.grid(color='lightgray')
            axis.xaxis.grid(color='lightgray')

        # Axes range
        if len(x_range) > 0:
            axis.set_xlim([x_range[0], x_range[1]])
        if len(y_range) > 0:
            axis.set_ylim([y_range[0], y_range[1]])

        # Row labels
        if len(row_labels) > 0 and (idx % n_cols == 0): 
            id = idx // n_cols         
            axis.set_ylabel(row_labels[id], rotation=90, size='large')

        # Store marker sizes
        if marker_legend:
            all_sizes = np.concatenate([all_sizes, axis.collections[0].get_sizes()])

    # --- Post processing ---
    
    # Set column labels
    if len(col_labels) > 0:
        if n_rows > 1:
            for ax, col in zip(ax[0, :], col_labels):
                ax.set_title(col, fontweight='bold', fontsize="12")
        else:
            for ax, col in zip(ax, col_labels):
                ax.set_title(col, fontweight='bold', fontsize="12")
    
    # Joint legend
    if "palette" in kwargs:
        legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, 
                                 alpha=kwargs["alpha"], markeredgecolor=kwargs["markeredgecolor"],
                                 markeredgewidth=kwargs["markeredgewidth"]) 
                          for color in  kwargs["palette"].values()]  
        legend = fig.legend(legend_handles, kwargs["palette"].keys(), loc='lower left', 
                   bbox_to_anchor=bbox_to_anchor, title=hue_title, ncols=ncols_legend)     
        plt.setp(legend.get_texts(), fontsize=legend_text_fs)
        plt.setp(legend.get_title(), fontsize=legend_title_fs)

    # Hardcoded sizes!
    if marker_legend:
        # Double check
        uniques = np.unique([round(n / 10) * 10 for n in np.unique(all_sizes)])
        size_handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(size)*1.2, label=str(500*(i+1))) 
            for i, size in enumerate(uniques)
        ]
        legend2 = fig.legend(handles=size_handles, title="Sample size", loc='upper left', 
                   bbox_to_anchor=bbox_to_anchor,  ncols=1)
        plt.setp(legend2.get_texts(), fontsize=legend_text_fs)
        plt.setp(legend2.get_title(), fontsize=legend_title_fs)

    fig.suptitle(title, fontsize=12, y=-0.01)

    # Some (external) plots overwrite figsize
    plt.gcf().set_size_inches(figsize)
    plt.show()

    if save_file:
        plt.tight_layout()
        fig.figure.savefig(f'plots/{save_file}.png', dpi=300, bbox_inches=bbox_inches)

    return fig

def plot_phdim_fractals(row, ax, dim, **kwargs):
    sns.set_theme(style='white')
    logspace = row[f"ph_dim_logspace{dim}"]
    logedges = row[f"ph_dim_logedges{dim}"]
    ax.yaxis.set_tick_params(labelsize = 10)
    ax.xaxis.set_tick_params(labelsize = 10)
    ax.set_ylabel('log(L)')
    ax.set_xlabel('log(n)')
    name = r"$dim_0^{PH}$" if dim == 0 else r"$dim_1^{PH}$"
    ax.text(0.05, 0.95,  name+ "={0:.2g}".format(row[f"ph_dim_{dim}"]), ha="left", va="top", fontsize=12, transform=ax.transAxes)
    sns.regplot(x=logspace, y=logedges, ax=ax, marker='o')
    sns.despine(offset=5)


def plot_ph_interval_hist(row, ax, legend):
    sns.set_theme(style='white')
    intervals = row["intervals_0"]
    ax.yaxis.set_tick_params(labelsize = 10)
    ax.xaxis.set_tick_params(labelsize = 10)
    ax.text(0.5, 0.95, r"$\bar{L}_0$" + "={0:.2g}".format(row["lifetimes_mean_0"]), ha="left", va="top", fontsize=10, transform=ax.transAxes)
    sns.histplot(intervals, ax=ax, stat="frequency")
    ax.set_ylabel('')
    ax.set_xlabel('')
    sns.despine(offset=5)


def plot_ph_diagrams(row, ax, legend, palette=None, alpha=1, markeredgecolor="k", markeredgewidth=1.0, colormap=None, size=20):
    # Filter out diagrams with no persistence
    plot_diagrams([d for d in row["dgms"] if d.shape[0] > 0], legend=legend, ax=ax, colormap=colormap, 
                  edgecolor=markeredgecolor, alpha=alpha, linewidths=markeredgewidth, size=size)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.text(0.1, 0.95, r'$\bar{L}_{0}$' + "={:.2g}".format(row["lifetimes_mean_0"]), ha="left", va="top", fontsize=10, transform=ax.transAxes)
    sns.despine(offset=5)


def plot_phd_corr(row, ax, palette, legend=False, stat=None, err=None, rep=None, alpha=None, markeredgecolor="k", markeredgewidth=1.0):
    def annotate(ax, **kws):
        s = np.nan_to_num(row[stat], neginf=0)  
        e = np.nan_to_num(row[err], neginf=0)
        r, p = sp.stats.pearsonr(s, e)
        ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
                transform=ax.transAxes)
    
    fig = gcf()
    fig.set_size_inches(20, 4, forward=True)
    sns.set_theme(style='white')
    x = row[stat]
    y = row[err]

    sns.regplot(x=x, y=y, ax=ax, scatter=False, color=".3", ci=None, line_kws=dict(color="grey", linestyle="--")) # ci=None
    sax = sns.scatterplot(x=x, y=y, hue=row[rep], ax=ax, s=120, alpha=alpha, palette=sns.color_palette(palette.values()))
    annotate(sax)
    try:
        sns.move_legend(sax, "upper left", bbox_to_anchor=(1, 1))
    except:
        pass
    ax.yaxis.set_tick_params(labelsize = 10)
    ax.xaxis.set_tick_params(labelsize = 10)

    if legend == False: 
        ax.legend([],[], frameon=False)


def plot_phd_samples(row, ax, legend=False, stat=None, metric=None, rep=None):
    plot = sns.lineplot(x=row[stat], y=row[metric], hue=row[rep], ax=ax)
    if legend:
        sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
    else:
        ax.legend([],[], frameon=False)


def filter_df(df, filter_dict):
    subset = df.copy()
    for key, value in filter_dict.items():
        if isinstance(value, dict):
            operator = list(value.keys())[0]
            if operator == "gte":
                subset = subset[subset[key] >= value[operator]]
            elif operator == "lte":
                subset = subset[subset[key] <= value[operator]]
            elif operator == "not":
                subset = subset[~subset[key].isin(value[operator])]
            elif operator == "all":
                subset = subset
        elif isinstance(value, list):
            subset = subset[subset[key].isin(value)]
        else:
            subset = subset[subset[key] == value]
    return subset


def convert_lists_to_arr(col, ragged=False):
    if ragged: 
        return col.apply(lambda x: [np.array(arr) for arr in literal_eval(x)])
    else:
        return col.apply(lambda x: np.array(literal_eval(x.replace("-inf,", "").replace("-inf", ""))))
    
def interval_convert(row, idx=0):
    try:
        interval = row[idx][:, 1] - row[idx][:, 0]
    except: 
        # Sometimes no features were recorded
        interval = []
    return interval


def compute_2_3_ratio(arr):
    try:
        if len(arr) <= 1:
            return 0
        arr = np.array(arr)
        m = arr.max()
        max2 = np.partition(arr.flatten(), -2)[-2]
    except:
        print(arr)
        raise
    return max2/m

def compute_frac(arr, f=0.05):
    if len(arr) <= 1:
        return 0
    arr = np.array(arr)
    m = arr.max()
    n_longer = len(arr[arr > f*m])
    return (len(arr) - n_longer + 1) / n_longer



