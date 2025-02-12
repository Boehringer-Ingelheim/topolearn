import time
import umap
import umap.plot
import networkx as nx
import numpy as np

from functools import wraps
from scipy import stats
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline


def scaffold_split(df, n_splits):
    df['MurckoScaffold'] = df.apply(lambda x: MurckoScaffoldSmiles(x["smiles"],
                                                                   includeChirality=True), axis=1)
    gkf = GroupKFold(n_splits=n_splits)
    split_generator = gkf.split(X=df["smiles"], groups=df["MurckoScaffold"])
    splits = [next(iter(split_generator)) for i in range(n_splits)]
    return splits


def random_split(df, test_size=0.2, seed=0):
    indices = np.arange(df.shape[0])

    _, _, _, _, train_index, test_index = train_test_split(df["smiles"], df["target"], indices, test_size=test_size, random_state=seed)

    return [[train_index, test_index]]


def standard_scale(col):
    scaler = StandardScaler()
    return scaler.fit_transform(col)


def min_max_scale(arr):
    transform_pipe = Pipeline([('robust_scaler', RobustScaler()),   # First apply robust scaler to avoid that the distribution gets messed up
                                 #('gauss_transform', PowerTransformer()), # Make sure the data follows the same distribution
                                 #   ('minmax_scaler', MinMaxScaler())
                                    ]) # Scale between 0 and 1 to simplify AC threshold
    return transform_pipe.fit_transform(arr), transform_pipe

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.2f} seconds.')
        return result
    return timeit_wrapper


def manifold_density(X):
    mapper = umap.UMAP()
    _ = mapper.fit_transform(X)

    # Weighted matrix
    adj = nx.adjacency_matrix(nx.from_scipy_sparse_array(mapper.graph_)).toarray()
    weights = adj[adj!=0].flatten()
    desc = stats.describe(weights)
    return desc[2:] # mean, variance, skewness, kurtosis

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))