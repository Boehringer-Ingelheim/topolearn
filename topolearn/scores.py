import sys
import ripser
import skdim
import logging
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from IsoScore import IsoScore
from gtda.diagrams import PersistenceEntropy
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.covariance import ShrunkCovariance
from rogi import RoughnessIndex, RMODI, SARI

from topolearn.train_utils import timeit

sys.path.append('topolearn/external')
from topolearn.external.istar import istar
from topolearn.external.rogi_xd.rogi import rogi as rogixd

logging.basicConfig(level = logging.INFO)

# ----------------------- GENERAL SUMMARY STATS -------------------------------

@timeit
def compute_train_dataset_size(X):
    return X.shape[0]

@timeit
def compute_dimensionality(X):
    return X.shape[1]

@timeit
def compute_outlier_dimension_ratio(X):
    """
    According to Rudman et al. https://arxiv.org/abs/2310.17715.
    Outlier dimensions = dimensions in representations whose variance 
    is 5x larger than overall variance.
    """
    logging.info('Computing Outlier Dimension ratio.')
    return sum(X.var(axis=0) > 5* X.var()) / X.shape[0]


# ----------------------- PERSISTENT HOMOLOGY -------------------------------
@timeit
def compute_betti(dgm):
    """
    @param dgm: A persistence diagram as birth/death features.
    """
    logging.info('Computing betti number for specified homology dimension.')
    b = dgm.shape[0]
    return b

@timeit
def compute_betti_norm(X, dgm):
    """
    @param X: The data used to get the persistence diagram.
    @param dgm: A persistence diagram as birth/death features.
    """
    logging.info('Computing normalized betti number for specified homology dimension.')
    b_norm = dgm.shape[0] / X.shape[0]
    return b_norm

@timeit
def compute_persistence_entropy(dgms, idx=0):
    """
    Only supported for maximum 0, 1 homology dimensions.

    @param dgms: The persistence diagrams for all homology dimensions 
                 having the shape (homology_dims, birth, death).
                 Will be adjusted to fit the function call for giotto-tda,
                 which is (homology_dims, birth, death, homology_dim)
    """
    logging.info('Computing persistence entropy.')
    # Align the format for ripser with giotto-tda
    q_pad = np.concatenate([dgms[idx], np.full((dgms[idx].shape[0], 1), idx)], axis=1)
    dgms_giotto = np.expand_dims(q_pad, 0)
    PE = PersistenceEntropy(n_jobs=-1)
    return PE.fit_transform(dgms_giotto)

@timeit
def compute_lifetime_stats(dgm, h_dim, suffix):
    """
    Computes several descriptive statistics for persistence diagram lifetimes
    using the aggregations specified in aggs applied on lifetimes and midlifes.

    @param dgm: Persistence diagram for a specific homology dimension.
    @param h_dim: Homological dimension
    @param suffix: Suffix added to descriptor names in dictionary
    
    """
    logging.info('Computing lifetime statistics for specified homology dimension.')

    # Normal persistence lifetimes
    dgm = dgm[~np.isinf(dgm).any(1)]
    descriptors = {}
    aggs = [np.min, np.max, np.mean, np.var, np.sum]
    
    # Lifetime descriptors
    lifetimes = dgm[:, 1] - dgm[:, 0]
    if lifetimes.shape[0] > 0:
        norm_lifetimes = lifetimes / lifetimes.sum()
        for agg in aggs:
            descriptors[f"lifetimes_{agg.__name__}_{h_dim}{suffix}"] = agg(lifetimes)
            descriptors[f"norm_lifetimes_{agg.__name__}_{h_dim}{suffix}"] = agg(norm_lifetimes)
        
    else:
        for agg in aggs:
            descriptors[f"lifetimes_{agg.__name__}_{h_dim}{suffix}"] = np.nan
            descriptors[f"norm_lifetimes_{agg.__name__}_{h_dim}{suffix}"] = np.nan

    # Midlife descriptors
    midlifes = (dgm[:, 1] + dgm[:, 0]) / 2
    if midlifes.shape[0] > 0:
        norm_midlifes = midlifes / midlifes.sum()
        for agg in aggs:
            descriptors[f"midlifes_{agg.__name__}_{h_dim}{suffix}"] = agg(midlifes)
            descriptors[f"norm_midlifes_{agg.__name__}_{h_dim}{suffix}"] = agg(norm_midlifes)
        
    else:
        for agg in aggs:
            descriptors[f"midlifes_{agg.__name__}_{h_dim}{suffix}"] = np.nan
            descriptors[f"norm_midlifes_{agg.__name__}_{h_dim}{suffix}"] = np.nan        
    return descriptors


@timeit
def compute_persistent_homology_dim(X, metric="euclidean", min_samples=100, max_samples=2000, stepsize=50, h_dim=0, alpha=1, seed=24):
    """
    Code is inspired by https://github.com/tolgabirdal/PHDimGeneralization
    and https://github.com/CSU-PHdimension/PHdimension/blob/master/code_ripser/SumEdgeLengths.m. 
    Based on https://arxiv.org/pdf/1808.01079.
    """
    logging.info(f'Computing persistent homology fractal dimension for homology dim {h_dim}.')
    # Clip max samples to dataset size
    max_samples = max_samples if X.shape[0] > max_samples else X.shape[0]

    if X.shape[0] < min_samples:
        logging.warn("Number of samples for PH Dim calculation too small.")
        return np.nan, np.array([]), np.array([]) 

    # Define sampling space
    linspace = range(min_samples, max_samples, stepsize)
    edge_lengths = []
    
    np.random.seed(seed)
    # Randomly sample increasing number of points and generate fractals
    for n in linspace:
        sampled_x = X[np.random.choice(X.shape[0], size=n, replace=False)]
        dgms = ripser.ripser(sampled_x, maxdim=h_dim, metric=metric)['dgms']

         # Append sum of weighted diagram lengths (birth-death)
        dgm = dgms[h_dim]
        dgm = dgm[dgm[:, 1] < np.inf]
        edge_lengths.append(np.power((dgm[:, 1] - dgm[:, 0]), alpha).sum())

    edge_lengths = np.array(edge_lengths)
    
    # Estimate slope using linear regression
    logspace = np.log10(np.array(list(linspace)))
    logedges = np.log10(edge_lengths)       
    coeff = np.polyfit(logspace, logedges, 1)
    m = coeff[0]
    b = coeff[1]

    return alpha / (1 - m), logspace, logedges

@timeit
def compute_persistent_homology_clustering_dim(X, 
                                               metric="euclidean", 
                                               min_samples=100, 
                                               max_samples=1000, 
                                               stepsize=30, 
                                               h_dim=0, 
                                               alpha=1, 
                                               seed=24):
    """
    Code is inspired by https://github.com/tolgabirdal/PHDimGeneralization
    and https://github.com/CSU-PHdimension/PHdimension/blob/master/code_ripser/SumEdgeLengths.m. 
    Based on https://arxiv.org/pdf/1808.01079.
    """
    logging.info(f'Computing persistent homology fractal dimension for homology dim {h_dim}.')
    # Clip max samples to dataset size
    max_samples = max_samples if X.shape[0] > max_samples else X.shape[0]

    if X.shape[0] < min_samples:
        logging.warn("Number of samples for PH Dim calculation too small.")

    # Define sampling space
    linspace = range(min_samples, max_samples, stepsize)
    edge_lengths = []
    
    np.random.seed(seed)
    # Randomly sample increasing number of points and generate fractals
    for n in linspace:
        sampled_x = X[np.random.choice(X.shape[0], size=n, replace=False)]
        dgms = ripser.ripser(sampled_x, maxdim=h_dim, metric=metric)['dgms']

         # Append sum of weighted diagram lengths (birth-death)
        dgm = dgms[h_dim]
        dgm = dgm[dgm[:, 1] < np.inf]
        edge_lengths.append(np.power((dgm[:, 1] - dgm[:, 0]), alpha).sum())

    edge_lengths = np.array(edge_lengths)
    
    # Estimate slope using linear regression
    logspace = np.log10(np.array(list(linspace)))
    logedges = np.log10(edge_lengths)
    coeff = np.polyfit(logspace, logedges, 1)
    m = coeff[0]
    b = coeff[1]

    return alpha / (1 - m), logspace, logedges

@timeit
def compute_persistent_homology_complexity(X, h_dim=0):
    """
    Based on MacPherson and Schweinhart: https://arxiv.org/pdf/1011.2258.
    Measures the complexity of the connectivity of the shape. 
    More explanations can be found here in 2.1.3: https://arxiv.org/pdf/1907.11182.
    It might serve as indicator how good the PH Dim estimate is.
    """
    logging.info(f'Computing persistent homology complexity for homology dim {h_dim}.')
    # Perform filtration
    dgms = ripser.ripser(X, maxdim=h_dim)['dgms']
    dgm = dgms[h_dim]
    dgm = dgm[dgm[:, 1] < np.inf]

    # Get interval lengths
    I = dgm[:, 1] - dgm[:, 0]

    # Get counts and epsilon
    eps_range = np.linspace(I.min(), I.max(), num = 100)
    f = [sum(I > eps) for eps in eps_range]

    # Log transform
    logf = np.log(f)
    logeps = np.log(eps_range)

    return logf, logeps


# ----------------------- INTRINSIC DIMENSIONALITY -------------------------------
@timeit
def compute_global_int_dim_pca(X):
    try:
        logging.info('Computing global intrinsic dimensionality.')
        pca = skdim.id.lPCA()
        intrdim = pca.fit(X).dimension_    
    except np.linalg.LinAlgError: 
        # SVD did not converge
        return np.nan
    return intrdim 

@timeit
def compute_global_int_dim_twonn(X):
    logging.info('Computing TwoNN intrinsic dimensionality.')
    tnn = skdim.id.TwoNN(discard_fraction=0.5)
    intrdim = tnn.fit_transform(X)     
    return intrdim 

# ----------------------- QSAR  --------------------------------------------------
@timeit
def compute_rogi(X, y, metric="euclidean"):
    logging.info('Computing Roughness Index (ROGI).')
    dist = pairwise_distances(X, metric=metric)
    rogi = RoughnessIndex(X=dist, max_dist=dist.max(), Y=y, metric="precomputed")
    try:
        rogi = rogi.compute_index()
    except FloatingPointError:
        logging.warn("FloatingPointError in computation of ROGI.")
        rogi = np.nan
    return rogi

@timeit
def compute_rogi_xd(X, y, metric="euclidean"):
    """
    Corrected version of rogi based on https://github.com/coleygroup/rogi-xd.
    """
    logging.info('Computing Roughness Index XD (ROGI-XD).')
    try:
        dist = pairwise_distances(X, metric=metric)
        rogi = rogixd(dist, y, metric="precomputed", max_dist=dist.max()).rogi
    except FloatingPointError:
        logging.warn("FloatingPointError in computation of ROGI-XD.")
        rogi = np.nan
    return rogi

@timeit
def compute_sari(X, labels, metric):
    """
    Computes the structure activity relationship index.
    Code from https://github.com/coleygroup/rogi.
    """
    dist = pairwise_distances(X, metric=metric)
    sim = 1 - dist
    sari = SARI(pKi=labels, sim_matrix=sim)
    return sari.compute_sari()

@timeit
def compute_rmodi(X, labels, metric):
    """
    Computes the regression modelability index.
    Code from https://github.com/coleygroup/rogi.
    """
    dist = pairwise_distances(X, metric=metric)
    rmodi = RMODI(Dx=dist, Y=labels)
    return rmodi

@timeit
def compute_dataset_descriptors(smiles_list, return_mean=True):
    """
    Computes the QED descritors for a list of smiles.
    """
    druglikeness_scores = []
    mws = []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        druglikeness = Descriptors.MolLogP(molecule)
        mw = Descriptors.MolWt(molecule)
        druglikeness_scores.append(druglikeness)
        mws.append(mw)
    if return_mean:
        return np.array(druglikeness_scores).mean(), np.array(mws).mean()
    else:
        return druglikeness_scores, mws

# ----------------------- ISOTROPY -----------------------------------------------
@timeit
def compute_isoscore(X):
    logging.info('Computing IsoScore.')
    iso_score = IsoScore.IsoScore(X)
    return iso_score.item()


@timeit
def compute_isoscore_star(X):
    logging.info('Computing IsoScore*.')
    isc = istar()
    cov = ShrunkCovariance().fit(X).covariance_
    return isc.isoscore_star(points=X, C0=cov, zeta=0.1, is_eval=True).item()

