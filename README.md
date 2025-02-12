# TopoLearn

This is the code for the publication "The topology of Molecular Representations in Machine Learning" [paper link]. Below you can find details how to use the python package and how to reproduce the plots and results of the paper.


## Quick-Setup
The topolearn model and code is easily accessible via pypi and can be installed running:

`pip install topolearn`

Below is a minimalistic example how to use the installed package.
```
import numpy as np
from topolearn import topolearn

# Prepare data
X = np.random.normal(3, 1, size=(1000, 128))
sim_metric = "euclidean"

# Compute score
tl = topolearn.TopoLearn()
tl.compute_score(X, sim_metric=sim_metric, clip=True)

# Output: 0.5690018080283844
```

## Advanced-Setup

If you want to re-run experiments or want to continue on existing code you can find more details provided below. 

### General notes and modifications
- The code was executed on Python 3.11.1
- A `requirements.txt` file is provided with all the versions used to run this code. Some of the packages are used for specific representations e.g. transformers for all language models ect. You might not need all of them.
- Map4 was installed running `pip install git+https://git@github.com/reymond-group/map4.git@refs/pull/27/head` which contained some fixes that were not available in the released version
- For `molfeat.trans.pretrained.hf_transformers` we manually added this line
`inputs = [i if isinstance(i, str) else smiles[idx] for idx, i in enumerate(inputs)]` to the PretrainedHFTransformer class, to deal with some errors that were caused in the _convert function
- In the persim module we modified the code to `ax.scatter(dgm[:, 0], dgm[:, 1], size, label=label, edgecolor=edgecolor, alpha=alpha)` to make the persistence diagram plots nicer
- For the SHAP plots we used the beeswarm code provided here `https://github.com/shap/shap/blob/master/shap/plots/_beeswarm.py`


### Data preparation

**Raw datasets**

Navigate to `data` and run `tar -xvzf raw.tar.gz` to unpack the preprocessed 12 datasets provided as sdf files. 

**Featurized data**

The datasets have already been featurized for different molecular representations using the code in `00_dataset_featurization.ipynb`. By running `tar -xvzf features.tar.gz` all pre-computed numeric representations can be extracted. If you plan to add other representations or datasets for training, follow the logic provided in the notebook. Note that some of the used features have been computed with licensed Software.

**Result data**

The results of the analysis are stored in a csv file, which can be unpacked using `tar -xvzf results.tar.gz` in `topolearn/processed_data/`.

### Reproducing the results
**File descriptions**

- `analysis.py`: Main file used to run the analysis, it will iterate all permutations of dataset, representation, model, sample size, ... to train the models and compute the errors and topological descriptors. The results are saved as pandas dataframe to `topolearn/processed_data`. 
- `config.py`: Basic configurations such as the analyzed hyperparameters, splitting technique, error metrics and more.

The code can be executed running:

`python -m topolearn.analysis` 

**Parallel execution**
A SLURM script is provided to run the analysis in parallel: `run_analysis_parallel.bash`. It will start a separate process for each dataset. Run `sbatch <scriptname>` to execute it on a computing cluster. 


### Analysis Notebooks
After the preprocessing has been completed, several notebooks are available to perform visual and statistical analysis of the results. Additionally, the topolearn model is trained based on the results dataframe. All notebooks are explained in detail below.

- `00_dataset_featurization.ipynb`: Notebook to create numeric representations for each SMILES dataset using various featurizers. 
- `01_dataset_processing.ipynb`: Post-processing of the analysis results, such as aggregating seed data, removing nans, mapping representation categories, and computing normalized errors.
- `02_analysis_errors.ipynb`: Analysis of model errors per dataset and representation and other variables.
- `03_analysis_perhom.ipynb`: Analysis and comparison of persistence diagrams and other PH attributes.
- `04_analysis_corr.ipynb`: Statistical analysis including correlations and explained variance analyses.
- `05_analysis_topolearn.ipynb`: Cross-validation and SHAP plots for different evaluation schemes.
- `06_inference.ipynb`: Final training of topolearn model and serialization of model.
- `07_analysis_dataset.ipynb`: Some experiments regarding the distribution of the regression target variables.


### Re-Build the pip TopoLearn package

In case you want to update the underlying model e.g. using a larger training dataset you can easily re-build the pip package as follows.
1. Navigate to `topolearn/pip/topolearn` and implement any changes you have in mind. Here you can find topolearn.pkl, which is the model resulting from `06_inference.ipynb`.
2. In `setup.py` update the package information according to your needs.
3. Run `pip install .` to build the wheels for the package