from sklearn.metrics import (mean_absolute_error, root_mean_squared_error, 
                             mean_absolute_percentage_error, r2_score)
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

def get_configs(dataset_list):
    config = {
        "representations": ["ChemBERTa", "ChemGPT", "SafeGPT", "MoLFormer-XL", "Graphormer", 
                            "GIN", "Mol2Vec", "RAND", "AVALON", "CATS2D", "ECFP4", "MACCS",      
                            "MAP4", "EState", "PubChem", "KR", "RDKit", "Pharm2D", "TOPO",       
                            "RingDesc", "FGCount", "2DAP", "ConstIdx", "MolProp", "WalkPath",   
                            "MolProp2", "Combined"],
        "datasets": dataset_list, 
        "n_samples": [500, 1000, 1500, 2000, 2500], 
        "models": [{"model": MLPRegressor,
                    "param_grid": {'hidden_layer_sizes': [(100,), (300,), (50, 50), (128, 64)], 
                                'learning_rate_init': [0.001, 0.1, 0.0001],
                                "max_iter": [1000],
                                "activation": ["relu"],
                                "batch_size": [32, 128],  
                                "early_stopping": [True]}},
                    {"model": RandomForestRegressor,
                    "param_grid": {'n_estimators': [100, 200, 300], 
                                    'max_depth': [5, 10, 20, 50],
                                    'min_samples_leaf': [1, 5, 10]}
                        }
                    ]
    }
    global_params = {"eval_metrics": {
                        "MAE": mean_absolute_error,
                        "RMSE": root_mean_squared_error,
                        "MAPE": mean_absolute_percentage_error,
                        "R2": r2_score
                    },
                    "n_splits": 5,    
                    "split_type": "random",
                    "sim_metrics":  ["jaccard", "euclidean", "manhattan", "cosine"],
                    "store_preds": True,
                    "debug": True,
                    "scale_y": False,
                    "seeds": [2023, 2024, 2025, 2026]} 
    
    return config, global_params
