import seaborn as sns
from matplotlib import rc

representation_colors = {'ChemBERTa': "#5d4c86",
                        'ChemGPT': "#7dfc00", 
                        'SafeGPT': "#0ec434", 
                        'MoLFormer-XL': "#228c68", 
                        'Graphormer': "#8ad8e8",
                        'GIN': "#235b54", 
                        'Mol2Vec': "#29bdab", 
                        'RAND': "#3998f5", 
                        'AVALON': "#3750db", 
                        'CATS2D': "#f22020", 
                        'ECFP4': "#f47a22", 
                        'MACCS': "#b732cc",
                        'MAP4': "#2f2aa0",
                        'EState': "#772b9d", 
                        'PubChem': "#f07cab", 
                        'KR': "#d30b94",  
                        'RDKit': "#c3a5b4", 
                        'Pharm2D': "#ffc413", 
                        'TOPO': "#632819",
                        'RingDesc': "#96341c", 
                        'FGCount': "#c56133", 
                        '2DAP': "#e68f66", 
                        'ConstIdx': "#ffcba5", 
                        'MolProp': "#991919", 
                        'WalkPath': "#7f7f7f",
                        'MolProp2': "#277da7", 
                        'Combined': "#37294f"}


dataset_colors = {'LIPO': "#f47a22", 
                  'MUSC1': "#f22020", 
                  'ADRA1A': "#b732cc", 
                  'JAK1': "#3998f5", 
                  'ATR': "#3750db", 
                  'ALOX5AP': "#29bdab", 
                  'JAK2': "#0ec434", 
                  'SOL': "#277da7", 
                  'MUSC2': "#d30b94", 
                  'HLMC': "#ffc413", 
                  'KOR': "#e68f66",
                  "DPP4": "#632819"}

BI_colors = {
    "blue": "#6ad2e2",
    "dark_blue": "#076d7e",
    "red": "#f58a68",
    "violet": "#5d4495",
    "yellow": "#ffe667"
}


def get_palettes():
    return representation_colors, dataset_colors

def load_plot_config():
    rc('text', usetex=False) 
    sns.set_context("paper", rc={"axes.titlesize": 11, 
                                 "axes.labelsize": 11, 
                                 'xtick.labelsize': 9,
                                 'ytick.labelsize': 9,
                                 "legend.textsize": 8,
                                 "legend.titlesize": 11, 
                                 "font.family": 'Liberation Sans'})
    sns.set_theme(style="whitegrid")

def get_default_palette():
    return sns.color_palette(["#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E", "#8EBA42", "#FFB5B8"])

def get_bi_palette():
    return sns.color_palette(BI_colors.values())

