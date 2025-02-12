import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*') 

DATA_DIR = Path('../data/raw/')

def load_lipo_regression(return_mask=False):
    """
    Lipophilicity, also known as hydrophobicity, is a measure of how readily a substance dissolves 
    in nonpolar solvents (such as oil) compared to polar solvents (such as water). 
    The dataset contains experimental measurements.
    """
    sdf_file = DATA_DIR / Path("LIPO.sdf")
    suppl = Chem.SDMolSupplier(sdf_file)
    data = pd.DataFrame()
    for i, mol in enumerate(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            target = float(mol.GetProp("target").replace(">", "").replace("<", ""))
        except:
            smiles = "invalid"
            target = np.nan
        data.loc[i, "smiles"] = smiles
        data.loc[i, "target"] = target

    # Filter invalid mols
    mask = [True if Chem.MolFromSmiles(m) != None else False for m in data.loc[:, "smiles"]]
    target_mask = [not np.isnan(m) for i, m in enumerate(data.loc[:, "target"]) ]
    mask = (mask and target_mask)
    data = data[mask].reset_index(drop=True)
    data["dataset"] = "LIPO"
    return data, mask if return_mask else data


def load_dpp4_regression(return_mask=False):
    """
    DPP-4 inhibitors (DPP4) was extract from ChEMBL with DPP-4 target. 
    The data was processed by removing salt and normalizing molecular structure,
    with molecular duplication examination, leaving 3933 molecules.
    """

    sdf_file = DATA_DIR / Path("DPP4.sdf")
    suppl = Chem.SDMolSupplier(sdf_file)
    data = pd.DataFrame()
    for i, mol in enumerate(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            target = float(mol.GetProp("target").replace(">", "").replace("<", ""))
        except:
            smiles = "invalid"
            target = np.nan
        data.loc[i, "smiles"] = smiles
        data.loc[i, "target"] = target

    # Filter invalid mols
    mask = [True if Chem.MolFromSmiles(m) != None else False for m in data.loc[:, "smiles"]]
    target_mask = [not np.isnan(m) for i, m in enumerate(data.loc[:, "target"]) ]
    mask = (mask and target_mask)
    data = data[mask].reset_index(drop=True)
    data["dataset"] = "DPP4"
    return data, mask if return_mask else data


def load_adra1a_regression(return_mask=False):
    """
    Alpha-1 adrenergic receptor dataset with 3092 compounds and their
    inhibitory constant (Ki).
    """
    sdf_file = DATA_DIR / Path("ADRA1A.sdf")
    suppl = Chem.SDMolSupplier(sdf_file)
    data = pd.DataFrame()
    for i, mol in enumerate(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            target = np.log(float(mol.GetProp("Ki (nM)").replace(">", "").replace("<", "")))
        except:
            smiles = "invalid"
            target = np.nan
        data.loc[i, "smiles"] = smiles
        data.loc[i, "target"] = target

    # Filter invalid mols
    mask = [True if Chem.MolFromSmiles(m) != None else False for m in data.loc[:, "smiles"]]
    data = data[mask].reset_index(drop=True)
    data["dataset"] = "ADRA1A"
    return data, mask if return_mask else data


def load_alox5ap_regression(return_mask=False):
    """
    Arachidonate 5-lipoxygenase-activating protein inhibition data.
    """
    sdf_file = DATA_DIR / Path("ALOX5AP.sdf")
    suppl = Chem.SDMolSupplier(sdf_file)
    data = pd.DataFrame()
    for i, mol in enumerate(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            target = np.log(float(mol.GetProp("Ki (nM)").replace(">", "").replace("<", "")))
        except:
            smiles = "invalid"
            target = np.nan
        data.loc[i, "smiles"] = smiles
        data.loc[i, "target"] = target

    # Filter invalid mols
    mask = [True if Chem.MolFromSmiles(m) != None else False for m in data.loc[:, "smiles"]]
    data = data[mask].reset_index(drop=True)
    data["dataset"] = "ALOX5AP"
    return data, mask if return_mask else data


def load_musc1_regression(return_mask=False):
    """
    Muscarinic acetylcholine receptor M1 dataset 
    with 2616 compounds and their inhibitory constant (Ki).
    """
    sdf_file = DATA_DIR / Path("MUSC1.sdf")
    suppl = Chem.SDMolSupplier(sdf_file)
    data = pd.DataFrame()
    for i, mol in enumerate(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            target = np.log(float(mol.GetProp("Ki (nM)").replace(">", "").replace("<", "")))
        except:
            smiles = "invalid"
            target = np.nan
        data.loc[i, "smiles"] = smiles
        data.loc[i, "target"] = target

    # Filter invalid mols
    mask = [True if Chem.MolFromSmiles(m) != None else False for m in data.loc[:, "smiles"]]
    data = data[mask].reset_index(drop=True)
    data["dataset"] = "MUSC1"
    return data, mask if return_mask else data


def load_atr_regression(return_mask=False):
    """
    Serinthreonine protein kinase ATR dataset
    with 4251 compounds and their inhibitory constant (Ki).
    """
    sdf_file = DATA_DIR / Path("ATR.sdf")
    suppl = Chem.SDMolSupplier(sdf_file)
    data = pd.DataFrame()
    for i, mol in enumerate(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            target = np.log(float(mol.GetProp("Ki (nM)").replace(">", "").replace("<", "")))
        except:
            smiles = "invalid"
            target = np.nan
        data.loc[i, "smiles"] = smiles
        data.loc[i, "target"] = target

    # Filter invalid mols
    mask = [True if Chem.MolFromSmiles(m) != None else False for m in data.loc[:, "smiles"]]
    data = data[mask].reset_index(drop=True)
    data["dataset"] = "ATR"
    return data, mask if return_mask else data


def load_jak1_regression(return_mask=False):
    """
    Tyrosine kinease protein dataset
    with 4251 compounds and their inhibitory constant (Ki).
    """
    sdf_file = DATA_DIR / Path("JAK1.sdf")
    suppl = Chem.SDMolSupplier(sdf_file)
    data = pd.DataFrame()
    for i, mol in enumerate(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            target = np.log(float(mol.GetProp("Ki (nM)").replace(">", "").replace("<", "")))
        except:
            smiles = "invalid"
            target = np.nan
        data.loc[i, "smiles"] = smiles
        data.loc[i, "target"] = target

    # Filter invalid mols
    mask = [True if Chem.MolFromSmiles(m) != None else False for m in data.loc[:, "smiles"]]
    data = data[mask].reset_index(drop=True)
    data["dataset"] = "JAK1"
    return data, mask if return_mask else data


def load_jak2_regression(return_mask=False):
    """
    Tyrosine kinease protein dataset
    """
    sdf_file = DATA_DIR / Path("JAK2.sdf")
    suppl = Chem.SDMolSupplier(sdf_file)
    data = pd.DataFrame()
    for i, mol in enumerate(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            target = float(mol.GetProp("target").replace(">", "").replace("<", ""))
        except:
            smiles = "invalid"
            target = np.nan
        data.loc[i, "smiles"] = smiles
        data.loc[i, "target"] = target

    # Filter invalid mols
    mask = [True if Chem.MolFromSmiles(m) != None else False for m in data.loc[:, "smiles"]]
    data = data[mask].reset_index(drop=True)
    data["dataset"] = "JAK2"
    return data, mask if return_mask else data


def load_sol_regression(return_mask=False):
    sdf_file = DATA_DIR / Path("SOL.sdf")
    suppl = Chem.SDMolSupplier(sdf_file)
    data = pd.DataFrame()
    for i, mol in enumerate(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            target = float(mol.GetProp("LOG SOLUBILITY PH 6.8 (ug/mL)").replace(">", "").replace("<", ""))
        except:
            smiles = "invalid"
            target = np.nan
        data.loc[i, "smiles"] = smiles
        data.loc[i, "target"] = target

    # Filter invalid mols
    mask = [True if Chem.MolFromSmiles(m) != None else False for m in data.loc[:, "smiles"]]
    target_mask = [not np.isnan(m) for m in data.loc[:, "target"]]
    mask = (mask and target_mask)
    data = data[mask].reset_index(drop=True)
    data["dataset"] = "SOL"
    return data, mask if return_mask else data


def load_hlmc_regression(return_mask=False):
    sdf_file = DATA_DIR / Path("HLMC.sdf")
    suppl = Chem.SDMolSupplier(sdf_file)
    data = pd.DataFrame()
    for i, mol in enumerate(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            target = float(mol.GetProp("LOG HLM_CLint (mL/min/kg)").replace(">", "").replace("<", ""))
        except:
            smiles = "invalid"
            target = np.nan
        data.loc[i, "smiles"] = smiles
        data.loc[i, "target"] = target

    # Filter invalid mols
    mask = [True if Chem.MolFromSmiles(m) != None else False for m in data.loc[:, "smiles"]]
    target_mask = [not np.isnan(m) for m in data.loc[:, "target"]]
    mask = (mask and target_mask)
    data = data[mask].reset_index(drop=True)
    data["dataset"] = "HLMC"
    return data, mask if return_mask else data


def load_kor_regression(return_mask=False):
    sdf_file = DATA_DIR / Path("KOR.sdf")
    suppl = Chem.SDMolSupplier(sdf_file)
    data = pd.DataFrame()
    for i, mol in enumerate(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            target = np.log(float(mol.GetProp("Ki (nM)").replace(">", "").replace("<", "")))
        except:
            smiles = "invalid"
            target = np.nan
        data.loc[i, "smiles"] = smiles
        data.loc[i, "target"] = target

    # Filter invalid mols
    mask = [True if Chem.MolFromSmiles(m) != None else False for m in data.loc[:, "smiles"]]
    data = data[mask].reset_index(drop=True)
    data["dataset"] = "KOR"
    return data, mask if return_mask else data


def load_musc2_regression(return_mask=False):
    sdf_file = DATA_DIR / Path("MUSC2.sdf")
    suppl = Chem.SDMolSupplier(sdf_file)
    data = pd.DataFrame()
    for i, mol in enumerate(suppl):
        try:
            smiles = Chem.MolToSmiles(mol)
            target = np.log(float(mol.GetProp("Ki (nM)").replace(">", "").replace("<", "")))
        except:
            smiles = "invalid"
            target = np.nan
        data.loc[i, "smiles"] = smiles
        data.loc[i, "target"] = target

    # Filter invalid mols
    mask = [True if Chem.MolFromSmiles(m) != None else False for m in data.loc[:, "smiles"]]
    data = data[mask].reset_index(drop=True)
    data["dataset"] = "MUSC2"
    return data, mask if return_mask else data