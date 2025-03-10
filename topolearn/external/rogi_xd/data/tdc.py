from __future__ import annotations

from typing import Optional

import pandas as pd
from tdc.single_pred import ADME, Tox
from tdc.utils import retrieve_label_name_list

from rogi_xd.utils.utils import CACHE_DIR
from rogi_xd.data.base import DataModule
from rogi_xd.utils.exceptions import InvalidDatasetError


class TdcDataModule(DataModule):
    __ADME_DATASETS = {
        "CLEARANCE_HEPATOCYTE_AZ",
        "SOLUBILITY_AQSOLDB",
        "CLEARANCE_MICROSOME_AZ",
        "VDSS_LOMBARDO",
        "CACO2_WANG",
        "HALF_LIFE_OBACH",
        "LIPOPHILICITY_ASTRAZENECA",
        "PPBR_AZ",
        "HYDRATIONFREEENERGY_FREESOLV",
    }
    __TOX_DATASETS = {"HERG_CENTRAL", "LD50_ZHU"}

    @classmethod
    @property
    def datasets(cls) -> set[str]:
        return {*cls.__ADME_DATASETS, *cls.__TOX_DATASETS}

    @classmethod
    def tasks(cls, dataset: str) -> list[str]:
        cls.check_dataset(dataset)
        try:
            return retrieve_label_name_list(dataset)
        except KeyError:
            return []

    @classmethod
    def get_all_data(cls, dataset: str, task: Optional[str] = None) -> pd.DataFrame:
        cls.check_task(dataset, task)

        dataset_ = dataset.upper()
        if dataset_ in cls.__ADME_DATASETS:
            df: pd.DataFrame = ADME(dataset, CACHE_DIR, label_name=task).get_data("df")
        elif dataset_ in cls.__TOX_DATASETS:
            df: pd.DataFrame = Tox(dataset, CACHE_DIR, label_name=task).get_data("df")
        else:
            raise InvalidDatasetError(dataset, cls.datasets)

        return df.rename(columns={"Drug": "smiles", "Y": "y"})
