import numpy as np
import pandas as pd
import torch
import safe
import types
import deepchem as dc
from tqdm import tqdm
from map4 import MAP4
from rdkit import Chem
from numpy.typing import ArrayLike
from rdkit.Chem import Descriptors
from typing import Any, Callable, Iterable, Optional
from typing_extensions import Self
from transformers import AutoModel, AutoTokenizer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from molfeat.trans.pretrained.hf_transformers import HFModel
from safe.trainer.model import SAFEDoubleHeadsModel
from safe.tokenizer import SAFETokenizer
from sklearn.preprocessing import MinMaxScaler



class PretrainedHuggingfaceTransformer():
    def __init__(self, kind_long) -> None:
        self.model = AutoModel.from_pretrained(kind_long, deterministic_eval=True, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(kind_long, trust_remote_code=True)

    def __call__(self, X) -> torch.Any:
        if isinstance(X, pd.core.series.Series):
            X = X.values.tolist()
        inputs = self.tokenizer(X, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output.detach().numpy()
    

class SAFEMolTransformer(PretrainedHFTransformer):
    def __init__(self, kind=None, notation="safe", **kwargs):
        if kind is None:
            safe_model = SAFEDoubleHeadsModel.from_pretrained("datamol-io/safe-gpt")
            safe_tokenizer = SAFETokenizer.from_pretrained("datamol-io/safe-gpt")
            kind = HFModel.from_pretrained(safe_model, safe_tokenizer.get_pretrained())
        super().__init__(kind, notation=None, **kwargs)
        self.converter.converter = types.SimpleNamespace(decode=safe.decode, encode=safe.utils.convert_to_safe)
        self.kind = "safe-gpt"


    def __call__(self, X):
        if isinstance(X, pd.core.series.Series):
            X = X.values.tolist()
        return self.transform(X)


class MACCSFingerprint():
    def __init__(self) -> None:
        self.featurizer = dc.feat.MACCSKeysFingerprint()

    def __call__(self, X) -> torch.Any:
        if isinstance(X, pd.core.series.Series):
            X = X.values.tolist()
        results = []
        for x in X:
            feat = self.featurizer.featurize([x])
            results.append(feat)
        return np.array(results)



class DescriptorFeaturizer():
    """
    Code from: https://github.com/coleygroup/rogi-xd
    """
    def __init__(self, descs: Optional[Iterable[str]] = None, scale: bool = True, **kwargs):
        self.DEFAULT_DESCRIPTORS = [
            "MolWt",
            "FractionCSP3",
            "NumHAcceptors",
            "NumHDonors",
            "NOCount",
            "NHOHCount",
            "NumAliphaticRings",
            "NumAliphaticHeterocycles",
            "NumAromaticHeterocycles",
            "NumAromaticRings",
            "NumRotatableBonds",
            "TPSA",
            "qed",
            "MolLogP",
        ]
        self.DESC_TO_FUNC: dict[str, Callable[[Chem.Mol], float]] = dict(Descriptors.descList)
        self.descs = set(descs or self.DEFAULT_DESCRIPTORS)
        self.scale = scale
        self.quiet = False

        super().__init__(**kwargs)

    @property
    def descs(self) -> list[str]:
        return self.__names

    @descs.setter
    def descs(self, descs: Iterable[str]):
        self.__names = []
        self.__funcs = []
        invalid_names = []
        
        for desc in descs:
            func = self.DESC_TO_FUNC.get(desc)
            if func is None:
                invalid_names.append(desc)
            else:
                self.__names.append(desc)
                self.__funcs.append(func)


    def __len__(self) -> int:
        return len(self.__funcs)

    def __call__(self, smis):
        mols = [Chem.MolFromSmiles(smi) for smi in smis]
        xss = [
            [self.DESC_TO_FUNC[func](mol) for func in self.DEFAULT_DESCRIPTORS]
            for mol in tqdm(mols, leave=False, disable=self.quiet)
        ]
        X = np.array(xss)

        return MinMaxScaler().fit_transform(X) if self.scale else X

    def finetune(self, *splits: Iterable[tuple[Iterable[str], ArrayLike]]) -> Self:
        """a :class:`DescriptorFeaturizer` can't be finetuned"""

        return self

    def __str__(self) -> str:
        return "descriptor"


class RandomFeaturizer():
    """
    Code from: https://github.com/coleygroup/rogi-xd
    """
    def __init__(self, length: Optional[int] = 128, **kwargs):
        self.length = length or 128

    def __call__(self, smis: Iterable[str]) -> np.ndarray:
        n = sum(1 for _ in smis)

        return np.random.rand(n, self.length)

    def finetune(self, *splits: Iterable[tuple[Iterable[str], ArrayLike]]) -> Self:
        return self

    def __str__(self) -> str:
        return f"{self.alias}{self.length}"
    

class Mol2VecFeaturizer():
    def __init__(self) -> None:
        self.model = dc.feat.Mol2VecFingerprint()

    def __call__(self, smiles) -> Any:
        return self.model.featurize(smiles)


class MAP4Featurizer():
    def __init__(self) -> None:
        self.model = MAP4(1024, 2)

    def __call__(self, smiles) -> Any:
        mol_smiles = [Chem.MolFromSmiles(s) for s in smiles]
        return self.model.calculate_many(mol_smiles)
