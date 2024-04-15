from abc import ABC, abstractmethod
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

import sys

sys.path.append("molskill")
from molskill.scorer import MolSkillScorer

sys.path.append("synthetic_accessibility_project")
from scripts.mpscore import MPScore


class DiscreteChemProblem(ABC):
    """
    Parameters:
    -----------
    is_maximize: bool
    """

    DATA_DIR = "./data"
    CACHE_DIR = "./cache"

    def __init__(self, feature_type: str, is_maximize: bool) -> None:
        assert feature_type in ["fingerprints"]
        self.feature_type = feature_type
        self.is_maximize = is_maximize

        # To be overloaded
        self.problem_name = None
        self.data_pd_orig = None
        self.SMILES_COL = None
        self.OBJ_COL = None

        self.cand_smiles = None  # List[str]
        self.cand_feats = None  # List[Tensor(n_dim)]
        self.cand_objs = None  # List[float]

    def get_dataloader(self, batch_size: int = 256) -> DataLoader:
        """
        Get the candidate data, each `(feat, target)`, in the form of a dataloader.

        Parameters:
        -----------
        batch_size: int, default 256

        Returns:
        --------
        dataloader: torch.utils.data.DataLoader
            A non-shuffled PyTorch dataloader of the candidate molecules.
        """
        return DataLoader(
            TensorDataset(torch.stack(self.cand_feats), torch.stack(self.cand_obj)),
            batch_size=batch_size,
            shuffle=False,
        )

    def pop_candidate(self, idx: int) -> Tuple[str, float]:
        """
        Remove a candidate molecule from the pool of candidates, and return it.

        Parameters:
        -----------
        idx: int
            The index is consistent to the index of the dataloader from `get_dataloader`

        Returns:
        --------
        smiles: str
        obj: float
        """
        smiles = self.cand_smiles.pop(idx)
        self.cand_feats.pop(idx)
        obj = self.cand_objs.pop(idx)

        return smiles, obj

    def get_preference(self, smiles0: str, smiles1: str) -> int:
        """
        Given a pair of smiles, return 0 if the first one is preferred,
        otherwise return 1.

        Parameters:
        -----------
        smiles0: str
        smiles1: str

        Returns:
        --------
        label: int
            Either 0 or 1, depending which x's is preferred
        """
        return np.argmax([self._score(smiles0), self._score(smiles1)])

    @abstractmethod
    def _score(self, smiles: str) -> float:
        raise NotImplementedError

    def _get_features(self, force: bool = False) -> None:
        path = f"{self.CACHE_DIR}/chem/{self.problem_name}"
        if not os.path.exists(path):
            os.makedirs(path)

        # Cache features if not already exists
        if force or not os.path.exists(f"{path}/{self.feature_type}.pt"):
            match self.feature_type:
                case "fingerprints":
                    feats = []
                    for smiles in tqdm(self.data_pd_orig[self.SMILES_COL]):
                        fp = AllChem.GetMorganFingerprintAsBitVect(
                            Chem.MolFromSmiles(smiles), radius=2, nBits=1024
                        )
                        feats.append(torch.tensor(np.array(fp)).float())
                case _:
                    raise ValueError("Unsupported feature type")

            torch.save(feats, f"{path}/feats_{self.feature_type}.pt")
            return feats
        else:
            return torch.load(f"{path}/feats_{self.feature_type}.pt")

    def _validate(self) -> bool:
        names = ["data_pd_orig", "SMILES_COL", "OBJ_COL", "cand_smiles", "cand_objs"]

        for name in names:
            if getattr(self, name) is None:
                raise ValueError(f"Class variable {name} is uninitialized!")

        return True


class KinaseMolSkill(DiscreteChemProblem):

    def __init__(self, feature_type: str, is_maximize: bool) -> None:
        super().__init__(feature_type, is_maximize)

        self.problem_name = "Kinase"
        self.data_pd_orig = pd.read_csv(f"{self.DATA_DIR}/enamine10k.csv.gz")
        self.SMILES_COL = "SMILES"
        self.OBJ_COL = "score"

        self.cand_smiles = self.data_pd_orig[self.SMILES_COL].to_list()
        self.cand_objs = self.data_pd_orig[self.OBJ_COL].to_list()
        self.cand_feats = self._get_features()

        self._validate()

        self.scorer = MolSkillScorer()

    def _score(self, smiles: str) -> float:
        return self.scorer(smiles)


class ReaxysMPScore(DiscreteChemProblem):

    def __init__(self, feature_type: str, is_maximize: bool) -> None:
        super().__init__(feature_type, is_maximize)

        self.problem_name = "Reaxys"
        self.data_pd_orig = pd.read_csv(f"{self.DATA_DIR}/reaxys_database.csv")
        self.SMILES_COL = "smiles"
        self.OBJ_COL = "score"

        self.cand_smiles = self.data_pd_orig[self.SMILES_COL].to_list()
        self.cand_objs = self.data_pd_orig[self.OBJ_COL].to_list()
        self.cand_feats = self._get_features()

        self._validate()

        self.scorer = MPScore()

    def _score(self, smiles: str) -> float:
        return self.scorer(smiles)


if __name__ == '__main__':
    problem = KinaseMolSkill(feature_type="fingerprints", is_maximize=True)
