from abc import ABC, abstractmethod
from typing import Any, List, Tuple
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
from molskill.data.featurizers import get_featurizer
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

    def __init__(
        self,
        feature_type: str,
        is_maximize: bool,
        problem_name: str,
        csv_path: str,
        smiles_col: str,
        obj_col: str,
        scorer: Any,
    ) -> None:
        assert feature_type in ["fingerprints"]
        self.feature_type = feature_type
        self.is_maximize = is_maximize
        self.problem_name = problem_name
        self.data_pd_orig = pd.read_csv(csv_path)
        self.SMILES_COL = smiles_col
        self.OBJ_COL = obj_col
        self.scorer = scorer

        # Populate D_cand, consisting of candidate SMILES, cand's objectives
        # and cand's features
        self.cand_smiles = self.data_pd_orig[self.SMILES_COL].to_list()
        self.cand_objs = list(
            torch.from_numpy(self.data_pd_orig[self.OBJ_COL].to_numpy())
            .float()
            .unsqueeze(1)
        )
        assert len(self.cand_objs) == len(self.cand_smiles)
        assert self.cand_objs[0].shape == (1,)
        self.cand_feats = self._get_features()
        assert len(self.cand_feats[0].shape) == 1
        self.dim = self.cand_feats[0].shape[0]
        self._validate()

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
            TensorDataset(
                torch.stack(self.cand_feats),
                torch.stack(self.cand_objs),
            ),
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
        feats = self.cand_feats.pop(idx)
        obj = self.cand_objs.pop(idx)

        return smiles, feats, obj

    def get_preference(self, smiles0: List[str], smiles1: List[str]) -> int:
        """
        Given a pair of smiles, return 0 if the first one is preferred,
        otherwise return 1. Accept list for batch processing.

        Parameters:
        -----------
        smiles0: List[str]
        smiles1: List[str]

        Returns:
        --------
        label: List[int]
            Either 0 or 1, depending which x's is preferred; shape `(len(smiles0), 1)`
        """
        assert len(smiles0) == len(smiles1)
        # Shape `(len(smiles0), 2)`
        scores = np.stack([self._score(smiles0), self._score(smiles1)]).T
        return np.argmax(scores, axis=1, keepdims=True)

    @abstractmethod
    def _score(self, smiles: List[str]) -> float:
        raise NotImplementedError

    def _get_features(self, force: bool = False) -> None:
        path = f"{self.CACHE_DIR}/chem/{self.problem_name}"
        if not os.path.exists(path):
            os.makedirs(path)

        # Cache features if not already exists
        if force or not os.path.exists(f"{path}/feats_{self.feature_type}.pt"):
            match self.feature_type:
                case "fingerprints":
                    feats = []
                    pbar = tqdm(self.data_pd_orig[self.SMILES_COL])
                    pbar.set_description(f"[Caching features]")
                    for smiles in pbar:
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
        names = [
            "data_pd_orig",
            "SMILES_COL",
            "OBJ_COL",
            "cand_smiles",
            "cand_objs",
            "dim",
        ]

        for name in names:
            if getattr(self, name) is None:
                raise ValueError(f"Class variable {name} is uninitialized!")

        return True

    def __len__(self):
        return len(self.cand_smiles)


class MolSkillProblem(DiscreteChemProblem, ABC):

    def __init__(
        self,
        feature_type: str,
        is_maximize: bool,
        problem_name: str,
        csv_path: str,
        smiles_col: str,
        obj_col: str,
    ) -> None:
        super().__init__(
            feature_type,
            is_maximize,
            problem_name,
            csv_path,
            smiles_col,
            obj_col,
            scorer=MolSkillScorer(num_workers=0, verbose=False),
        )

    def _score(self, smiles: List[str]) -> float:
        return self.scorer.score(smiles)


class Kinase(MolSkillProblem):

    def __init__(self, feature_type: str) -> None:
        super().__init__(
            feature_type,
            is_maximize=False,
            problem_name="Kinase",
            csv_path=f"{self.DATA_DIR}/enamine10k_kinase_filtered.csv.gz",
            smiles_col="SMILES",
            obj_col="score",
        )


class AmpC(MolSkillProblem):

    def __init__(self, feature_type: str) -> None:
        super().__init__(
            feature_type,
            is_maximize=False,
            problem_name="AmpC",
            csv_path=f"{self.DATA_DIR}/Zinc_AmpC_Docking_filtered.csv",
            smiles_col="SMILES",
            obj_col="dockscore",
        )


class D4(MolSkillProblem):

    def __init__(self, feature_type: str) -> None:
        super().__init__(
            feature_type,
            is_maximize=False,
            problem_name="D4",
            csv_path=f"{self.DATA_DIR}/Zinc_D4_Docking_filtered.csv",
            smiles_col="SMILES",
            obj_col="dockscore",
        )


PROBLEM_LIST = {
    "kinase": Kinase,
    "ampc": AmpC,
    "d4": D4,
}


if __name__ == "__main__":
    problem = Kinase(feature_type="fingerprints", is_maximize=True)
