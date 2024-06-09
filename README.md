# Bayesian Optimization With _Asynchronous_ Expert Feedback

Code to reproduce the paper:

```
@inproceedings{kristiadi2024asyncBO,
  title={How Useful is Intermittent, Asynchronous Expert Feedback for {B}ayesian Optimization?},
  author={Kristiadi, Agustinus and Strieth-Kalthoff, Felix and Subramanian, Sriram Ganapathi and Fortuin, Vincent and Poupart, Pascal and Pleiss, Geoff},
  booktitle={AABI Non Archival Track},
  year={2024}
}
```

## Setup

Requires `python >= 3.10` and `pytorch >= 2.0`. Then install the dependencies:

```bash
pip install git+https://git@github.com/aleximmer/laplace
pip install git+https://git@github.com/wiseodd/laplace-bayesopt
pip install lightning rdkit tqdm gauche botorch
```

Next, download the submodules (MolSkill, for expert simulator in chemistry problems).

```bash
git submodule init
git submodule update
```

## Usage

The expert simulator for toy problems is trained using `toy_train_reward.py`. The experiment scripts are `toy_bo.py`, `chem_bo.py`. Run `<SCRIPT_NAME>.py --help` for available options.
