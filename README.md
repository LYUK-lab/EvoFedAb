<p align="left">
  <img src="assets/evofedab-logo.png" alt="EvoFedAb logo" width="360" />
</p>

We provide a data-decentralized training that uses cached, frozen ESM contextualized embeddings, a feature-space
encoder, and a Gaussian affinity head that predicts `mu` and `log_var`.
It supports various tasks in Ab-Ag binding affinity prediction.
We hope this framework can contribute to the development of antibody engineering collaboration when privacy matters.

## Setup

Python 3.12 is required. Install PyTorch for your hardware, then

```bash
pip install -r requirements.txt
```

| Dataset | `data.group` |
| --- | --- |
| AbRank | `pkd`, `pic50` |
| AbBiBench | `pkd`, `lenrich` |


For regression, set `task` to `regression` and the selected algorithm's `loss` to `{"gaussian_nll": 1.0}`.

Set `distribution` to `true` to
broadcast the global model each round.

Encoders: Siamese MLP (`mlp`), DuaDeep-D (`duadeep`), MINT-D (`mint`), AntiBinder-D (`antibinder`).

Algorithms: FedAvg-Ab (`fedavg`), FedAvgM-Ab (`fedavgM`), FedAdamW-Ab (`fedadamW`), FedProx-Ab (`fedprox`), SCAFFOLD-Ab (`scaffold`), FedDisco-Ab (`feddisco`),
FedImpro-Ab (`fedimpro`), FedMoSWA-Ab (`fedmoswa`).

## Run

Feel free to customize the setup in `configs/abrank.json` and
`configs/abbibench.json` to run your own experiments based on our framework.


Run preprocessing first if the configured embedding cache is missing.

Set `model.esm_path` to `"no-esm"` in either config to train directly on raw
sequences with learned amino acid embeddings.
This mode requires no preprocessing mode in advance.

```bash
python run_fedaff.py --dataset abrank --mode preprocess --device cuda:0
python run_fedaff.py --dataset abbibench --mode preprocess --device cuda:0

python run_fedaff.py --dataset abrank --mode train --device cuda:0 \
  --fl-algorithm fedavg --feature-space-adaptor mlp
python run_fedaff.py --dataset abbibench --mode train --device cuda:0 \
  --fl-algorithm fedavg --feature-space-adaptor duadeep
```

## Acknowledgements
A paper based on this implementation is under review: Federated Antibody-Antigen Binding Affinity Prediction with Evolutionary Landscape, Bioinformatics, 2026.

The implementation referred to many existing works and repositories as remarked in the related modules. We sincerely thank the contributors to the community.
