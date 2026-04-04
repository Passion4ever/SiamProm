# SiamProm: a framework for cyanobacterial promoter identification

SiamProm originated from the paper **Recognition of Cyanobacteria Promoters via Siamese Network-based Contrastive Learning under Novel Non-promoter Generation**.

## The architecture of SiamProm

![SiamProm](./figs/fig2.webp)


## Dependency

| Main Package 	| Version 	|
| ------------	| -------:	|
| Python       	| 3.9.18  	|
| Pytorch      	| 1.13.1  	|
| CUDA         	| 11.6.1   	|
| Scikit-learn  | 1.3.2   	|
| Pandas      	| 2.1.4   	|
| Hydra        	| 1.3.2   	|
| Pyyaml      	| 6.0.1   	|

Build your environment manually or through a yaml file.

### YAML file

```bash
conda env create -f env_SiamProm.yaml
conda activate SiamProm
```

## Usage

### Training

```bash
python train.py sampling=phantom
```

Optional values for the parameter `sampling` are `phantom`(default), `random`, `cds`, and `partial`.

> For more detailed instructions on parameter management and configuration usage, please refer to the [Hydra](https://hydra.cc/docs/1.3/intro/) documentation.

### Inference

```bash
python predict.py --checkpoint weights/siamprom_phantom.pth --fasta seqs.fasta --output results.csv
```

| Argument | Description | Default |
| -------------- | --------------------------------- | ------- |
| `--checkpoint` | Model checkpoint path | — |
| `--fasta` | Input FASTA file (81bp sequences) | — |
| `--output` | Output CSV path | — |
| `--device` | `cpu` or GPU index | 0 |
| `--batch-size` | Batch size | 256 |
| `--threshold` | Classification threshold | 0.5 |

The output CSV contains columns: `name`, `sequence`, `prediction`, `confidence`.