# SiamProm (Under Preparation)

Recognition of Cyanobacteria Promoters via Siamese Network-based Contrastive Learning under Novel Non-promoter Generation

## The architecture of SiamProm

![SiamProm](./figs/fig2.webp)


## Dependency

| Main Package 	| Version 	|
| ------------	| -------:	|
| Python       	| 3.9.16  	|
| Pytorch      	| 1.12.1  	|
| CUDA         	| 11.7    	|
| cuDNN         | 8.5.0    	|
| Scikit-learn  | 1.2.2   	|
| Pandas      	| 1.5.3   	|
| Hydra        	| 1.3.2   	|
| Pyyaml      	| 6.0   	|

Build your environment manually or through a yaml file.

### YAML file

```bash
conda env create -f env_SiamProm.yaml
conda activate SiamProm
```

## Usage

```python
python train.py sampling=phantom
```

Optional values for the parameter `sampling` are `phantom`(default), `random`, `cds`, and `partial`.

> For more detailed instructions on parameter management and configuration usage, please refer to the [Hydra](https://hydra.cc/docs/1.3/intro/) documentation.