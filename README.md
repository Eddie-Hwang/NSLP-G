# NSLP-G

This is repository for NSLP-G.
Pytorch implementation of the paper [**"Non-Autoregressive Sign Language
Production with Gaussian Space"**](https://www.bmvc2021-virtualconference.com/assets/papers/1102.pdf), [BMVC 2021](https://www.bmvc2021-virtualconference.com).

![teaser_dark](figures/nslpg.png)

#### Bibtex
If you find this code useful in your research, please cite:

```
@inproceedings{hwang2021non,
  title={Non-Autoregressive Sign Language Production with Gaussian Space},
  author={Hwang, Eui Jun and Kim, Jung-Ho and Park, Jong C.},
  booktitle={The 32nd British Machine Vision Conference (BMVC 21)},
  year={2021},
  organization={British Machine Vision Conference (BMVC)}
}
```

## Installation :construction_worker:
### 1. Create conda environment
```
conda env create -f environment.yaml
conda activate nslp
```

### 2. Download the datasets
Please download dataset from
[Phoenix14-T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) and 
[How2Sign](https://how2sign.github.io/)

Note: The data classes are implemented in ```scripts/data.py```, but you can create your own.

## How to use NSLP-G
#### Training Spatial VAE
```bash
python main.py -c configs/spavae.yaml
```
#### Training NonAutoregressive SLP
```bash
python main.py -c configs/gs.yaml
```