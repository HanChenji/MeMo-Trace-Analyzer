
# MeMo: Enhancing representative sampling via mechanistic micro-mdoel signatures.

## What's MeMo
MeMo is the code signature proposed to enhance the performance evaluation accuracy of representative sampling in the pre-silicon stage of $\mu$Arch designs.

MeMo employs fetch, issue, and cache micro-models under different parameter configurations, to categorize imperfections in $\mu$Arch structures systematically, 
Each of these micro-model concentrates on certain structure constraints while idealizing the others.
These models enable the isolated portray of different program characteristics and their performance responses without being confined to specific $\mu$Arch.

## How to Run

### Compile:
MeMo is built based on the ZSim platform, which utilizes the PIN as the trace generator.
```shell
scons
```

Python environment setup:
```shell
pip install -r requirements.txt
```

### Demo

This repo provides a demo to profile, cluster, and then analyze for __gcc__ in SPE CPU 2017, using MeMo and BBV as the code signatures, respectively.

```shell
./profile.sh 
./cluster.sh
./analyze.sh
```