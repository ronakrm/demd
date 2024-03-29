# Efficient Computation of d-Dimensional Earth Mover's Distance
![DEMD Minimization](assets/demd_hists.png)

This folder contains the implementation and experiments for ICLR 2023: Efficient Discrete Multi Marginal Optimal Transport Regularization.
Each sub-folder corresponds to a unique experiment in the main paper, and each folder contains the corresponding version of our DEMD codebase.
For current plug-and-play, check out the `demd` folder within, and particularly `demdLoss.py`, `DEMDLayer.py`, and `demdFunc.py`.


### Dependencies
Dependencies are in requirements or Pipfile/locks in each folder. Old packages with security vulnerabilities (requests, tornado)
have been removed from the requirements and pipfiles (see recent commit). __Please take care when installing via these requirements files,
and check that versions of installed packages include security patches__. 


## Reference
Efficient Discrete Multi Marginal Optimal Transport Regularization.  
Ronak Mehta, Jeffery Kline, Vishnu Suresh Lokhande, Glenn Fung, Vikas Singh.  
ICLR 2023, Spotlight.  
[https://openreview.net/forum?id=R98ZfMt-jE](https://openreview.net/forum?id=R98ZfMt-jE)

```
@inproceedings{
mehta2023efficient,
title={Efficient Discrete Multi Marginal Optimal Transport Regularization},
author={Ronak Mehta and Jeffery Kline and Vishnu Suresh Lokhande and Glenn Fung and Vikas Singh},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=R98ZfMt-jE}
}
```
