# Fair_HIN
  * Source code for ICWSM 2021 paper [**Fair Representation Learning for Heterogeneous Information Networks**](http://arxiv.org/abs/2104.08769)


## Requirements
  * Python3
  * tensorflow >=1.12.0
  * cython >=0.29.2
  * pyyaml >= 3.12
  * g++ >= 5.4.0
  * openmp >= 4.0


## Compile
GraphSAINT have a cython module which need compilation before training can start. Compile the module by running the following command under the ```GNNs``` directory:
```bash
python graphsaint/setup.py build_ext --inplace
```

## Usage
* adversarial
```bash
python tune_para_movie_adversarial.py
```

* GNN-demographic-parity
```bash
python tune_para_movie_fair_aware_loss.py --loss_type dp
```

* GNN-equal-opportunity
```bash
python tune_para_movie_fair_aware_loss.py --loss_type eo
```

* GNN
```bash
python tune_para_movie_gnn_base.py
```


## Parse Result
```bash
python read_result.py --method [gnn_base,fair_loss,adv] --criterion [eo,dp] --fair_level [low,med,high] --dataset ml
```