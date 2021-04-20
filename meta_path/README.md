# Requirements
  * Python3
  * Pytorch 


# Compile
  * We have a c++ file which need compilation before training can start. Compile the module by running the following command under the ```meta_path``` directory:
  ```bash
  make
  ```


# Usage
  * balance-data
  ```bash
  python tune_para_movie_balance.py
  ```

  * M2V
  ```bash
  python tune_para_movie.py --method default
  ```

  * M2V + fair-sampling
  ```bash
  python tune_para_movie.py --method bias
  ```

  * M2V + projection
  ```bash
  python tune_para_movie_projection.py --method default
  ```

  * M2V + fair-sampling + projection
  ```bash
  python tune_para_movie_projection.py --method bias
  ```

# Parse Result
  ```bash
  python read_result.py --method [m2v_balance,m2v_default,m2v_bias,m2v_default_projection,m2v_bias_projection] --criterion [eo,dp] --fair_level [low,med,high] --dataset ml
  ```

