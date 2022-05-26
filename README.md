# Notebooks on sequence to sequence modelling

This demo is tested with python version 3.8.13.

## Quick Start

Create and activate a new virtual environment (`virtualenv` or `conda`).
For example, using `conda`:

```bash
$conda create --name seqdemo python=3.8.13
$conda activate seqdemo
```

Now, install the required packages.
* To reproduce the demo only
  ```bash
  $pip install -r demo_requirements.txt
  ```
* To reproducing the model training (GPU is recommended), additional packages are needed
  ```bash
  $pip install -r full_requirements.txt
  ```

The demo can be found in `seq2seq.ipynb`.

## Contributors

*  Haotian Jiang (NUS)
*  Qianxiao Li (NUS)