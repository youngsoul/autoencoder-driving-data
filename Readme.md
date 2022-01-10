# Using AutoEncoders to reduce dimensionality in self-driving example

Example of using an AutoEncoder for dimensionality reduction.

## Overview

Collect driving data ( left, right, straight ) which contains 250 features and the target for driving instruction.

The training process collects:

* 150 rows

* 250 features

* 1 Target

Can we reduce the 250 features to a smaller number and still successfully drive the simulated car?

## Install

```shell
pip install sklearn tensorflow pandas matplotlib
```

## Execute

---
* Run `01_training.py` to collect training data.

The shape of the data will be 150 rows by 251 features, where feature[0] is the target value.

---

* Train the model or AutoEncode then Train

If you want to autoencode to reduce the dimensionality:

Run `autoencode_dim_reduction.py`.

This will by default reduce the 250 features to 32 and assume to use `training.csv`.  Both of these options can be changed.  The resulting reduced dimension training data will be saved in file like: `training_[dims].csv

Run `02_model_training.py` and provide the training data.  use the `--training-data` option to specify the training data file.  Default is `training.py`, but if you autoencoded, then use the generated autoencoded training file name.

---

* Drive by model

Run `03_drive_by_model.py`.  If you want to use autoencoding, use the `--autoencode` option.  This will cause the script to read `encoder_scaler.pkl` and `encoder_model.h5` and use those during the simulated driving along with the `best_driving_model.sav`.



