# Codes-for-WSDM-CUP-Music-Rec-1st-place-solution

This is the corresponding codes for WSDM CUP 2018 Music Recommendation Challenge's 1st place solution.


Please create following folders before testing:

- input/training/source_data/

- input/training/temporal_data/

- input/validation/source_data/

- input/validation/temporal_data/

- temp_nn/

- submission/

Put the data in the folder "source_data", then run script/run.sh, features will be extracted. For validation, you need to prepare data by hand, and create a "test_label.csv" file with "target" field.


The hyper-parameters is recorded in lgb_record.csv and nn_record.csv, you can try it directly. If everything is right, you should be able to get 0.744+ with LightGBM, and 0.742+ with 30-ensemble of NNs. 0.6 * LightGBM + 0.4 * NN should be able to get you ~0.749.


The code is tested on a small part of the data under python 2.7, if you find any bug, please contract me under the topic on Kaggle.

The versions of dependencies:

- pandas: 0.20.1

- sklearn: 0.18.1

- keras: 2.0.4

- lightgbm: 0.1

- numpy: 1.12.1

- scipy: 0.19.0

- Tensorflow 1.0.1
