# Codes-for-WSDM-CUP-Music-Rec-1st-place-solution

This is the corresponding codes for WSDM CUP 2018 Music Recommendation Challenge's 1st place solution.


Please create following folders before testing:

input/training/source_data

input/training/temporal_data

input/validation/source_data

input/validation/temporal_data


Put the data in the folder "source_data", then run script/run.sh, features will be extracted. For validation, you need to prepare data by hand, and create a "test_label.csv" file with "target" field.


The hyper-parameters is recorded in lgb_record.csv and nn_record.csv, you can try it directly. If everything is right, you should be able to get 0.744+ with LightGBM, and 0.742+ with 30-ensemble of NNs.


The code is tested on a small part of the data under python 2.7, if you find any bug, please contract me by lystdo@163.com.

