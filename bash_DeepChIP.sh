#!/bin/bash

python train_DeepChIP.py -d paired_sharded_dataset2 -o /home/alex/ChIP_ENCODE/Trainedmodels/model_siameseinception4 \
    -arch SiameseInceptionNetwork --paired -mt 1024 -mv 256 -p 10 -v

python predict_DeepChIP.py --config /home/alex/ChIP_ENCODE/Trainedmodels/model_siameseinception4/config.json