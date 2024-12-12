#!/bin/bash

# python train_DeepChIP.py -d paired_sharded_dataset2 -o /home/alex/ChIP_ENCODE/Trainedmodels/model_siameseinception4 \
#     -arch SiameseInceptionNetwork --paired -mt 1024 -mv 256 -p 10 -v

# python predict_DeepChIP.py --config /home/alex/ChIP_ENCODE/Trainedmodels/model_siameseinception4/config.json

# python train_DeepChIP.py -d /home/alex/shared_folder/CTCF/paired_sharded_dataset \
#     -o /home/alex/ChIP_ENCODE/Trainedmodels/CTCF/model_siameseinception \
#     -arch SiameseInceptionNetwork --paired -mt 1024 -mv 256 -p 10 -v

# python predict_DeepChIP.py --config /home/alex/ChIP_ENCODE/Trainedmodels/CTCF/model_siameseinception/config.json

python train_DeepChIP.py -d /home/alex/shared_folder/ChIP_ENCODE/paired_sharded_dataset3 -o /home/alex/ChIP_ENCODE/Trainedmodels/model_siameseinception5 \
    -arch SiameseInceptionNetwork --paired -mt 1024 -mv 256 -p 10 -v

python predict_DeepChIP.py --config /home/alex/ChIP_ENCODE/Trainedmodels/model_siameseinception5/config.json