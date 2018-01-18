#!/bin/bash

python main.py --dataset OULU --y_dim 6 --checkpoint_dir controller_oulu_50z --save_dir oulu_50z --batch_size 48 --is_stage_one True --epoch 50
python main.py --dataset OULU --y_dim 6 --checkpoint_dir oulu_50z --save_dir oulu_50z2 --batch_size 48 --is_stage_one False --epoch 50
