#!/bin/bash

for i in 0 1 2 3 4 5 6 7
do

    python train.py \
        --device_ids=0,1 \
         --batch-size=64 \
          --fold=$i \
           --workers=12 \
            --lr=0.01 \
             --n-epochs=1000 \
              --loss=focal_lava \
               --requires_grad=True \
                --start_epoch=150 \
                  --scheduler=co \
                   --optim=sgd \
                    --save_best_count=100 \
                     --model=SE_ResNeXt_50

done
