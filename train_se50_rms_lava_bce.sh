#!/bin/bash

for i in 0 1 2 3 4 5 6 7
do
    python train.py \
     --device_ids=0,1 \
      --batch-size=64 \
       --fold=$i \
        --workers=12 \
         --lr=0.0003 \
          --n-epochs=100 \
           --loss=bce_lava \
            --requires_grad=False \
             --rop_step=6 \
              --hem_sample_count=0 \
               --optim=rmsprop \
                --start_epoch=0 \
                 --model=SE_ResNeXt_50

    python train.py \
        --device_ids=0,1 \
         --batch-size=64 \
          --fold=$i \
           --workers=12 \
            --lr=0.0003 \
             --n-epochs=300 \
              --loss=bce_lava \
               --rop_step=8 \
                --requires_grad=True \
                 --start_epoch=100 \
                  --early_stop_patience=150 \
                   --hem_sample_count=0 \
                    --scheduler=rop \
                     --optim=rmsprop \
                      --model=SE_ResNeXt_50

done
