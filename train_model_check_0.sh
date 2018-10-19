#!/bin/bash
for i in 0 1 2 3 4 5 6 7
do
    python train.py \
        --device_ids=0,1,2,3 \
         --batch-size=128 \
          --fold=$i \
           --workers=20 \
            --lr=0.0003 \
             --n-epochs=200 \
              --loss=lava \
               --requires_grad=True \
                --rop_step=40 \
                 --hem_sample_count=0 \
                  --scheduler=rop \
                   --optim=adam \
                    --start_epoch=0 \
                     --save_best_count=9 \
                      --root=runs/lava \
                       --model=SE_ResNeXt_50
done
