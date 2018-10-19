#!/bin/bash

for i in 0 1 2 3 4 5 6 7
do
    python train.py \
        --device_ids=0,1 \
         --batch-size=128 \
          --fold=$i \
           --workers=12 \
            --lr=0.0003 \
             --n-epochs=200 \
              --loss=focal_lava \
               --requires_grad=True \
                --rop_step=40 \
                 --hem_sample_count=0 \
                  --scheduler=rop \
                   --optim=adam \
                    --start_epoch=0 \
                     --save_best_count=5 \
                      --root=runs/focal_lava \
                       --model=SE_ResNeXt_50
done

for i in 0 1 2 3 4 5 6 7
do
    python train.py \
        --device_ids=0,1 \
         --batch-size=128 \
          --fold=$i \
           --workers=12 \
            --lr=0.0003 \
             --n-epochs=200 \
              --loss=bce_lava \
               --requires_grad=True \
                --rop_step=40 \
                 --hem_sample_count=0 \
                  --scheduler=rop \
                   --optim=adam \
                    --start_epoch=0 \
                     --save_best_count=5 \
                      --root=runs/bce_lava \
                       --model=SE_ResNeXt_50
done

for i in 0 1 2 3 4 5 6 7
do
    python train.py \
        --device_ids=0,1 \
         --batch-size=128 \
          --fold=$i \
           --workers=12 \
            --lr=0.0003 \
             --n-epochs=200 \
              --loss=focal_jaccard \
               --requires_grad=True \
                --rop_step=40 \
                 --hem_sample_count=0 \
                  --scheduler=rop \
                   --optim=adam \
                    --start_epoch=0 \
                     --save_best_count=5 \
                      --root=runs/focal_jaccard \
                       --model=SE_ResNeXt_50
done

for i in 0 1 2 3 4 5 6 7
do
    python train.py \
        --device_ids=0,1 \
         --batch-size=128 \
          --fold=$i \
           --workers=12 \
            --lr=0.0003 \
             --n-epochs=200 \
              --loss=focal_lava_jaccard \
               --requires_grad=True \
                --rop_step=40 \
                 --hem_sample_count=0 \
                  --scheduler=rop \
                   --optim=adam \
                    --start_epoch=0 \
                     --save_best_count=5 \
                      --root=runs/focal_lava_jaccard \
                       --model=SE_ResNeXt_50
done

for i in 0 1 2 3 4 5 6 7
do
    python train.py \
        --device_ids=0,1 \
         --batch-size=128 \
          --fold=$i \
           --workers=12 \
            --lr=0.0003 \
             --n-epochs=200 \
              --loss=lava \
               --requires_grad=True \
                --rop_step=40 \
                 --hem_sample_count=0 \
                  --scheduler=rop \
                   --optim=adam \
                    --start_epoch=0 \
                     --save_best_count=5 \
                      --root=runs/lava \
                       --model=SE_ResNeXt_50
done

for i in 0 1 2 3 4 5 6 7
do
    python train.py \
        --device_ids=0,1 \
         --batch-size=128 \
          --fold=$i \
           --workers=12 \
            --lr=0.0003 \
             --n-epochs=200 \
              --loss=focal \
               --requires_grad=True \
                --rop_step=40 \
                 --hem_sample_count=0 \
                  --scheduler=rop \
                   --optim=adam \
                    --start_epoch=0 \
                     --save_best_count=5 \
                      --root=runs/focal \
                       --model=SE_ResNeXt_50
done

for i in 0 1 2 3 4 5 6 7
do
    python train.py \
        --device_ids=0,1 \
         --batch-size=128 \
          --fold=$i \
           --workers=12 \
            --lr=0.0003 \
             --n-epochs=200 \
              --loss=bce_jaccard \
               --requires_grad=True \
                --rop_step=40 \
                 --hem_sample_count=0 \
                  --scheduler=rop \
                   --optim=adam \
                    --start_epoch=0 \
                     --save_best_count=5 \
                      --root=runs/bce_jaccard \
                       --model=SE_ResNeXt_50
done


:'
    python train.py \
        --device_ids=0,1 \
         --batch-size=16 \
          --fold=$i \
           --workers=12 \
            --lr=0.0003 \
             --n-epochs=300 \
              --loss=focal_lava \
               --rop_step=8 \
                --requires_grad=True \
                 --start_epoch=70 \
                  --early_stop_patience=50 \
                   --hem_sample_count=0 \
                    --scheduler=rop \
                     --optim=adam \
                      --model=SE_ResNeXt_50


    python train.py \
        --device_ids=0,1 \
         --batch-size=64 \
          --fold=$i \
           --workers=12 \
            --lr=0.01 \
             --n-epochs=1000 \
              --loss=focal_lava \
               --requires_grad=True \
                --start_epoch=200 \
                  --scheduler=co \
                   --optim=sgd \
                    --save_best_count=100 \
                     --model=SE_ResNeXt_50
'
