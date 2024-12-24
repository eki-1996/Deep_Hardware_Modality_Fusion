CUDA_VISIBLE_DEVICES=0 python test.py \
  --input-channels-list 3,1,1,1,1,1 \
  --backbone resnet_adv\
  --lr 0.007 \
  --workers 8 \
  --epochs 500 \
  --batch-size 8 \
  --ratio 3 \
  --gpu-ids 0 \
  --weight-decay 0.0005 \
  --checkname HardFuse \
  --eval-interval 1 \
  --loss-type ce \
  --dataset multimodal_dataset \
  --dataset-modality rgb,aolp,dolp,nir,image_000,image_045,image_090,image_135 \
  --use-modality rgb,nir,image_000,image_045,image_090,image_135 \
  --use-pretrained-resnet \
  --use-hardware-modality-fusion \
  --fusion-kernel-size 1 \
  --fused-out-dim 3 \
  --pth-path path to your model


# CUDA_VISIBLE_DEVICES=0 python test.py \
#   --input-channels-list 3,1 \
#   --backbone resnet_adv\
#   --lr 0.007 \
#   --workers 8 \
#   --epochs 500 \
#   --batch-size 8 \
#   --ratio 3 \
#   --gpu-ids 0 \
#   --weight-decay 0.0005 \
#   --checkname HardFuse \
#   --eval-interval 1 \
#   --loss-type ce \
#   --dataset rgb_thermal_dataset \
#   --dataset-modality rgb,thermal \
#   --use-modality rgb,thermal \
#   --use-pretrained-resnet \
#   --use-hardware-modality-fusion \
#   --fusion-kernel-size 1 \
#   --fused-out-dim 3 \
#   --pth-path path to your model