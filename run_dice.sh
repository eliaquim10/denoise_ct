# bash -i
# conda activate seg 1336 63503
# python treinando.py --size 1200 --batch_size 2 --load_type nii.gz2 --images_dir /opt/notebooks/denoise_ct/dataset/v1 --caches_dir /opt/notebooks/denoise_ct/dataset/caches --weights 
python treinando.py \
    --size 140 \
    --unet 1 \
    --loss 2 \
    --batch_size 4 \
    --load_type nii.gz \
    --images_dir /home/work/dataset/OrganSegmentations \
     --caches_dir /home/work/dataset/dataset/caches \
     
# tensorboard --port 35781  --logdir logs/gradient_tape --load_fast=false  --bind_all