# bash -i
# conda activate seg 1336 63503 140
# python treinando.py --size 1200 --batch_size 2 --load_type nii.gz2 --images_dir /opt/notebooks/denoise_ct/dataset/v1 --caches_dir /opt/notebooks/denoise_ct/dataset/caches --weights 
python treinando.py --size 1336 \
    --batch_size 2 \
    --load_type nii.gz2 \
    --caches_dir C:\\Users\\emn3\\Documents\\workspace\\seg\\denoise_ct\\dataset\\caches \
    --images_dir C:\\Users\\emn3\\Documents\\workspace\\seg\\denoise_ct\\dataset\\v1  #\    
    # --images_dir C:\\Users\\emn3\\Documents\\workspace\\seg\\denoise_ct\\OrganSegmentations #\
# --weights 
# tensorboard --port 35781  --logdir logs/gradient_tape --load_fast=false  --bind_all

# quando eu uso uma peso default -> funciona bem
# quando eu uso uma parte 50 -> losss funciona bem
# quando eu uso uma parte 1200 -> losss nan

