bash -i
# conda activate seg
pip install IProgress
pip install cython pyyaml==5.1
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install matplotlib
pip install nibabel
python -m pip install tensorflow



# loss sendo calculada mas náo esta conseguindo calcular as metricas para avaliacäo 
# loss náo sendo calculada no final mas agora as metricas estäo sendo calculada
# loss estava sendo usada a meansquared error -> problema para segmentar multiplos Orgaos
# loss SparseCategoricalCrossentropy
# reduzir  learning_rate=1e-4 -> learning_rate=1e-6
# batch_size = 2
# reduzir  crop_size = 256 -> crop_size = 128
# docker -> windows normal

# treinei sem atualizar os pesos e foi de boas
# The experiments were conducted using the Geforce GTX1070
# GPU with 8GB memory. CR-U-Net was constructed by using the
# Tensorflow framework [21] and trained by forward propagation with
# batch size of 2 and learning rate of 1
# −4
# -1
# −10, totaling 20,000 epochs.
# só usa parte do dataset que contem o figado 
 
