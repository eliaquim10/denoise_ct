bash -i
# conda activate seg
pip install IProgress
pip install cython pyyaml==5.1
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install matplotlib
pip install nibabel
python -m pip install tensorflow