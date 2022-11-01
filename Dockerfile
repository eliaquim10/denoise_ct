FROM continuumio/miniconda3

WORKDIR /opt/notebooks

# Create the environment:
#clona o repositorio para dentro do container docker
# RUN git clone https://github.com/ericsujw/InstColorization .

# instala pacotes do python e para o sistema
RUN apt update -y 
RUN apt upgrade -y 
RUN bash -i 
RUN conda create -n seg python=3.6
RUN bash -i && activate seg
# RUN pip install IProgress
# RUN pip install cython pyyaml==5.1
# RUN conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# RUN python3 -m pip install tensorflow
RUN apt install gcc -y
RUN apt install unzip -y
RUN apt install zip -y

ENTRYPOINT ["python"]
# /root/.torch/fvcore_cache/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl
# ["python", "download.py"]
# docker run -i --name recolor --gpus all continuumio/miniconda3
# docker run -i --name color --gpus all colorization
# docker cp checkpoints/ color:/opt/notebooks/
# docker build -t colorization .

# docker run -i --name et --gpus all entity_relation
# docker system prune

# docker start super -p 35781:35781 -p 37855:37855
# docker build -t super .
# docker run -i --name super -p 35781:35781 -p 37855:37855 --gpus all continuumio/miniconda3
# docker run -i --name super -p 35781:35781 -p 37855:37855 --gpus all super