FROM continuumio/miniconda3:4.8.2

RUN conda install -c conda-forge m2crypto \
    && pip install  --no-cache-dir \
        opencv-python==4.2.0.34 \
        opendiamond==10.0.1 \
        protobuf==3.12.4 \
        pyyaml==5.3.1 \
        rekallpy==0.3.2 \
    && conda clean --all -y

COPY . /root/
WORKDIR /root/
RUN python setup.py install
COPY ["fil_stsearch.py", "/usr/local/share/diamond/filters/"]