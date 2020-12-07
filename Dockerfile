FROM continuumio/miniconda3:4.8.2

RUN apt-get update --fix-missing \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libcurl4-openssl-dev \
        libgl1-mesa-glx \
        libssl-dev \
        python3-dev \
        unzip \
    && apt-get clean

RUN  conda install -c conda-forge m2crypto \
    && pip install  --no-cache-dir \
        filterpy==1.4.5 \
        logzero \
        mmh3 \
        opencv-contrib-python>=4.2.0.0 \
        opendiamond[DIAMONDD]==10.1.0 \
        protobuf==3.12.4 \
        pycurl \
        pyyaml==5.3.1 \
        rekallpy==0.3.2 \
        scikit-image==0.17.2 \
        scikit-learn==0.23.2 \
        simplejson \
    && conda clean --all -y

WORKDIR /root/
RUN wget https://github.com/fzqneo/stsearch/releases/download/20201204/frcnn_cache.zip \
    && unzip frcnn_cache.zip \
    && rm frcnn_cache.zip 

COPY . /root/
RUN pip install --no-cache-dir . 

COPY ["fil_stsearch.py", "/usr/local/share/diamond/filters/"]