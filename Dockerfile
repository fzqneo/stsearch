FROM continuumio/miniconda3:4.8.2

RUN apt-get update --fix-missing \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libcurl4-openssl-dev \
        libssl-dev \
        python3-dev \
    && apt-get clean

RUN  conda install -c conda-forge m2crypto \
    && pip install  --no-cache-dir \
        logzero \
        mmh3 \
        opencv-python==4.2.0.34 \
        opendiamond[DIAMONDD]==10.1.0 \
        protobuf==3.12.4 \
        pycurl \
        pyyaml==5.3.1 \
        rekallpy==0.3.2 \
        simplejson \
    && conda clean --all -y

COPY . /root/
WORKDIR /root/
RUN pip install --no-cache-dir .
COPY ["fil_stsearch.py", "/usr/local/share/diamond/filters/"]