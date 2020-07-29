FROM continuumio/miniconda3:4.8.2

RUN conda install -c conda-forge m2crypto \
    && pip install  --no-cache-dir pyyaml opendiamond==10.0.1 \
    && conda clean --all -y

COPY . /root/
WORKDIR /root/
RUN python setup.py install
COPY ["fil_stsearch.py", "/usr/local/share/diamond/filters/"]