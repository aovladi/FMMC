ARG SERVING_BUILD_IMAGE=nvcr.io/nvidia/pytorch:22.01-py3

FROM ${SERVING_BUILD_IMAGE} 

ARG user

RUN useradd -ms /bin/bash $user \
        && apt-get update && apt-get install -y \
        && apt-get clean \
        && pip install torchsummary==1.5.1 webdataset==0.1.40

USER $user

WORKDIR /home/$user

RUN git clone https://github.com/aovladi/FMMC.git

WORKDIR /home/$user/FMMC

LABEL maintainer="Olga_Andreeva"
