FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
  
ENV TZ Asia/Seoul
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install \
    python3 python3-pip python3-dev \
    jupyter-notebook \
    git ssh vim

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN echo "root:root" | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

RUN pip3 install --upgrade pip
RUN pip3 install setuptools

RUN apt-get update && apt-get install -y locales
RUN locale-gen ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8

WORKDIR /workspace
ADD . .

RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.password = 'key'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py

ENV PYTHONPATH $PYTHONPATH:/workspace

RUN chmod -R a+w /workspace
