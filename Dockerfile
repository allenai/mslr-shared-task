# The base image, which will be the starting point for the Docker image.
FROM continuumio/miniconda3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# This is the directory that files will be copied into.
# It's also the directory that you'll start in if you connect to the image.
WORKDIR /app

# Update
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get -y autoremove && \
    apt-get clean

# Install some stuff
RUN apt-get -y install git
RUN apt-get -y install unzip

# Make dirs
RUN mkdir /app/data
RUN mkdir /app/evaluator
RUN mkdir /app/models
RUN mkdir /app/output
RUN mkdir /app/ms2

# Copy files
COPY data/ /app/data
COPY evaluator/ /app/evaluator
COPY models/ /app/models
COPY ms2/ /app/ms2

# Create env and activate
COPY environment.yml .
RUN conda env create -n mslr --file environment.yml
SHELL ["conda", "run", "-n", "mslr", "/bin/bash", "-c"]
RUN conda install -c conda-forge jsonnet
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz
COPY setup.py /app/setup.py
RUN python setup.py develop

# Copy evidence inference models
WORKDIR /app/models
RUN wget https://ai2-s2-ms2.s3-us-west-2.amazonaws.com/evidence_inference_models.zip
RUN unzip -q evidence_inference_models.zip

# Copy test data to data dir
WORKDIR /app/data

# Switch working dir
WORKDIR /app