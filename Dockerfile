FROM quay.io/jupyter/base-notebook
USER root
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libpq-dev \
    gcc \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir opencv-python imgbeddings pillow psycopg2 pandas