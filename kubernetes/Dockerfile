FROM gcr.io/tensorflow/tensorflow:latest-py3

RUN apt update && apt install -y g++ python3-gdal gdal-bin
RUN pip install joblib scikit-image

WORKDIR "/content"
