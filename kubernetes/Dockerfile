FROM gcr.io/tensorflow/tensorflow

RUN apt update && apt install -y g++ python-gdal gdal-bin
RUN pip install joblib scikit-image

WORKDIR "/content"
