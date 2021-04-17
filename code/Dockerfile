# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:20.12-py3

COPY requirements.txt .
RUN python -m pip install -r requirements.txt --no-cache-dir #to avoid caching

WORKDIR /opt/lstm-autoencoder-seq2seq/

CMD /bin/bash
