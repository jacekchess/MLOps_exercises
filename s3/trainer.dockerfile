# Base image
FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY s2/requirements.txt requirements.txt
COPY s2/setup.py setup.py
COPY s2/src/ src/
COPY s2/data/ data/
COPY s2/reports/figures/ reports/figures/
COPY s2/models/ models/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]