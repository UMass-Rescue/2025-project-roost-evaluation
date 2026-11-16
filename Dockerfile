FROM python:3.10-slim

RUN apt-get update && apt-get install -y iputils-ping && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements.txt /build/
RUN pip install --no-cache-dir -r requirements.txt

COPY hma/hma_client.py /build/hma/hma_client.py
COPY hma/logging_utils.py /build/hma/logging_utils.py
COPY hma/results_utils.py /build/hma/results_utils.py
COPY hma/path_id_store.py /build/hma/path_id_store.py
COPY hma/metrics/ /build/hma/metrics/
COPY hma/image/evaluate.py /build/hma/image/evaluate.py
COPY hma/image/series_labels_utils.py /build/hma/image/series_labels_utils.py
COPY hma/image/tests/ /build/hma/image/tests/
COPY hma/image/resources/ /build/hma/image/resources/

# Set default image directory, can be overridden at runtime
ENV IMAGE_INPUT_DIR=/build/hma/image/resources/images
ENV PYTHONPATH=/build

CMD ["python", "hma/image/evaluate.py"]
