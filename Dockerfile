FROM python:3.10-slim

RUN apt-get update && apt-get install -y iputils-ping && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements.txt /build/
RUN pip install --no-cache-dir -r requirements.txt

COPY hma/evaluate.py /build/hma/evaluate.py
COPY hma/series_labels_utils.py /build/hma/series_labels_utils.py
COPY hma/tests/ /build/hma/tests/
COPY hma/metrics/ /build/hma/metrics/
COPY hma/resources/images /build/hma/resources/images
COPY hma/resources/labels /build/hma/resources/labels

# Set default image directory, can be overridden at runtime
ENV IMAGE_INPUT_DIR=/build/hma/resources/images
ENV PYTHONPATH=/build

CMD ["python", "hma/evaluate.py"]
