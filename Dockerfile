FROM python:3.10-slim

RUN apt-get update && apt-get install -y iputils-ping && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements.txt /build/
RUN pip install --no-cache-dir -r requirements.txt

COPY evaluate.py /build/
COPY tests/ /build/tests/
COPY metrics/ /build/metrics/
COPY resources/images /build/resources/images
COPY resources/labels /build/resources/labels 

# Set default image directory, can be overridden at runtime
ENV IMAGE_INPUT_DIR=/build/resources/images
ENV PYTHONPATH=/build

CMD ["python", "evaluate.py"]
