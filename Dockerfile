FROM python:3.10-slim

RUN apt-get update && apt-get install -y iputils-ping && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY evaluate.py /build/
COPY tests/ /build/tests/
COPY resources/images /build/resources/images

RUN pip install requests numpy scipy psycopg2-binary tqdm 

# Set default image directory, can be overridden at runtime
ENV IMAGE_INPUT_DIR=/build/resources/images
ENV PYTHONPATH=/build

CMD ["python", "evaluate.py"]
