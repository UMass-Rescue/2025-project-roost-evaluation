FROM python:3.10-slim

RUN apt-get update && apt-get install -y iputils-ping && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY evaluate.py /build/
COPY tests/ /build/tests/
COPY resources/images /build/resources/images

RUN pip install requests  

# Set default image directory, can be overridden at runtime
ENV IMAGE_INPUT_DIR=/build/resources/images

CMD ["python", "evaluate.py"]
