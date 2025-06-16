FROM python:3.10-slim

RUN apt-get update && apt-get install -y iputils-ping && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY evaluate.py /build/
COPY resources/images /build/resources/images

RUN pip install requests  

CMD ["python", "evaluate.py"]
