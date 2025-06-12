# 2025-project-roost-evaluation
# Steps for running HMA with CLIP extensions

The hma demo repo is at https://github.com/UMass-Rescue/2025-project-hma-clip-demo.

To get the HMA up and running, follow these steps:

### Prerequisites

- Docker and Docker Compose installed on your machine.

### Setup and Run

1. **Clone the Repository**

```bash
   git clone [your-repository-url]
   cd [repository-name]
```

2. **Launch the Services**

Use Docker Compose to build and start the services defined in the `docker-compose.yml`:

```bash
    docker compose up --build
```

This command builds the Docker image and starts the services defined, including the application and the database.

You can set the postgres db and other env variables for hma-demo in docker-compose.yaml and omm_config.py


### Instructions to run evaluation pipeline
Ensure in the dockerfile of hma, the latest version of hma is pulled. If not, change line 1 in Dockerfile to FROM ghcr.io/facebook/threatexchange/hma:1.0.17 or hma:latest tag.

Before running hma, you need to change it to use an external network 

1) create a docker network called shared-hma-network by running below command
docker network create shared-hma-network

2) change the networks in docker compose to these lines 
networks:
  shared-hma-network:
    external: true

under services: also change network to shared-hma-network

3) Make sure hma is running and then run evaluation using command docker-compose up
