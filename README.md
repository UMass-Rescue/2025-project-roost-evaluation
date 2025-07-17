# 2025 Project Roost Evaluation

## Steps for Running HMA with CLIP Extensions

The HMA CLIP demo repository is available at [2025-project-hma-clip-demo](https://github.com/UMass-Rescue/2025-project-hma-clip-demo).

To get HMA CLIP up and running, follow these steps:

### Prerequisites

- Docker and Docker Compose must be installed on your machine.

### Setup and Run

1. **Clone the Repository**

   ```bash
   git clone [your-repository-url]
   cd [repository-name]
   ```

2. **Launch the Services**

   Use Docker Compose to build and start the services defined in the `docker-compose.yml` file:

   ```bash
   docker compose up --build
   ```

   This command builds the Docker image and starts the services, including the application and the database.

   You can configure the PostgreSQL database and other environment variables for HMA CLIP in `docker-compose.yml` and `omm_config.py`.

---

## Instructions to Run the Evaluation Pipeline

1. **Ensure the Latest HMA Version**

   In the Dockerfile of HMA CLIP, ensure the latest version of HMA is pulled. If not, update line 1 in the Dockerfile to:

   ```dockerfile
   FROM ghcr.io/facebook/threatexchange/hma:1.0.17
   ```

   Alternatively, use the `hma:latest` tag.

2. **Use an External Network**

   Before running HMA CLIP, create the shared network:

   - Create a Docker network called `shared-hma-network` by running the following command on your terminal:

     ```bash
     docker network create shared-hma-network
     ```

3. **Run the Evaluation**

   Ensure HMA CLIP is running, then execute the evaluation pipeline using the following command:

   ```bash
   docker-compose up --build
   ```
