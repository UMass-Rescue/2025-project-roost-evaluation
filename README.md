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