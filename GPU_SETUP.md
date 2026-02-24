# GPU Setup Guide

Run evaluation with NVIDIA GPU acceleration for faster CLIP embeddings.

**Default is CPU** (works on macOS, CI, no GPU). Follow this guide to enable GPU on Linux.

**Platform:** Linux with NVIDIA GPU only (macOS not supported - Docker can't access GPU)

**No Docker access?** See [NATIVE_GPU_SETUP.md](NATIVE_GPU_SETUP.md) to run natively without containers.

---

## Switch from CPU to GPU

Do these in order:

1. **Prerequisites** – `nvidia-smi` works, nvidia-container-toolkit installed (see below).

2. **docker-compose.yaml** – Uncomment the `deploy` block under `hma-app` (lines 72–78):
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```

3. **Build with CUDA** – Match `PYTORCH_CUDA` to your `nvidia-smi` CUDA version:

   | CUDA | PYTORCH_CUDA |
   |------|--------------|
   | 12.8+ | cu128 |
   | 12.6–12.7 | cu126 |
   | 12.4–12.5 | cu124 |
   | 11.x | cu118 |
   ```bash
   PYTORCH_CUDA=cu128 docker compose build --no-cache hma-app hma-migrations
   docker compose up -d
   ```

4. **Verify** – `docker compose exec hma-app python -c "import torch; print(torch.cuda.is_available())"` should print `True`.

**To switch back to CPU:** Comment out the `deploy` block again and rebuild with `docker compose build --no-cache hma-app hma-migrations` (no PYTORCH_CUDA; uses cpu).

---

## Prerequisites

### 1. Check NVIDIA GPU and Drivers
```bash
nvidia-smi
```
If this fails, install NVIDIA drivers and reboot.

### 2. Install nvidia-container-toolkit
```bash
# Check if installed
nvidia-ctk --version

# If not installed (Ubuntu/Debian):
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Test Docker GPU Access
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
Should show your GPU info. If this fails, GPU won't work in containers.

### 4. Match PyTorch to Your GPU
Run `nvidia-smi` and check **CUDA Version** (top-right). Set `PYTORCH_CUDA` when building:

| CUDA | PYTORCH_CUDA |
|------|--------------|
| 12.8+ | cu128 |
| 12.6–12.7 | cu126 |
| 12.4–12.5 | cu124 |
| 11.x | cu118 |

```bash
PYTORCH_CUDA=cu128 docker compose build --no-cache   # use cu124 etc. to match your GPU
```

## Build and Run (GPU)

After completing "Switch from CPU to GPU" above:

```bash
docker network create shared-hma-network
PYTORCH_CUDA=cu128 docker compose build --no-cache   # match your nvidia-smi
docker compose up -d
```

### Verify GPU Works
```bash
# Check GPU visible
docker compose exec hma-app nvidia-smi

# Check PyTorch CUDA
docker compose exec hma-app python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Run Tests with GPU

```bash
# Monitor GPU in one terminal
watch -n 1 nvidia-smi

# Run tests in another
time docker compose run --rm -e EVAL_MODE=test evaluation
```

## Troubleshooting

**GPU not available in container:**
- Verify `nvidia-smi` works on host
- Verify Docker GPU test works (step 3 above)
- Check GPU config is uncommented in docker-compose.yaml
- Restart Docker: `sudo systemctl restart docker`

**"no kernel image" / CUDA mismatch:** Rebuild with correct `PYTORCH_CUDA` (step 4).

**Force CPU (e.g. GPU driver issues):** Comment out the `deploy` block in docker-compose.yaml, then rebuild without `PYTORCH_CUDA` (uses cpu).

**On macOS:** GPU not supported. Use default (CPU). No changes needed.
