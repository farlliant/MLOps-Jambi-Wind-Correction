"""Safely add the LK10 native MLflow service to the existing LK09 compose YAML.

Does not remove/edit api, db, existing network, or persistent volume.
"""
from pathlib import Path

root = Path(__file__).resolve().parent
compose = root / "docker-compose.yaml"
if not compose.is_file():
    raise SystemExit("[FAIL] docker-compose.yaml missing; use the merged LK09 repository")
text = compose.read_text(encoding="utf-8")
if "\n  model-serving:\n" in text:
    raise SystemExit("[NOTE] model-serving already exists. No changes made.")
anchor = "\nvolumes:\n"
if text.count(anchor) != 1 or "\n  api:\n" not in text or "\n  db:\n" not in text:
    raise SystemExit("[FAIL] Unexpected LK09 Compose layout; refusing an unsafe edit")
block = '''
  # LK10: built separately with `mlflow models build-docker`.
  # Three replicas share a model image; each gets a random loopback host port.
  model-serving:
    image: jambi-wind-lk10-mlflow:latest
    restart: unless-stopped
    environment:
      MLFLOW_MODELS_WORKERS: "1"
    networks:
      - wind-network
    ports:
      - target: 8080
        host_ip: 127.0.0.1
        protocol: tcp
    deploy:
      replicas: 3
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/ping', timeout=3)"]
      interval: 15s
      timeout: 5s
      retries: 20
      start_period: 180s
'''
compose.write_text(text.replace(anchor, "\n" + block + anchor, 1), encoding="utf-8")
gitignore = root / ".gitignore"
ig = gitignore.read_text(encoding="utf-8")
if ".lk10/" not in ig.splitlines():
    gitignore.write_text(ig.rstrip() + "\n\n# LK10: downloaded MLflow model (not source code)\n.lk10/\n", encoding="utf-8")
print("[PASS] Added model-serving with deploy.replicas: 3")
print("[PASS] Preserved LK09 api/db/network/volume services")
print("[PASS] Ignored locally downloaded model artifacts")
