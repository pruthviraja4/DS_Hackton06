# Urban AI Ops Data Science Suite (GPU, Full Datasets)

This project scaffold prepares end-to-end pipelines for five Kaggle datasets:
- Flood Prediction: naiaykhalid/flood-prediction-dataset
- SEN12FLOOD Segmentation: rhythmroy/sen12flood-flood-detection-dataset
- Air Quality in India: rohanrao/air-quality-data-in-india
- US Accidents: sobhanmoosavi/us-accidents
- Earthquake Events: warcoder/earthquake-dataset

**You chose: GPU + Full datasets** — the project is configured to train a U-Net on GPU (PyTorch + CUDA)
and runs tabular models on CPU/GPU where applicable.

## Important pre-reqs (host machine)
- NVIDIA GPU with CUDA 11.7+ and nvidia-docker2 installed (for Docker GPU runtime)
- Docker + NVIDIA Container Toolkit (nvidia-docker)
- Kaggle CLI configured (place kaggle.json in ~/.kaggle/)
- Recommended: >= 50 GB disk for all datasets

## Quick start
1. (Optional) Inspect the repo:
   ```
   unzip data_science_project_gpu_full.zip -d ds_project
   cd ds_project
   ```
2. Install Python dependencies (for local dev):
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Download datasets (full):
   ```
   python src/download_datasets.py
   ```
   This will download and unzip all Kaggle datasets to `data/raw/<dataset>/`.
4. Prepare GPU Docker (optional — recommended for U-Net training):
   ```
   docker build -t ds-gpu:latest -f docker/Dockerfile.gpu .
   docker run --gpus all -it --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models ds-gpu:latest bash
   ```
5. Train U-Net (inside container or locally with CUDA-enabled PyTorch):
   ```
   python projects/sen12flood_segmentation/train_unet.py --data-dir data/raw/sen12flood-flood-detection-dataset --epochs 30 --batch-size 8 --output models/unet_flood.pt
   ```
6. Train tabular baselines:
   ```
   python projects/flood_prediction/train_flood.py --data-dir data/raw/flood-prediction-dataset --output models/flood_baseline.pkl
   python projects/air_quality/forecast_air_quality.py --data-dir data/raw/air-quality-data-in-india --output models/air_quality_model.pkl
   python projects/traffic_accidents/train_traffic.py --data-dir data/raw/us-accidents --output models/traffic_model.pkl
   python projects/earthquake_risk/cluster_earthquake.py --data-dir data/raw/earthquake-dataset --output models/earthquake_model.pkl
   ```
7. Serve predictions with FastAPI:
   ```
   uvicorn api.main:app --reload
   ```

## Notes and caveats
- The repo includes **download scripts** but DOES NOT bundle Kaggle data due to license and size.
- The Dockerfile.gpu uses a CUDA-enabled base image and installs the appropriate PyTorch build.
- If you need me to attempt to download the datasets into the archive (I cannot fetch them from here), run the download script on your machine after unzipping.

