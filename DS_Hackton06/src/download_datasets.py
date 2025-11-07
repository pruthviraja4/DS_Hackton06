import os, subprocess, argparse
DATASETS = {
    "flood": "naiyakhalid/flood-prediction-dataset",
    "sen12flood": "rhythmroy/sen12flood-flood-detection-dataset",
    "air_quality_india": "rohanrao/air-quality-data-in-india",
    "us_accidents": "sobhanmoosavi/us-accidents",
    "earthquake": "warcoder/earthquake-dataset"
}
parser = argparse.ArgumentParser()
parser.add_argument('--local', action='store_true', help='Skip Kaggle; expect files in data/raw/')
parser.add_argument('--datasets', nargs='*', help='Subset of datasets to download', default=list(DATASETS.keys()))
args = parser.parse_args()
os.makedirs('data/raw', exist_ok=True)
if args.local:
    print('Local mode: please place dataset files in data/raw/<name>/')
    exit(0)
for key in args.datasets:
    if key not in DATASETS:
        print('Unknown dataset', key)
        continue
    slug = DATASETS[key]
    outdir = os.path.join('data','raw', key)
    os.makedirs(outdir, exist_ok=True)
    print(f'Downloading {slug} to {outdir} ...')
    # Use kaggle CLI: kaggle datasets download -d <slug> -p <outdir> --unzip
    subprocess.run(['kaggle', 'datasets', 'download', '-d', slug, '-p', outdir, '--unzip'])
print('All downloads attempted. Check data/raw/')
