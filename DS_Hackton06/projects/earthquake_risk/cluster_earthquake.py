import argparse, os, joblib
import pandas as pd
from sklearn.cluster import KMeans
from src.utils.logger import get_logger
logger = get_logger('cluster_earthquake')
def run(data_dir, output):
    import glob
    csvs = glob.glob(os.path.join(data_dir,'**','*.csv'), recursive=True)
    if not csvs:
        raise FileNotFoundError('No CSVs found in '+data_dir)
    df = pd.read_csv(csvs[0], low_memory=False)
    numcols = df.select_dtypes(include=['float','int']).columns.tolist()
    if not numcols:
        raise RuntimeError('No numeric cols')
    X = df[numcols].fillna(0)
    model = KMeans(n_clusters=5, random_state=42)
    model.fit(X)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    joblib.dump(model, output)
    logger.info('Saved earthquake clustering model to '+output)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/raw/earthquake')
    p.add_argument('--output', default='models/earthquake_model.pkl')
    args=p.parse_args()
    run(args.data_dir, args.output)
