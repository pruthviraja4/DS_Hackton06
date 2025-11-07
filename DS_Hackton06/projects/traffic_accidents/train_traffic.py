import argparse, os, joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger
logger = get_logger('train_traffic')
def train(data_dir, output):
    import glob
    csvs = glob.glob(os.path.join(data_dir,'**','*.csv'), recursive=True)
    if not csvs:
        raise FileNotFoundError('No CSVs found in '+data_dir)
    df = pd.read_csv(csvs[0], low_memory=False)
    # naive: create binary target if 'Severity' exists else use first numeric
    if 'Severity' in df.columns:
        df = df.dropna(subset=['Severity'])
        df['target'] = (df['Severity']>2).astype(int)
    else:
        numcols = df.select_dtypes(include=['float','int']).columns.tolist()
        if not numcols:
            raise RuntimeError('No numeric cols')
        df['target'] = (df[numcols[0]]>df[numcols[0]].median()).astype(int)
    features = df.select_dtypes(include=['float','int']).drop(columns=['target'])
    X_train,X_test,y_train,y_test = train_test_split(features, df['target'], test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train.fillna(0), y_train)
    preds = model.predict(X_test.fillna(0))
    from src.utils.metrics import binary_metrics
    print(binary_metrics(y_test, preds))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    joblib.dump(model, output)
    logger.info('Saved model to '+output)
if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/raw/us_accidents')
    p.add_argument('--output', default='models/traffic_model.pkl')
    args=p.parse_args()
    train(args.data_dir, args.output)
