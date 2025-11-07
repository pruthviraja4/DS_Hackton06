import argparse, os, joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger
logger = get_logger('train_flood')
def train(data_dir, output):
    # find first csv
    import glob
    csvs = glob.glob(os.path.join(data_dir,'**','*.csv'), recursive=True)
    if not csvs:
        raise FileNotFoundError('No CSVs found in '+data_dir)
    df = pd.read_csv(csvs[0], low_memory=False)
    # naive preprocessing
    df = df.dropna(axis=1, thresh=int(0.1*len(df)))
    numcols = df.select_dtypes(include=['float','int']).columns.tolist()
    if not numcols:
        raise RuntimeError('No numeric columns')
    ycol = numcols[0]
    df['target'] = (df[ycol] > df[ycol].median()).astype(int)
    X = df[numcols]
    y = df['target']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    from src.utils.metrics import binary_metrics
    print(binary_metrics(y_test, preds))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    joblib.dump(model, output)
    logger.info('Saved model to '+output)
if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/raw/flood')
    p.add_argument('--output', default='models/flood_baseline.pkl')
    args=p.parse_args()
    train(args.data_dir, args.output)
