import argparse, os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from src.utils.logger import get_logger
logger = get_logger('train_air')
def train(data_dir, output):
    import glob
    csvs = glob.glob(os.path.join(data_dir,'**','*.csv'), recursive=True)
    if not csvs:
        raise FileNotFoundError('No CSVs found in '+data_dir)
    df = pd.read_csv(csvs[0], parse_dates=True, low_memory=False)
    # simple: pick first numeric as target
    numcols = df.select_dtypes(include=['float','int']).columns.tolist()
    if not numcols:
        raise RuntimeError('No numeric cols')
    target = numcols[0]
    df = df.fillna(method='ffill').fillna(method='bfill').dropna()
    X = df[numcols].drop(columns=[target])
    y = df[target]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = xgb.XGBRegressor(n_estimators=200, tree_method='hist', enable_categorical=True)
    model.fit(X_train,y_train, eval_set=[(X_test,y_test)], verbose=False)
    preds = model.predict(X_test)
    from src.utils.metrics import regression_metrics
    print(regression_metrics(y_test, preds))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    joblib.dump(model, output)
    logger.info('Saved model to '+output)
if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/raw/air_quality_india')
    p.add_argument('--output', default='models/air_quality_model.pkl')
    args=p.parse_args()
    train(args.data_dir, args.output)
