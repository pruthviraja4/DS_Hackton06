from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os
import torch
app = FastAPI(title='Urban AI Ops API')
MODEL_DIR = os.getenv('MODEL_DIR','models')
models = {}
# load tabular models if present
for name in ['flood_baseline','air_quality_model','traffic_model','earthquake_model']:
    path = os.path.join(MODEL_DIR, name + '.pkl')
    if os.path.exists(path):
        models[name]=joblib.load(path)
# UNet state dict path
unet_path = os.path.join(MODEL_DIR,'unet_flood.pt')
UNET_LOADED = os.path.exists(unet_path)
class PredictRequest(BaseModel):
    model: str
    features: dict = None
@app.get('/')
def root():
    return {'status':'running','models_loaded': list(models.keys()), 'unet_loaded': UNET_LOADED}
@app.post('/predict')
def predict(req: PredictRequest):
    if req.model not in models:
        return {'error':'model not found'}
    mdl = models[req.model]
    X = [list(req.features.values())]
    pred = mdl.predict(X)
    return {'prediction': pred.tolist()}
