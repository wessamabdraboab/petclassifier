from typing import Union

from fastapi import FastAPI

app = FastAPI()


from fastai.vision.all import *

learn = load_learner('export.pkl')

labels = learn.dls.vocab

@app.get("/predict/{img_url}")
def predict(img_url):
    img = PILImage.create(img_url)
    pred,pred_idx,probs = learn.predict(img)
    return {"label":pred}
    #return {labels[i]: float(probs[i]) for i in range(len(labels))}
