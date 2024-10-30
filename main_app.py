# Created by Christian Huyghe on 10/29/2024
# Launches predictor application with Gradio
import gradio as gr
import pandas as pd
import numpy as np
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

X = pd.read_csv('Cleaned Data/cleaned_features.csv')
y = pd.read_csv('Cleaned Data/cleaned_target.csv')
samp = X.sample(n=10)
y = y[~X.isin(samp).all(axis=1)]
X = X[~X.isin(samp).all(axis=1)]

pca = PCA(n_components=22)
X = pca.fit_transform(X)

features = json.load(open("Cleaned Data/feature_statistics.json"))
restored_data = (samp * np.array(features["std"])) + np.array(features["mean"])
restored_data[restored_data.keys()[0]] = restored_data[restored_data.keys()[0]].astype(int)

print("Generating model...")

rfr = RandomForestRegressor(n_estimators=50, random_state=20, n_jobs=100, max_depth=23)
rfr.fit(X, y.to_numpy().ravel())

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
with gr.Blocks(theme="citrus") as app:
    gr.Markdown("# Superconductor Critical Temperature Model")
    slider = gr.Slider(minimum=1, maximum=len(samp), label="data ID", step=1)
    df = gr.DataFrame(restored_data.iloc[0:1], label="Data parameters")
    slider.change(fn=lambda i: restored_data.iloc[i:i+1], inputs=slider, outputs=df)
    with gr.Row(equal_height=True):
        button = gr.Button("Estimate Critical Temperature")
        result_box = gr.Textbox("", label="Estimate")
        button.click(fn=lambda i: f"{rfr.predict(pca.transform(samp.iloc[i-1:i]))[0]:.3f} \u00B1 10.376 Kelvin", inputs=slider,
                     outputs=result_box)

app.launch()
