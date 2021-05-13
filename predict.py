import argparse
import numpy as np
import pandas as pd
from main import pre_procces
from pycaret.regression import load_model, predict_model


parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()


prediction_df = pre_procces(args.tsv_path)

model = load_model('model')

prediction_df = predict_model(model, prediction_df)
prediction_df.loc[prediction_df['Label']<0, 'Label'] = 0

prediction_df[['movie_id', 'Label']].to_csv("prediction.csv", index=False, header=False)
