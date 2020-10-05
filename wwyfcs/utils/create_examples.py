# -*- coding: utf-8 -*-
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# create response sets where each row includes n previous responses as context
def load_data():
  file_path='raw/Game_of_Thrones_Script.csv'
  file_dir = os.path.join(os.getcwd(),'data') #os.path.join(os.cwd(),os.path.dirname(__file__))
  data = pd.read_csv(Path(file_dir)/file_path)
  print(data.info())
  return data

def extract_dialogues(df, data_col, id_col, character=None, n=9):

  dialogue_chains = []

  for i in range(n, len(df[data_col])):
    if character: #collect responses from specified character
      if df[id_col][i] == character:
        row = []
        prev = i - 1 - n # include current response and previous n responses
        for j in range(i, prev, -1):
          row.append(df[data_col][j])
        dialogue_chains.append(row)
    else:
      row = []
      prev = i - 1 - n
      for j in range(i, prev, -1):
        row.append(df[data_col][j])
      dialogue_chains.append(row)

  columns = ['response','context']+['context/' + str(i) for i in range(n-1)]

  return pd.DataFrame.from_records(dialogue_chains, columns= columns)

# create train and test sets
script = load_data()
script['Name:Sentence'] = script['Name']+':'+script['Sentence']
# script['Name'][:3]
DATA_COL = 'Name:Sentence'
ID_COL = 'Name'
df = extract_dialogues(script, DATA_COL, ID_COL)
df = df.dropna().reset_index(drop=True)
df.info()

train, eval = train_test_split(df, test_size = 0.1, random_state=42)

train.to_csv(Path(os.getcwd())/'data'/'preprocessed'/'all_train.csv', index=False)
eval.to_csv(Path(os.getcwd())/'data'/'preprocessed'/'all_eval.csv', index=False)
