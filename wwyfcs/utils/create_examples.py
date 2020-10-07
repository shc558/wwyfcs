# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd

# load data and tag lines with source characters' names
def load_data(args):
  data = pd.read_csv(args.file_path)
  data['id:data'] = data[args.id_colname]+':'+data[args.data_colname]
  return data

# create response sets where each row includes n previous responses as context
def extract_dialogues(df, args):
  dialogue_chains = []
  n = args.len_context
  for i in range(n, len(df[args.data_colname])):
    if args.character: #collect responses from specified character
      if df[args.id_colname][i] == args.character:
        row = []
        prev = i - 1 - n # include current response and previous n responses
        for j in range(i, prev, -1):
          row.append(df[args.data_colname][j])
        dialogue_chains.append(row)
    else:
      row = []
      prev = i - 1 - n
      for j in range(i, prev, -1):
        row.append(df[args.data_colname][j])
      dialogue_chains.append(row)

  columns = ['response','context']+['context/' + str(i) for i in range(n-1)]

  df = pd.DataFrame.from_records(dialogue_chains, columns= columns)
  df = df.dropna().reset_index(drop=True)

  return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str,
    help='Path to row data.')
    parser.add_argument('--data_colname', type=str,
    help='Name of the data field.')
    parser.add_argument('--id_colname', type=str,
    help='Name of the ID field.')
    parser.add_argument('--output_path', type=str,
    default=None, help='Path to output data')
    parser.add_argument('--character', type=str,
    default=None,help='Name of the character to extract.')
    parser.add_argument('--len_context', type=int,
    default = 9, help='Number of previous lines to use as context')

    args = parser.parse_args()

    extracted = extract_dialogues(load_data(args), args)
    if args.output_path:
        extracted.to_csv(args.output_path, index=False)
    else:
        extracted.to_csv(os.path.join(os.getcwd(),'examples.csv'), index=False)

if __name__ == "__main__":
    main()
