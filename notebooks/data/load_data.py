import pandas as pd


def load_train():
    return pd.read_csv('./data/train2.tsv', delimiter='\t')


def load_val():
    return pd.read_csv('./data/val2.tsv', delimiter='\t')


def load_test():
    return pd.read_csv('./data/test2.tsv', delimiter='\t')
