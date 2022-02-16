import pandas as pd
from transformers import EvalPrediction
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from typing import Dict

def read_data(data_path, columns):
    '''
    read data
    :param data_path:
    :param columns:
    :return:
    '''
    train_data = pd.read_csv(data_path, header=None, names=columns)
    return train_data

def split_data(pd, split_ratio):
    '''
    split data to train and val
    :param pd:
    :return:
    '''
    train_set = pd.sample(frac=split_ratio, replace=False)
    val_set = pd[~pd.index.isin(train_set.index)]

    return train_set, val_set

def compute_metrics(p: EvalPrediction) -> Dict:
    preds, labels = p
    preds = np.argmax(preds, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels.flatten(), preds.flatten(), average='weighted', zero_division=0)
    return {
        'accuracy': (preds == p.label_ids).mean(),
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


