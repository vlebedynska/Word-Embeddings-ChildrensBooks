import holoviews as hv
from holoviews import dim
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig
from transformers import AdamW
import torch
from torch.nn import BCEWithLogitsLoss, Sigmoid
from torch.utils.data import DataLoader
from tqdm import tqdm
import toxic


hv.extension('bokeh')

torch.manual_seed(0)

train, y_train, test, y_test = toxic.load_data()

num_labels = y_test.shape[1]
config = DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
