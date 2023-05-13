import pickle
from transformers import pipeline
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

clf_lr = pickle.load(open('model_lgb.pkl', 'rb'))
fill_mask_pipeline = pipeline('fill-mask', model='distilroberta-base')

CLASS_NAMES = [0, 1]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class EmbeddingsOfTextDataset(Dataset):
    def __init__(self, df, fill_mask_pipeline, class_names=CLASS_NAMES, device=device):
        self.df = df
        self.tokenizer = fill_mask_pipeline.tokenizer
        self.preprocessor_model = fill_mask_pipeline.model
        self.preprocessor_model.lm_head = nn.Identity()
        self.preprocessor_model = self.preprocessor_model.to(device)
        self.class_names = CLASS_NAMES
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tweet_info = self.df.iloc[index]
        X = self.tokenizer(tweet_info['text'], truncation=True, max_length=128, return_tensors='pt')["input_ids"]
        X = X.to(device)
        with torch.no_grad():
            X = self.preprocessor_model.forward(X)[-1][0].sum(axis=0)

        t = X.to(torch.device("cpu"))
        del X
        return t, tweet_info["target"]


def GetPrediction(text):
    data = pd.DataFrame({'text': [text], 'target': 1})
    X = EmbeddingsOfTextDataset(data, fill_mask_pipeline).__getitem__(0)[0]
    p = float(clf_lr.predict_proba(X.reshape(1, -1))[:, 1])
    verdict = 'suicide' if float(clf_lr.predict_proba(X.reshape(1, -1))[:, 1]) >= 0.393939393939394 else 'not suicide'

    return (p, verdict)

