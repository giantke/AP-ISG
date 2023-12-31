from src.CheXbert.src.models.bert_labeler import bert_labeler
import torch
import torch.nn as nn
from src.path_datasets_and_weights import path_chexbert_weights
model = bert_labeler()
model = nn.DataParallel(model)
checkpoint = torch.load(path_chexbert_weights, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint['model_state_dict'], strict=False)