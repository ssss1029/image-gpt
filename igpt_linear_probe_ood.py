"""
Cifar-10 vs CIfar-100 OOD on iGPT Linear Probe using MSP score
"""


import os
import transformers
from transformers.modeling_gpt2 import GPT2Model, GPT2LMHeadModel, GPT2DoubleHeadsModel
import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)
from skimage.filters import gaussian as gblur

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from tqdm import tqdm
import random

import argparse

import argparse

import numpy as np

parser = argparse.ArgumentParser(description='iGPT-L on CIFAR')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--temp', default=1, type=float)
args = parser.parse_args()


val_data_in = torch.load("igpt_cifar_val_hiddens.pt")
val_penultimates_in, val_labels_in = val_data_in['penultimates'], val_data_in['labels']

print(val_penultimates_in.shape)
print(val_labels_in.shape)

val_loader_in = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(val_penultimates_in, val_labels_in),
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=0, 
    pin_memory=True
)


# val_data_out = torch.load("igpt_cifar100_val_hiddens.pt")
# val_penultimates_out, val_labels_out = val_data_out['penultimates'], val_data_out['labels']

# print(val_penultimates_out.shape)
# print(val_labels_out.shape)

# val_loader_out = torch.utils.data.DataLoader(
#     torch.utils.data.TensorDataset(val_penultimates_out, val_labels_out),
#     batch_size=args.batch_size, 
#     shuffle=True,
#     num_workers=0, 
#     pin_memory=True
# )


val_data_out = torch.load("igpt_32x32_solidcolors_hiddens.pt")
val_penultimates_out = val_data_out['penultimates']

print(val_penultimates_out.shape)

val_loader_out = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(val_penultimates_out),
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=0, 
    pin_memory=True
)

classifier = torch.nn.Linear(1536, 10).cuda()
classifier.load_state_dict(torch.load("igpt_classifier.pt"))
print("Loaded classifier weights")


in_scores = []
out_scores = []

for i, (penultimates, labels) in enumerate(tqdm(val_loader_in)):
    penultimates = penultimates.cuda()
    labels = labels.cuda()

    logits = classifier(penultimates)
    softmaxes = F.softmax(logits / args.temp, dim=1)
    max_smax_value = [e.item() for e in torch.max(softmaxes, dim=1)[0]]

    in_scores = in_scores + max_smax_value


# for i, (penultimates, labels) in enumerate(tqdm(val_loader_out)):
#     penultimates = penultimates.cuda()
#     labels = labels.cuda()

#     logits = classifier(penultimates)
#     softmaxes = F.softmax(logits / args.temp, dim=1)
#     max_smax_value = [e.item() for e in torch.max(softmaxes, dim=1)[0]]

#     out_scores = out_scores + max_smax_value


for i, (penultimates,) in enumerate(tqdm(val_loader_out)):
    penultimates = penultimates.cuda()

    logits = classifier(penultimates)
    softmaxes = F.softmax(logits / args.temp, dim=1)
    max_smax_value = [e.item() for e in torch.max(softmaxes, dim=1)[0]]

    out_scores = out_scores + max_smax_value

from sklearn.metrics import roc_auc_score
AUROC = roc_auc_score(
    [1 for _ in in_scores] + [0 for _ in out_scores], 
    in_scores + out_scores
)

print(f"AUROC = {AUROC}")

