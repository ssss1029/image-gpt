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
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()


train_data = torch.load("igpt_cifar_train_hiddens.pt")
train_penultimates, train_labels = train_data['penultimates'], train_data['labels']

print(train_penultimates.shape)
print(train_labels.shape)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_penultimates, train_labels),
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=0, 
    pin_memory=True
)

val_data = torch.load("igpt_cifar_val_hiddens.pt")
val_penultimates, val_labels = val_data['penultimates'], val_data['labels']

print(val_penultimates.shape)
print(val_labels.shape)

val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(val_penultimates, val_labels),
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=0, 
    pin_memory=True
)


classifier = torch.nn.Linear(1536, 10).cuda()

optimizer = torch.optim.SGD(
    classifier.parameters(), 
    args.learning_rate, # Learning Rate 
    momentum=0.9,
    weight_decay=0, 
    nesterov=False
)

# optimizer = torch.optim.Adam(
#     classifier.parameters(), 
#     args.learning_rate, # Learning Rate 
#     betas=(0.9, 0.999),
#     weight_decay=0, 
# )

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))

for epoch in range(args.epochs):
    correct = 0
    count = 0
    for i, (penultimates, labels) in enumerate(train_loader):
        count += penultimates.shape[0]

        penultimates = penultimates.cuda()
        labels = labels.cuda()

        logits = classifier(penultimates)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pred = logits.data.max(1)[1]
        correct += pred.eq(labels.data).sum().item()

    train_acc_avg = correct / count

    torch.save(
        classifier.state_dict(),
        "igpt_classifier.pt"
    )

    correct = 0
    count = 0
    for i, (penultimates, labels) in enumerate(val_loader):
        count += penultimates.shape[0]

        penultimates = penultimates.cuda()
        labels = labels.cuda()

        logits = classifier(penultimates)

        pred = logits.data.max(1)[1]
        correct += pred.eq(labels.data).sum().item()
    
    val_acc_avg = correct / count

    print(f"Epoch {epoch}: Train Acc: {train_acc_avg}, Val Acc: {val_acc_avg}")