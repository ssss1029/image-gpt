

model_sizes = ["s", "m", "l"] #small medium large, xl not available
bs = 8 
n_px = 32

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

parser = argparse.ArgumentParser(description='iGPT-L on CIFAR')
parser.add_argument('--batch-size', type=int, default=1)
args = parser.parse_args()


def load_tf_weights_in_image_gpt2(model, config, gpt2_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    for name, array in zip(names, arrays):
        name = name[6:]  # skip "model/"
        name = name.split("/")

        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ) or name[-1] in ['_step']:
            logger.info("Skipping {}".format("/".join(name)))
            continue
        
        pointer = model
        if name[-1] not in ["wtet"]:
          pointer = getattr(pointer, "transformer")
        
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]

            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            elif scope_names[0] in ['q_proj','k_proj','v_proj']:
                pointer = getattr(pointer, 'c_attn')
                pointer = getattr(pointer, 'weight')
            elif len(name) ==3 and name[1]=="attn" and scope_names[0]=="c_proj":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, 'weight')
            elif scope_names[0]=="wtet":
                pointer = getattr(pointer, "lm_head")
                pointer = getattr(pointer, 'weight')
            elif scope_names[0]=="sos":
                pointer = getattr(pointer,"wte")
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]

        if len(name) > 1 and name[1]=="attn" or name[-1]=="wtet" or name[-1]=="sos" or name[-1]=="wte":
           pass #array is used to initialize only part of the pointer so sizes won't match
        else:
          try:
              assert pointer.shape == array.shape
          except AssertionError as e:
              e.args += (pointer.shape, array.shape)
              raise
          
        logger.info("Initialize PyTorch weight {}".format(name))

        if name[-1]=="q_proj":
          pointer.data[:,:config.n_embd] = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) ).T
        elif name[-1]=="k_proj":
          pointer.data[:,config.n_embd:2*config.n_embd] = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) ).T
        elif name[-1]=="v_proj":
          pointer.data[:,2*config.n_embd:] = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) ).T
        elif (len(name) ==3 and name[1]=="attn" and name[2]=="c_proj" ):
          pointer.data = torch.from_numpy(array.reshape(config.n_embd,config.n_embd) )
        elif name[-1]=="wtet":
          pointer.data = torch.from_numpy(array)
        elif name[-1]=="wte":
          pointer.data[:config.vocab_size-1,:] = torch.from_numpy(array)
        elif name[-1]=="sos":
          pointer.data[-1] = torch.from_numpy(array)
        else:
          pointer.data = torch.from_numpy(array)

    return model


from torch.nn.parameter import Parameter
class ln_mod(nn.Module):
    def __init__(self, nx,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.Tensor(nx))
    def forward(self,x):#input is not mean centered
        return x / torch.sqrt( torch.std(x,axis=-1,unbiased=False,keepdim=True)**2 + self.eps ) * self.weight.data[...,:] 

def replace_ln(m, name,config):
  for attr_str in dir(m):
      target_attr = getattr(m, attr_str)
      if type(target_attr) == torch.nn.LayerNorm:
          #print('replaced: ', name, attr_str)
          setattr(m, attr_str, ln_mod(config.n_embd,config.layer_norm_epsilon))

  for n, ch in m.named_children():
      replace_ln(ch, n,config)        

def gelu2(x):
    return x * torch.sigmoid(1.702 * x)

class ImageGPT2LMHeadModel(GPT2LMHeadModel):
  load_tf_weights = load_tf_weights_in_image_gpt2
  
  def __init__(self, config):
      super().__init__(config)
      self.lm_head = nn.Linear(config.n_embd, config.vocab_size - 1, bias=False)
      replace_ln(self,"net",config) #replace layer normalization
      for n in range(config.n_layer):
        self.transformer.h[n].mlp.act = gelu2 #replace activation 

  def tie_weights(self): #image-gpt doesn't tie output and input embeddings
    pass 

# class ImageGPT2DoubleHeadsModel(GPT2DoubleHeadsModel):

#     load_tf_weights = load_tf_weights_in_image_gpt2
  
#     def __init__(self, config):
#         super().__init__(config)
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size - 1, bias=False)
#         replace_ln(self,"net",config) #replace layer normalization
#         for n in range(config.n_layer):
#             self.transformer.h[n].mlp.act = gelu2 #replace activation 
        
#         # Replace MC head with right number of labels
#         config.num_labels = 10
#         self.multiple_choice_head = SequenceSummary(config)


#     def tie_weights(self): #image-gpt doesn't tie output and input embeddings
#         pass 


import numpy as np

color_clusters_file = "/data/sauravkadavath/igpt/colors/kmeans_centers.npy"
clusters = np.load(color_clusters_file) #get color clusters


MODELS={"l":(1536,16,48),"m":(1024,8,36),"s":(512,8,24) } 
n_embd,n_head,n_layer=MODELS["l"] #set model hyperparameters
vocab_size = len(clusters) + 1 #add one for start of sentence token
config = transformers.GPT2Config(vocab_size=vocab_size,n_ctx=n_px*n_px,n_positions=n_px*n_px,n_embd=n_embd,n_layer=n_layer,n_head=n_head)
model_path = "/data/sauravkadavath/igpt/large/model.ckpt-1000000.index"

model = ImageGPT2LMHeadModel.from_pretrained(model_path,from_tf=True,config=config).cuda()
model.eval()

print("Finished loading model")

# numpy implementation of functions in image-gpt/src/utils which convert pixels of image to nearest color cluster. 
def normalize_img(img):
  return img/127.5 - 1

def squared_euclidean_distance_np(a,b):
  b = b.T
  a2 = np.sum(np.square(a),axis=1)
  b2 = np.sum(np.square(b),axis=0)
  ab = np.matmul(a,b)
  d = a2[:,None] - 2*ab + b2[None,:]
  return d


def igpt_colors(image):
    """
    Image: C, W, H, in 0, 1
    """
    tensor_image = torch.round(transforms.ToTensor()(image) * 255)
    tensor_image = tensor_image.numpy().reshape(3, -1).transpose() # (1024, 3)
    tensor_image = normalize_img(tensor_image)
    d = squared_euclidean_distance_np(tensor_image, clusters)
    argmin = np.argmin(d, axis=1)

    return torch.tensor(argmin)

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "/data/sauravkadavath/cifar_data/",
        train=True, 
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            igpt_colors,
        ])
    ),
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=0, 
    pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(
        "/data/sauravkadavath/cifar_data/",
        train=False, 
        download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            igpt_colors,
        ])
    ),
    batch_size=args.batch_size, 
    shuffle=True,
    num_workers=0, 
    pin_memory=True
)

CLASSIFICATION_HIDDEN_LAYER = 20

# Batch size 1 so actually only 1 image and label in each batch
all_penultimate = []
all_labels = []

# with torch.no_grad():
#     for i, (images, labels) in enumerate(tqdm(val_loader)):
#         images = images.cuda()
#         labels = labels.cuda()

#         # import pdb; pdb.set_trace()
#         res = model.forward(images, output_attentions=True, output_hidden_states=True)

#         hidden_states = res[2][CLASSIFICATION_HIDDEN_LAYER] # torch.Size([1, 1024, 1536])
#         # penultimate = hidden_states[:, -1, :] # Last hidden vector
#         penultimate = torch.mean(hidden_states, dim=1)

#         # Save penultimate vectors
#         print(penultimate.shape)
#         print(labels.shape)

#         all_penultimate.append(penultimate.cpu())
#         all_labels.append(labels.cpu())
        


# Random Noise
with torch.no_grad():
    for i in tqdm(range(int(10000 / args.batch_size))):
        rand = random.randint(0, 511)
        noise = torch.tensor([[rand for i in range(32 * 32)] for b in range(args.batch_size)]).cuda()
        # print(noise.shape)
        res = model.forward(noise, output_attentions=True, output_hidden_states=True)

        hidden_states = res[2][CLASSIFICATION_HIDDEN_LAYER] # torch.Size([1, 1024, 1536])
        # penultimate = hidden_states[:, -1, :] # Last hidden vector
        penultimate = torch.mean(hidden_states, dim=1)

        # Save penultimate vectors
        # print(penultimate.shape)

        all_penultimate.append(penultimate.cpu())


# Save penultimates and labels
torch.save({
    "penultimates": torch.cat(all_penultimate, dim=0),
}, "igpt_32x32_solidcolors_hiddens.pt")


