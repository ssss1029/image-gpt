

model_sizes = ["s", "m", "l"] #small medium large, xl not available
bs = 8 
n_px = 32

import os
import transformers
from transformers.modeling_gpt2 import GPT2Model,GPT2LMHeadModel
import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)
from skimage.filters import gaussian as gblur


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


import numpy as np

color_clusters_file = "/data/sauravkadavath/igpt/colors/kmeans_centers.npy"
clusters = np.load(color_clusters_file) #get color clusters


MODELS={"l":(1536,16,48),"m":(1024,8,36),"s":(512,8,24) } 
n_embd,n_head,n_layer=MODELS["l"] #set model hyperparameters
vocab_size = len(clusters) + 1 #add one for start of sentence token
config = transformers.GPT2Config(vocab_size=vocab_size,n_ctx=n_px*n_px,n_positions=n_px*n_px,n_embd=n_embd,n_layer=n_layer,n_head=n_head)
model_path = "/data/sauravkadavath/igpt/large/model.ckpt-1000000.index"

model = ImageGPT2LMHeadModel.from_pretrained(model_path,from_tf=True,config=config).cuda()

print("Finished loading model")


# # visualize samples with Image-GPT color palette.
# context = np.full( (bs,1), vocab_size - 1 ) #initialize with SOS token
# context = torch.tensor(context).cuda()
# output = model.generate(input_ids=context,max_length= n_px*n_px + 1,temperature=1.0,do_sample=True,top_k=40)
# import pathlib
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# samples = output[:,1:].cpu().detach().numpy()
# samples_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [n_px, n_px, 3]).astype(np.uint8) for s in samples] # convert color cluster tokens back to pixels

# for i, img in enumerate(samples_img):
#     plt.imsave(f"image_{i}.png", img)


# # Example forward pass for loss
# T = torch.tensor([[510] * 1024]).cuda()
# R = model.forward(T, labels=T)

# import pdb; pdb.set_trace()

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

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        "/data/imagenet/val", 
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Resize(32),
            igpt_colors,
        ])
    ),
    batch_size=1, 
    shuffle=True,
    num_workers=0, 
    pin_memory=True
)

from tqdm import tqdm
import random

with torch.no_grad():
    num_samples = 1000

    in_scores = []
    c = 0
    for image, _ in tqdm(val_loader):
        image = image.cuda()
        ret = model.forward(image, labels=image)
        in_scores.append(ret[0].item())

        c += 1
        if c >= num_samples:
            break

    print(in_scores)

    # Noises

    # # Random Noise
    # out_scores = []
    # for i in tqdm(range(num_samples)):
    #     noise = torch.tensor([[random.randint(0, 511) for i in range(32 * 32)]]).cuda()
    #     ret = model.forward(noise, labels=noise)
    #     out_scores.append(ret[0].item())

    # # Rademacher noise
    # out_scores = []
    # for i in tqdm(range(num_samples)):
    #     noise_img = torch.round(torch.rand((3, 32, 32))) # Random 1s and 0s
    #     noise_img = noise_img * 255 # Random 255s and 0s
    #     noise_img = noise_img.numpy().reshape(3, -1).transpose() # (1024, 3)
    #     noise_img = normalize_img(noise_img)
    #     d = squared_euclidean_distance_np(noise_img, clusters)
    #     argmin = np.argmin(d, axis=1)
    #     noise_img_igpt = torch.tensor(argmin).unsqueeze(0).cuda()

    #     ret = model.forward(noise_img_igpt, labels=noise_img_igpt)
    #     out_scores.append(ret[0].item())

    # Blobs
    out_scores = []
    for i in tqdm(range(num_samples)):
        ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(32, 32, 3)))
        ood_data = gblur(ood_data, sigma=1.5, multichannel=False)
        ood_data[ood_data < 0.75] = 0.0
        ood_data = ood_data.transpose((2, 0, 1))
        ood_data = torch.tensor(ood_data)
        ood_data = ood_data.reshape(32*32, 3).numpy()

        d = squared_euclidean_distance_np(ood_data, clusters)
        argmin = np.argmin(d, axis=1)
        noise_img_igpt = torch.tensor(argmin).unsqueeze(0).cuda()


        ret = model.forward(noise_img_igpt, labels=noise_img_igpt)
        out_scores.append(ret[0].item())


    print(out_scores)

    from sklearn.metrics import roc_auc_score
    AUROC = roc_auc_score(
        [0 for _ in range(num_samples)] + [1 for _ in range(num_samples)],
        in_scores + out_scores
    )

    print("AUROC = ", AUROC)