import os

from torch.nn import ReLU

from common import *

model = 'resnet18_32'
# vit_tiny_patch4_32   deit_tiny_patch4_32   deit_base_patch4_32  resnet18_32  convnext_tiny   convnext_large
dataset = 'cifar100' # cifar10  cifar100  imagenet1k

image_size = 32
batch_size = 256
lr = 0.0005 * (batch_size / 512)
output_dir = f"{output_root}/{dataset}/{model}/{os.path.basename(__file__).split('.')[0]}"
# model_kwargs = dict(act_layer=ReLU, drop_path_rate=0.1)
model_kwargs = dict(act_layer=ReLU)
