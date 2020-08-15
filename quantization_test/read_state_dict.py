import argparse
import numpy as np
import os
import json

import torch


# load checkpoints
init_path = "/home/david/git/proxylessnas/search/test/learned_net/init"

if torch.cuda.is_available():
    checkpoint = torch.load(init_path)
else:
    checkpoint = torch.load(init_path, map_location='cpu')

if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']

pass

weight_norm = checkpoint['first_conv.conv.weight'] / abs(checkpoint['first_conv.conv.weight'])
quant_weight = checkpoint['first_conv.conv.quantized_weight']

diff = torch.max(torch.abs(weight_norm - quant_weight))
print(diff)