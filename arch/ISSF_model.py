import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.autograd import Variable
from arch.saliency_infer import Saliency_feat_infer
