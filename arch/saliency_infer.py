import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class Saliency_feat_infer(object):
    def __init__(self, p=0.5, epoch=None):
        self.p = p
        self.conv = nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=3, padding=1).cuda()

    def generate_saliency_x(self, x, max_class):
        batch, num_frames, dim = x.shape
        saliency_infer = torch.zeros(size=(batch, num_frames)) - 100.
        previous_x = torch.zeros_like(x)
        previous_x[:, 1:, :] = x[:, :-1, :]
        saliency_x = x - previous_x
        saliency_x = torch.abs(saliency_x)
        saliency_x = saliency_x.sum(dim=-1)
        _, num_saliency = saliency_x.shape
        median = int(num_saliency * self.p) - 1
        unsaliency_candidates = torch.topk(saliency_x, k=median, largest=True)[1]

        for b in range(unsaliency_candidates.size(0)):
            selected_clicks = unsaliency_candidates[b]
            saliency_infer[b, selected_clicks-1] = float(max_class)
            saliency_infer[b, selected_clicks] = float(max_class)
            saliency_infer = saliency_infer.long()

        return saliency_infer.cuda()
