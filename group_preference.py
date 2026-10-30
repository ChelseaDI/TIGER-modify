"""Aggregate fixed residuals using only the current sample's history."""
import torch


class GroupPreference:
    def __init__(self, path, indices, prefix_len):
        self.features = torch.load(path, map_location='cpu')
        if self.features['indices'] != indices or self.features['prefix_len'] != prefix_len:
            raise ValueError('Group features do not match the current SID mapping/prefix')
        self.groups = self.features['item_groups']
        self.residuals = self.features['residuals']
        self.content_dim = self.residuals.shape[1]

    def aggregate(self, history):
        items = torch.tensor([int(item) for item in history], dtype=torch.long)
        if not len(items):
            return torch.empty(0, self.content_dim)
        groups, inverse, counts = torch.unique(
            self.groups[items], sorted=True, return_inverse=True, return_counts=True)
        means = torch.zeros(len(groups), self.content_dim)
        means.index_add_(0, inverse, self.residuals[items])
        return means / counts.float().unsqueeze(1)
