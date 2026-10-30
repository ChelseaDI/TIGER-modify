"""Build fixed within-group residuals from the exact RQ-VAE input embeddings."""
import argparse
import json
from pathlib import Path

import torch


def build_features(embeddings, indices, prefix_len=2):
    embeddings = embeddings.detach().cpu().float()
    if embeddings.ndim != 2 or not torch.isfinite(embeddings).all():
        raise ValueError('Expected finite [items, content_dim] embeddings')
    if set(indices) != {str(i) for i in range(len(embeddings))}:
        raise ValueError('SID keys must match embedding row IDs 0..N-1')
    if prefix_len < 1 or any(len(code) < prefix_len for code in indices.values()):
        raise ValueError('Invalid group prefix length')
    prefixes = sorted({tuple(code[:prefix_len]) for code in indices.values()})
    lookup = {prefix: g for g, prefix in enumerate(prefixes)}
    groups = torch.tensor([lookup[tuple(indices[str(i)][:prefix_len])]
                           for i in range(len(embeddings))], dtype=torch.long)
    means = torch.zeros(len(prefixes), embeddings.shape[1])
    means.index_add_(0, groups, embeddings)
    means /= torch.bincount(groups, minlength=len(prefixes)).float().unsqueeze(1)
    return dict(item_groups=groups, group_means=means,
                residuals=embeddings - means[groups], group_prefixes=prefixes,
                prefix_len=prefix_len, indices=indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--indices', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--group_prefix_len', type=int, default=2)
    args = parser.parse_args()
    with open(args.indices) as stream:
        indices = json.load(stream)
    features = build_features(torch.load(args.embeddings, map_location='cpu'),
                              indices, args.group_prefix_len)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, str(output))
    print('Saved {} items, {} groups to {}'.format(
        len(indices), len(features['group_prefixes']), output))


if __name__ == '__main__':
    main()
