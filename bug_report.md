---
name: Bug report
about: Report a bug in TopoFederatedL
title: '[BUG] '
labels: bug
assignees: ''
---

## Describe the bug
A clear description of what the bug is.

## Which component fails?
- [ ] TDA feature extraction (persistent homology)
- [ ] Wasserstein distance / client clustering
- [ ] FL aggregation (server-side)
- [ ] Local client training
- [ ] Privacy descriptor computation
- [ ] Other: ___

## Minimal reproducible example
```python
from topofederatedl import TopoFLClient
client = TopoFLClient(data=X, n_clients=3)
# Error here:
```

## Error traceback
```
Paste full traceback here
```

## FL setup context
- Number of clients:
- Data domain: [medical / image / tabular / other]
- Non-IID setting: [yes / no]
- Privacy mode: [gradient sharing / topological descriptor / other]

## Environment
- OS:
- Python version:
- PyTorch version:
- giotto-tda version:

## Additional context
