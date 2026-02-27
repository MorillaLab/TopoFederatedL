---
name: Feature / research direction
about: Suggest a new FL strategy, TDA feature, or application domain
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

## Motivation
Why would this improve TopoFederatedL — technically, mathematically, or in terms of privacy?

## Proposed addition
New aggregation strategy, TDA feature type, privacy mechanism, benchmark, or application domain.

## Privacy analysis (if relevant)
- What is transmitted between clients and server?
- Can raw data be recovered from the proposed descriptor?
- How does it relate to formal privacy guarantees (DP, k-anonymity)?

## Comparison baseline
Which existing FL method does this improve upon?
- [ ] FedAvg
- [ ] FedProx
- [ ] SCAFFOLD
- [ ] pFedMe
- [ ] Other: ___

## Implementation sketch (optional)
```python
server = TopoFLServer(aggregation='wasserstein_clustered', privacy='topological')
```

## References
Relevant papers or existing implementations.
