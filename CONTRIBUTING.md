# Contributing to pTopoFL

Thank you for your interest in contributing. Useful contributions include:

1. **Aggregation strategies:** alternative partitions, weights or blending rules. The convergence analysis covers any fixed partition and weights.
2. **Descriptors:** other vectorisations (landscapes, persistence images), other filtrations, or descriptors of learned representations.
3. **Privacy:** low-sensitivity topological summaries that make descriptor-level DP useful.
4. **Applications:** larger real cross-silo federations.

Please **open an issue** before submitting a pull request.

## Development setup

```bash
git clone https://github.com/MorillaLab/TopoFederatedL
cd TopoFederatedL
pip install -r requirements.txt pytest
pytest -q tests          # implementation and theorem checks
make quick               # smoke run of every experiment (a few minutes)
```

## Standards

- Keep `ptopofl_v2.py` dependency-light: NumPy, SciPy, scikit-learn and Matplotlib, with Ripser optional.
- New experiments must write their numbers through the macro mechanism (`Out.macro`), so that the paper and the code cannot diverge.
- Add a test for every new component. Where possible, the test should check a property that the theory guarantees.
- Fix seeds. Never report a number that the script does not produce.
