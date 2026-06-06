# Contributing to pTopoFL

Thank you for your interest in contributing to pTopoFL.
We welcome contributions across four areas:

1. **New FL aggregation strategies** — alternative clustering methods, weighting schemes, or blending heuristics
2. **Extended TDA features** — additional persistence statistics, filtrations, or descriptor types
3. **Privacy analysis** — formal DP bounds for descriptor transmission, composition theorems
4. **New application domains** — real federated healthcare datasets, cross-device FL, graph FL

---

## Before you start

Please **open an issue** before submitting a pull request, so we can discuss
the proposed change and avoid duplicate work.

---

## Development setup

```bash
git clone https://github.com/MorillaLab/TopoFederatedL
cd TopoFederatedL
pip install -r requirements.txt
pip install pytest
pip install -e .
```

Run the test suite to confirm your environment works:

```bash
pytest tests/ -v
# All 38 tests should pass
```

---

## Code standards

- **Python ≥ 3.9.** Type hints encouraged but not required.
- **No external TDA libraries** in the core `ptopofl/` package. All PH
  computation uses NumPy and SciPy only. External libraries (e.g., Ripser,
  GUDHI) may be used in notebooks but not in the package itself.
- **Docstrings** on all public classes and methods (NumPy style).
- **Reproducibility:** every stochastic component must accept a `random_state`
  argument and produce identical results for the same seed.

---

## Adding a new baseline

1. Implement the class in `ptopofl/baselines.py`.
   It must expose `.fit(client_data, eval_data=None)` and `.metrics_` (list of
   per-round dicts containing at least `{'round': int, 'auc': float, 'acc': float}`).
2. Export it from `ptopofl/__init__.py`.
3. Add a `test_<name>` method to `tests/test_package.py` (inherit the
   `TestBaselines` pattern: check output format and AUC > 0.5).
4. Add it to the `METHODS` list in `ptopofl/experiments/run_experiments.py`.

---

## Adding a new topological feature

1. Add the computation to `ptopofl/descriptor.py`, inside `PHDescriptor.compute()`.
2. Update the descriptor dimension constant `PHDescriptor.DIM` and the layout
   table in the docstring.
3. Update `README.md`, the descriptor-layout table in Appendix B of the paper,
   and adjust `test_output_dimension` in the test suite.

---

## Pull request checklist

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New tests added for new functionality
- [ ] Docstrings updated
- [ ] `README.md` updated if the public API or results change
- [ ] No author names or institutional affiliations added (for anonymous review compatibility)
- [ ] `random_state` accepted by all stochastic functions

---

## Reporting a bug

Use the issue tracker with the **Bug report** template. Please include:
- Python version and OS
- Minimal reproducible example
- Full traceback
