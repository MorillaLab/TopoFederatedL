# Contributing to TopoFederatedL

Thank you for your interest! This is an active research project — contributions to the FL aggregation strategies, TDA feature pipelines, privacy analysis, or new application domains are very welcome.

## 🐛 Reporting Bugs

Open a [GitHub Issue](https://github.com/MorillaLab/TopoFederatedL/issues) with:
- The script or notebook where the error occurs
- Your environment (OS, Python, PyTorch, giotto-tda versions)
- The full error traceback
- A minimal reproducible example

## 💡 Suggesting Features or Research Directions

Open an issue tagged `enhancement`. Especially welcome:
- New aggregation strategies (topology-weighted FedAvg, clustered FL)
- Integration of differential privacy with topological abstraction
- New TDA feature types for client-side representation
- Benchmarks against FedProx, SCAFFOLD, pFedMe, FedNova
- New application domains (finance, edge computing, NLP)
- Theoretical privacy guarantees for topological descriptors

## 🔧 Submitting Code

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install flake8 pytest
   ```
3. Keep FL client/server code modular and well-documented.
4. For any new TDA feature type, include:
   - A docstring with mathematical definition and reference
   - A privacy argument: why does this descriptor not leak raw data?
5. Lint:
   ```bash
   flake8 . --max-line-length=127
   ```
6. Clear notebook outputs before committing.
7. Open a pull request against `main` with motivation and results.

## 🔒 Privacy Contributions

Any contribution that modifies how client information is shared must include a **privacy analysis** explaining:
- What information is transmitted
- Whether raw data or individual records could be recovered from it
- How it relates to differential privacy or other formal privacy guarantees

## 📜 License

By contributing, you agree your work will be released under GPL-3.0.
