# Contributing to corpus-veritas

Thank you for your interest in contributing. Before you open a pull request, there are a few things to understand about this project.

## Read the Constitution First

[`CONSTITUTION.md`](CONSTITUTION.md) governs this project. All contributions must comply with the ethical constraints defined there. This is not a formality — it is the design specification for what this system is and is not allowed to do.

**Pull requests that weaken victim protections, lower the multi-source convergence threshold, remove a Hard Limit, or introduce features that serve no accountability journalism purpose will not be merged.**

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/corpus-veritas.git
cd corpus-veritas
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Branch Strategy

Work against the appropriate layer branch, not `main`. See [`README.md`](README.md) for the branch map. `main` receives only reviewed, tested code.

## Before Opening a Pull Request

- [ ] I have read `CONSTITUTION.md`
- [ ] My changes comply with all Hard Limits in Article III
- [ ] New features serve at least one purpose defined in Article I
- [ ] Any new query path that touches named entities passes through the victim flag check
- [ ] I have added or updated tests in `tests/integration/`
- [ ] If my changes touch the guardrail layer, I have added a corresponding red team test in `tests/red_team/`
- [ ] I have not committed any real document content, PII, or victim-adjacent data to the repository

## What Belongs in This Repo

- Pipeline code, RAG logic, guardrail logic, tests, infrastructure definitions
- Schema files (`corpus_registry.json`, `trusted_endorsers.json`) with placeholder/example entries
- Documentation

## What Does Not Belong in This Repo

- Actual document content from the Epstein corpus
- Any PII, victim names, or sensitive personal information
- API keys, AWS credentials, or secrets of any kind (use `.env` files and AWS Secrets Manager)

## Code Style

- Python: [Black](https://black.readthedocs.io/) formatting, type hints on all function signatures
- Docstrings on all public functions, including the confidence tier and provenance tag for any output the function can produce
- Comments explaining *why*, not just *what* — especially in the guardrail layer

## Questions

Open a GitHub Discussion rather than an Issue for questions about design intent or constitutional interpretation.
