# Contributing to Nāhiku

For bugs and support, open an issue at https://github.com/linneawolniewicz/nahiku/issues.
Include Python/Nāhiku/dependency versions, OS, a minimal reproducer, expected and observed behavior,
and public data product identifiers where relevant. Avoid posting private data or credentials.

Discuss scientific/API changes before substantial implementation. Contributions should explain
assumptions, include a meaningful numerical regression test, and update tutorials/docstrings.
Install `python -m pip install -e ".[dev]"`; run `python -m pytest` and a strict Sphinx build.
Use `tests/` for the maintained release suite. Use the scientific dependencies in numerical tests.

Notebook examples should run top-to-bottom, use reproducible seeds and public data with recorded provenance,
and include interpretation and limitations. Keep the three main notebooks in `examples/`;
feature tutorials belong in `docs/tutorials/`. Avoid source-path hacks and local absolute paths.

The maintainer makes release and API decisions after review. No guaranteed response time is
promised. Be respectful and constructive; explain disagreements with evidence. Contributors
retain credit through commit history, acknowledgements, and appropriate software/paper authorship.

Before release, validate wheel/sdist installation, tests, notebooks, docs, citation metadata,
third-party notices, and the JOSS checklist. Tagging, publishing, pushing, and deleting files require explicit maintainer approval.
