# JOSS readiness audit

Repository evidence was recorded on 2026-11-10. Submission guidance and the
review checklist were rechecked on 2026-11-23; remote repository state was not
re-audited. This checklist records preparation
evidence; acceptance and submission remain pending. The preparation described here
does not establish a published release, DOI registration, or submission.

Sources:
- [Submitting](https://joss.readthedocs.io/en/latest/submitting.html)
- [Review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html)
- [Review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html)
- [Example paper](https://joss.readthedocs.io/en/latest/example_paper.html)

## Software and documentation evidence

| Requirement | Evidence prepared | Remaining action |
|---|---|---|
| Public browsable source and issue/contribution access | GitHub API reports `linneawolniewicz/nahiku` public, issues enabled, pull requests allowed | Review and publish this local implementation only after approval |
| OSI-approved license | Existing MIT LICENSE; upstream Balmung notice verified and preserved in THIRD_PARTY_NOTICES.md | Human review of attribution before release |
| Research purpose and audience | README and documentation introduction explain stellar light-curve anomaly detection | Authors confirm the scientific framing |
| Standard installation | pyproject.toml, declared dependency extras, lockfile, wheel/sdist build and external import checks | Revalidate supported platforms before broader compatibility claims |
| Functionality and API documentation | Explicit central-object API, reference docstrings, 13 executed feature tutorials | Human scientific review |
| Real-world usage examples | Three maintained examples; fixed Kepler and TESS products loaded with Lightkurve | Review astrophysical interpretation and parameter choices |
| Verification | tests numerical suite, direct/precision equivalence, injection recovery, notebook execution | CI runs publicly after approved push |
| Performance claims | Complexity and limitations documented; runtimes recorded with result metadata | No comparative speed claim made; benchmark before making one |
| Community pathways | CONTRIBUTING.md, developer/support guide, issue links | Maintainer confirms support/governance expectations |
| Reproducibility | Fixture/product SHA-256, observation manifests, seeds, configurations, result exports | Archive the reviewed release and relevant reproducibility materials |
| Release record | CHANGELOG.md and 0.2.0 metadata prepared | No GitHub releases were present in the API audit; create release after approval |

## Eligibility evidence that code edits cannot establish

The 2026-11-10 repository API audit recorded creation on 2026-02-12 and public visibility at that time.
The inspected public commit history contains 21 commits, with recent entries in
April, May, and July 2026, and one issue/PR entry concerning a greedy-search array
range bug. These facts show development activity, but **repository creation and
commit dates do not prove the date it became public** or establish every JOSS gate.

JOSS submission guidance requires more than six months of public development history,
iteration over time, and demonstrated research use. Authors must verify the public
history and provide concrete research-use evidence: analyses, workflows, publications,
or adoption as applicable. Claims about external users, citations, or activity require supporting evidence.
The case studies demonstrate reproducible capabilities; they do not automatically
establish external adoption or satisfy editorial scope/significance screening.

The 2026-11-10 audit recorded remote commits later than this checkout’s initial HEAD.
The implementation is based on the supplied local workspace. No remote merge or push
was performed. Review changes against the current remote branch before integration.

## Research-use evidence

The author reports using the method to search K2 light curves for dips relevant to
exocomet searches and to quantify recovery of injected dips in observed light curves.
The astrophysical origin of the newly detected dips remains unconfirmed. This is
research use of the method; no confirmed exocomet discovery is claimed.

A local inspection on 2026-11-23 of the sibling research directory
`dipper-detection/data/k2_may2025/` found 619 FITS files in `raw/`, 619 FITS files in
`renamed/`, and 619 rows in each of `synthetic_params/gp_parameters.csv` and
`synthetic_params/high_residuals.csv`. There are also 619 GP-fit PNG files and a
processing log. These counts describe stored artifacts, not independently verified
unique targets, successful fits, or recovered injections. The log includes a
conjugate-gradient convergence warning, so the existence of a fit record alone does
not establish its numerical quality. These historical records must be linked to
the research code revision that produced them; they do not validate the current
package revision automatically.

To make this evidence assessable, prepare a research-use record containing:

- The scientific objective, analysis dates, researcher names, and code revision.
- A manifest linking renamed files to EPIC identifiers, K2 campaigns, archive
  products, selection criteria, and input checksums.
- The preprocessing, GP, search, and threshold settings, with execution commands
  and a small reproducible subset.
- Injection parameters and random seeds, the stage at which signals were injected,
  a defined event-matching rule, and counts of attempted, eligible, recovered, and
  failed trials. Report recovery fractions with uncertainty and stratify them by
  signal depth, duration, and relevant observing conditions where supported.
- A distinction between data used to choose settings and data used to evaluate
  them. Recovery on tuning data does not provide an independent performance estimate.
- Candidate-dip records with follow-up status. Unlabelled detections do not establish
  either confirmed exocomets or a measured false-positive rate.

The maintained β Pic example analyzes an event interpreted as an exocomet transit
in [Zieba et al. (2019)](https://doi.org/10.1051/0004-6361/201935552).
The Tabby's Star example recovers previously reported dimming; it should not be
presented as confirmation of an exocomet. Both examples document application to
observations, with settings selected using known event windows. They do not measure
blind-search completeness or independently validate the origin of a dip.

[JOSS submission guidance](https://joss.readthedocs.io/en/latest/submitting.html)
accepts demonstrated developer research workflows, including private workflows
shown to the editors. A published discovery is not required. The author should
select evidence that can be shared publicly and identify any unpublished supporting
records to provide privately. Historical quantitative K2 recovery claims remain pending a traceable analysis
of their original injection and detection records.

Local validation on 2026-11-23 passed 378 tests with 94.45% coverage including
branches, source/wheel builds, dependency checks, strict documentation builds,
and the original 16 notebooks. The clean CPU wheel environment passed 364 tests;
14 optional Lightkurve tests were skipped there and passed in the full environment.
Live archive retrieval and public CI remain separate checks.

## Paper and author tasks (in progress)

The paper remains outside this commit. Its sections and author metadata require
completion and review. Required sections are:

- Summary for a broad audience.
- Statement of need and intended users.
- State of the field, including a substantive build-versus-contribute rationale.
- Software design and scientific/architectural trade-offs.
- Research impact statement grounded in evidence.
- AI usage disclosure, with tools/models, scope, and human validation.
- References, with verified citations in paper.bib.

Authors still need to provide/confirm author order, affiliations, ORCIDs, contributions,
funding, conflicts, research-use evidence, and final paper wording. Document human review of generated material only after it has occurred. The JOSS paper should
focus on the software, not present these example analyses as new research discoveries.

## Figures and release/archive sequence

Fourteen candidate star figures and a β Pic prewhitening diagnostic live in the sibling
`codex/figures/` gallery with reproduction scripts and numerical outputs. They are
exploratory development artifacts, not bundled library assets. Select one two-panel
paper figure later and include that selected figure and its reproduction recipe in
the reviewed paper/software repository or associated archive before submission.

After scientific review and approval: integrate the changes, activate hosted docs,
prepare the submission paper, and follow JOSS's release/archive instructions. Upon
successful review, tag the reviewed version, archive it in Zenodo/figshare, obtain a
release DOI, and update citation metadata. Record software and JOSS DOIs only after they are assigned.

## AI assistance record for later author review

This local development session used OpenAI Codex assistance for API implementation,
numerical-test design, documentation/tutorial drafting, notebook execution, figure
scripts, and this checklist. The authors specified the central-object design, project
scope, real-star targets, figure variants, and constraints. An exact model/version
identifier was not exposed reliably in the working interface; the authors should
record it from their session metadata if available. Author review is ongoing; the final disclosure must identify the human review
and validation completed before submission.

Current JOSS policy permits disclosed assistance in code/docs/paper preparation but
prohibits AI-generated author/editor/reviewer conversational interactions except for
translation. The authors should conduct the review correspondence themselves.
