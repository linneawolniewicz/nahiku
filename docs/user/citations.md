# Citing Nāhiku and its dependencies

Use `CITATION.cff` for software metadata and record the version and commit used.
This checkout has no assigned publication or archive DOI. Add the release DOI after
archival and the JOSS DOI after acceptance.

Cite the components used in your analysis:

| Component | Reference |
|---|---|
| Gaussian processes | Rasmussen & Williams, [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/) |
| GPyTorch | Gardner et al. (2018), [GPyTorch](https://arxiv.org/abs/1809.11165) |
| PyTorch | Paszke et al. (2019), [PyTorch](https://arxiv.org/abs/1912.01703) |
| Astropy | [Astropy acknowledgement guidance](https://www.astropy.org/acknowledging.html) |
| Lightkurve | [Lightkurve citation guidance](https://lightkurve.github.io/lightkurve/about/citing.html) |
| NumPy | Harris et al. (2020), [Nature](https://doi.org/10.1038/s41586-020-2649-2) |
| SciPy | Virtanen et al. (2020), [Nature Methods](https://doi.org/10.1038/s41592-019-0686-2) |
| Matplotlib | Hunter (2007), [Computing in Science & Engineering](https://doi.org/10.1109/MCSE.2007.55) |
| Prewhitening antecedent | Dan Hey's [Balmung](https://github.com/danhey/balmung) |

The bundled Balmung implementation includes its attribution; its upstream MIT notice
is reproduced in `THIRD_PARTY_NOTICES.md`. The default preprocessing method uses the bundled Balmung algorithm. Nāhiku also
provides a separate weighted method using Astropy and SciPy.

For the case studies cite [Boyajian et al. (2016)](https://doi.org/10.1093/mnras/stw218)
and [Zieba et al. (2019)](https://arxiv.org/abs/1903.11071), together with NASA Kepler/TESS,
MAST, and the archive's mission-specific acknowledgement instructions. The manifests
record exact product URIs, checksums, selection windows, and quality masks.
