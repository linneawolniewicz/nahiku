# Complete examples

The documentation renders the notebooks maintained in `examples/`.
Download the notebook from the source repository and run it with the installed package.

```{toctree}
:maxdepth: 1

synthetic_lightcurve
tabbys_star
beta_pic
```

The synthetic example runs offline. The stellar examples default to Lightkurve archive loading;
set `NAHIKU_OFFLINE=1` for the checksum-verified fixture option. Both loading paths construct Lightkurve objects. Full-data loops run only when enabled. Small fixtures and their manifests live alongside
the notebooks; larger FITS products remain in `NAHIKU_CACHE` or `~/.cache/nahiku`.

## Download notebooks

- {download}`Synthetic light curve <../../examples/synthetic_lightcurve.ipynb>`
- {download}`Tabby's Star <../../examples/tabbys_star.ipynb>`
- {download}`β Pictoris <../../examples/beta_pic.ipynb>`
