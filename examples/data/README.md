# Fixed observation fixtures

These small CSV files are derived from public NASA Kepler/TESS PDCSAP products retrieved
using Lightkurve. The default notebooks load the archived FITS products through Lightkurve;
CSV fixtures provide an explicit offline fallback and reproducible test inputs.

The Tabby and β Pic JSON manifests record the target, product URI, raw FITS SHA-256, selected time window,
quality-bitmask integer, normalization, CSV checksum, time reference, and source reference.
CSV columns include source FITS row index and quality. No synthetic points replace observations.

Tabby's Star: quarter 16, BKJD 1507–1530. β Pic: sector 6, BTJD 1483–1489.
Times use TDB. The preparation recipe is `../codex/scripts/prepare_fixtures.py` relative to
the `nahiku/` repository root; raw FITS files remain outside the repository.

Acknowledge NASA Kepler/TESS and MAST and cite the papers and software listed in the citation guide.
