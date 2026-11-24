"""Archive boundary tests using real Lightkurve conversion and stitching.

Only the remote search/download boundary is replaced, so tests need no archive
access. LightCurve, LightCurveCollection, normalization, and uncertainties are real.
"""

import json

import numpy as np
import pytest

from nahiku import Nahiku


@pytest.fixture
def archive(monkeypatch):
    lk = pytest.importorskip("lightkurve")
    first = lk.LightCurve(
        time=[0.0, 1.0, 2.0, 3.0], flux=[10.0, np.nan, 8.0, 12.0],
        flux_err=[1.0] * 4, meta={"QUARTER": 15, "FILENAME": "q15.fits"},
    )
    second = lk.LightCurve(
        time=[10.0, 11.0, 12.0], flux=[20.0, 18.0, 22.0],
        flux_err=[2.0] * 3, meta={"QUARTER": 16, "FILENAME": "q16.fits"},
    )

    class Archive:
        count = 2
        single = first
        # Reverse archive order to exercise chronological sorting and row lineage.
        collection = lk.LightCurveCollection([second, first])

        def __init__(self):
            self.calls = []

        def __len__(self):
            return self.count

        def download(self, **kwargs):
            self.calls.append(("download", kwargs))
            return self.single

        def download_all(self, **kwargs):
            self.calls.append(("download_all", kwargs))
            return self.collection

        def search(self, target, **kwargs):
            self.calls.append(("search", target, kwargs))
            return self

    result = Archive()
    monkeypatch.setattr(lk, "search_lightcurve", result.search)
    return result


def load(**kwargs):
    return Nahiku.download("KIC 8462852", mission="Kepler", author="Kepler", exptime=1800, **kwargs)


def test_single_download_is_unchanged(archive, tmp_path):
    archive.count = 1
    result = load(quarter=15, cache_dir=tmp_path, quality_bitmask="hard")
    np.testing.assert_allclose(result.time, [0, 2, 3])
    np.testing.assert_allclose(result.flux, [1, 0.8, 1.2])
    np.testing.assert_allclose(result.flux_err, [0.1] * 3)
    assert result.metadata["quarter"] == "15"
    assert result.metadata["product_count"] == 1
    assert result.metadata["stitched"] is False
    assert archive.calls[-1] == ("download", dict(download_dir=tmp_path, quality_bitmask="hard"))


def test_stitch_normalizes_errors_filters_sorts_and_preserves_lineage(archive, tmp_path):
    result = load(stitch=True, quarter=[15, 16], cache_dir=tmp_path, quality_bitmask="hard")
    np.testing.assert_allclose(result.time, [0, 2, 3, 10, 11, 12])
    np.testing.assert_allclose(result.flux, [1, 0.8, 1.2, 1, 0.9, 1.1])
    np.testing.assert_allclose(result.flux_err, [0.1] * 6)
    assert archive.calls[0][2]["quarter"] == [15, 16]
    assert archive.calls[-1] == ("download_all", dict(download_dir=tmp_path, quality_bitmask="hard"))
    assert result.metadata["source_indices"] == [3, 5, 6, 0, 1, 2]
    assert result.metadata["source_products"] == [
        dict(start=0, stop=3, quarter="16", filename="q16.fits"),
        dict(start=3, stop=7, quarter="15", filename="q15.fits"),
    ]
    assert "quarter" not in result.metadata
    assert result.metadata["product_count"] == 2
    assert result.metadata["stitched"] is True
    assert result.history == [] and result.period is None
    np.testing.assert_array_equal(result.flux, result.original_flux)
    result.export(tmp_path / "stitched")
    assert json.loads((tmp_path / "stitched.json").read_text())["selection"]["quarter"] == [15, 16]


def test_stitch_also_accepts_one_match(archive):
    archive.count = 1
    archive.collection = type(archive.collection)([archive.single])
    result = load(stitch=True)
    np.testing.assert_allclose(result.flux, [1, 0.8, 1.2])
    assert result.metadata["product_count"] == 1


@pytest.mark.parametrize("stitch", [False, True])
def test_empty_search(archive, stitch):
    archive.count = 0
    with pytest.raises(ValueError, match="No products"):
        load(stitch=stitch)
    assert len(archive.calls) == 1


def test_multiple_matches_require_opt_in(archive):
    with pytest.raises(ValueError, match="stitch=True"):
        load()
    assert len(archive.calls) == 1


@pytest.mark.parametrize("missing", ["none", "empty", "partial"])
def test_incomplete_collection_is_not_silently_used(archive, missing):
    archive.collection = None if missing == "none" else type(archive.collection)(
        [] if missing == "empty" else [archive.single]
    )
    with pytest.raises(RuntimeError, match="every selected"):
        load(stitch=True)


def test_single_download_failure(archive):
    archive.count = 1
    archive.single = None
    with pytest.raises(RuntimeError, match="no light curve"):
        load()


def test_overlapping_products_do_not_silently_duplicate_samples(archive):
    archive.collection = type(archive.collection)([archive.single, archive.single.copy()])
    with pytest.raises(ValueError, match="strictly increasing"):
        load(stitch=True)


def test_invalid_stitch_setting(archive):
    with pytest.raises(ValueError, match="True or False"):
        load(stitch="False")
    assert not archive.calls
