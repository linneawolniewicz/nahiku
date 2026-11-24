"""Data lineage, event boundaries, and portable results on small deterministic inputs."""

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nahiku import Nahiku
from nahiku.results import SearchResult
from nahiku.search import evaluate_events


@pytest.mark.parametrize("with_errors", [False, True])
def test_copy_bin_and_export_preserve_observation_lineage(tmp_path, with_errors):
    time = np.array([0.0, 0.1, 1.0, 1.1, 2.0, 2.1])
    flux = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    errors = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]) if with_errors else None
    parent = Nahiku(time, flux, errors, metadata={"nested": {"label": "original"}})
    child = parent.copy()
    child.metadata["nested"]["label"] = "copy"
    child.inject_anomaly(center=1, width=0.01, depth=1)
    binned = child.bin(0.5)
    assert parent.metadata["nested"]["label"] == "original"
    np.testing.assert_array_equal(parent.flux, flux)
    assert not parent.truth_mask.any()
    np.testing.assert_array_equal(binned.truth_mask, [False, True, False])
    expected = [10.4, 13.6, 18.4] if with_errors else [11.0, 14.5, 19.0]
    np.testing.assert_allclose(binned.flux, expected)
    if with_errors:
        np.testing.assert_allclose(binned.flux_err, np.sqrt(0.8))
    else:
        assert binned.flux_err is None
    child.standardize()
    prefix = tmp_path / "nested" / "observations"
    child.export(prefix)
    with np.load(str(prefix) + ".npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["original_flux"], flux)
        np.testing.assert_array_equal(saved["flux"], child.flux)
        np.testing.assert_array_equal(saved["truth_mask"], child.truth_mask)
        np.testing.assert_array_equal(saved["original_flux_err"], [] if errors is None else errors)
    provenance = json.loads(prefix.with_suffix(".json").read_text())
    assert provenance["preprocessing"][-1]["operation"] == "standardize"
    child.reset()
    np.testing.assert_array_equal(child.flux, flux)
    assert child.history == [] and not child.truth_mask.any()


@pytest.mark.parametrize("gap_factor, expected", [(5, [(1, 3), (3, 5)]), (None, [(1, 5)])])
def test_result_gap_boundaries_and_empty_candidate_export(tmp_path, gap_factor, expected):
    time = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    result = SearchResult(
        "greedy",
        time,
        np.ones(6),
        np.zeros(6),
        np.ones(6),
        np.array([False, True, True, True, True, False]),
        [],
        {"gap_factor": gap_factor},
        0.0,
    )
    assert result.intervals == expected
    assert [(e["time_start"], e["time_last"]) for e in result.events] == [
        (time[a], time[b - 1]) for a, b in expected
    ]
    paths = result.export(tmp_path / "result")
    with paths["candidates"].open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["start", "stop", "score"]
        assert list(reader) == []
    samples = np.genfromtxt(paths["samples"], delimiter=",", names=True)
    np.testing.assert_array_equal(samples["detected"], result.mask)
    fig, axes = plt.subplots(2)
    returned, used = result.plot(axes=axes)
    assert returned is fig and len(used) == 2
    plt.close(fig)
    with pytest.raises(ValueError, match="two axes"):
        result.plot(axes=[])


@pytest.mark.parametrize("buffer, matches", [(0, 0), (1, 1)])
def test_event_buffer_respects_half_open_endpoints(buffer, matches):
    report = evaluate_events([(0, 2)], [(2, 4)], buffer=buffer)
    assert report["matched_events"] == matches
    assert report["false_positives"] == 1 - matches
    assert report["missed_events"] == 1 - matches


@pytest.mark.parametrize("buffer", [-1, 0.5])
def test_event_buffer_rejects_invalid_sample_counts(buffer):
    with pytest.raises(ValueError, match="nonnegative sample count"):
        evaluate_events([], [], buffer=buffer)


def test_plot_original_and_working_observations():
    lc = Nahiku(np.arange(6.0), np.arange(6.0) + 10)
    lc.standardize()
    fig, ax = lc.plot(original=True)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), lc.original_flux)
    plt.close(fig)
    fig, ax = plt.subplots()
    returned, _ = lc.plot(ax=ax)
    assert returned is fig
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), lc.flux)
    plt.close(fig)


@pytest.mark.parametrize("method", ["balmung", "weighted"])
def test_prewhitening_constant_flux_diagnostics(method):
    lc = Nahiku(np.linspace(0, 10, 101), np.full(101, 12.0))
    result = lc.prewhiten(fmin=1, fmax=5, method=method)
    assert result.stopping_reason == "minimum_snr"
    assert len(result.frequencies) == 0
    np.testing.assert_array_equal(result.residuals, lc.original_flux)
    np.testing.assert_array_equal(result.power_before, 0)
    np.testing.assert_array_equal(result.power_after, 0)
    fig, axes = result.plot()
    np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), lc.original_flux)
    plt.close(fig)
    fig, axes = plt.subplots(2)
    returned, _ = result.plot(axes=axes)
    assert returned is fig
    plt.close(fig)
