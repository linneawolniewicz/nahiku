"""Portable analysis results; numerical exports never pickle a live model."""

from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np


def events_from_mask(mask):
    """Return contiguous true runs as half-open ``[start, stop)`` sample intervals.

    :param mask: One-dimensional boolean detection mask; true runs become intervals.
    :type mask: array-like
    :returns: Contiguous true runs as half-open sample-index pairs.
    :rtype: list
    """
    # Pad with false endpoints so boundary events produce complete start/stop pairs.
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


@dataclass
class SearchResult:
    """Snapshot of a detection run, independent of subsequent object changes.

    ``std`` is predictive observation uncertainty. ``intervals`` contains merged
    events. For greedy search, ``candidates`` contains only accepted intervals,
    not rejected growth trials. For exhaustive search, it contains every tested
    interval and its score, including intervals not selected as detections.
    Exhaustive scores describe each held-out interval. Greedy scores are the minimum
    standardized strength of the qualifying seed before interval growth; they are
    not scores recomputed on the final expanded interval. Neither is a discovery probability.
    Search methods copy their arrays; direct dataclass construction does not copy inputs.

    :param conditional_predictions: Selected exhaustive candidate records, ranked strongest first.
        Each contains start/stop, score, mean_residual, the exact scored interval mean,
        observation std and covariance, and full_mean/full_std on the entire time grid.
        Full-grid predictions exclude only that candidate. Arrays are independent of live GPs.
        Storage scales with selected windows times grid length plus squared window lengths.
    :type conditional_predictions: list of dict
    :param method: Search method name.
    :type method: str
    :param time: Copied observation timestamps in days.
    :type time: numpy.ndarray
    :param flux: Copied working flux at search time.
    :type flux: numpy.ndarray
    :param mean: Final predictive mean after excluding detected samples.
    :type mean: numpy.ndarray
    :param std: Predictive observation standard deviation, including noise.
    :type std: numpy.ndarray
    :param mask: Boolean mask of detected samples.
    :type mask: numpy.ndarray
    :param candidates: Scored intervals, with half-open bounds and mean residuals; greedy records
        accepted intervals, exhaustive records all tested windows. Greedy mean_residual
        is measured over the accepted bounds using the prediction before excluding
        that event; its score describes the original seed.
    :type candidates: list of dict
    :param settings: Search configuration and score conventions.
    :type settings: dict
    :param runtime: Elapsed search runtime in seconds.
    :type runtime: float
    :param metadata: Observation, preprocessing, fit, and environment provenance.
    :type metadata: dict
    """

    method: str
    time: np.ndarray
    flux: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    mask: np.ndarray
    candidates: list
    settings: dict
    runtime: float
    metadata: dict = field(default_factory=dict)
    conditional_predictions: list = field(default_factory=list)

    @property
    def residuals(self):
        """Working flux minus the reported predictive mean.

        :returns: Working flux minus the predictive mean.
        :rtype: numpy.ndarray
        """
        return self.flux - self.mean

    @property
    def intervals(self):
        """Merged half-open sample intervals; adjacent detections form one event.

        :returns: Merged half-open intervals, split at gaps when gap_factor is enabled;
            gap_factor=None leaves contiguous sample runs intact.
        :rtype: list
        """
        events = events_from_mask(self.mask)
        # Split contiguous sample runs where timestamps reveal a large observing gap.
        gap_factor = self.settings.get("gap_factor", 5.0)
        if gap_factor is None:
            return events
        gaps = np.flatnonzero(np.diff(self.time) > gap_factor * np.median(np.diff(self.time))) + 1
        split = []
        for start, stop in events:
            bounds = [start, *gaps[(gaps > start) & (gaps < stop)].tolist(), stop]
            split.extend(zip(bounds[:-1], bounds[1:]))
        return split

    @property
    def events(self):
        """Event records including first and last observed timestamps (days).

        :returns: Event dictionaries with sample bounds and first/last observed times.
        :rtype: list
        """
        return [
            dict(start=a, stop=b, time_start=float(self.time[a]), time_last=float(self.time[b - 1]))
            for a, b in self.intervals
        ]

    def plot(self, axes=None, *, kind=None, selection="representative", event_index=None):
        """Plot an overview with conditional fits, a single full-grid fit, or residuals.

        Exhaustive results default to conditional fits. Each colored curve excludes only
        its candidate window, with observation uncertainty at ±2 standard deviations.
        Colored observations mark that window. Representatives are the strongest selected
        candidate in each merged event, breaking score ties by start then stop.
        A single-event view shows its representative prediction across the full time grid.
        All predictions are stored arrays; plotting never fits or conditions a live GP.

        :param axes: Two axes for overview panels, or one for a single-event view; None creates them.
        :type axes: matplotlib.axes.Axes or sequence or None
        :param kind: None uses conditional for exhaustive and residuals for greedy;
            "conditional" or "residuals" explicitly selects the diagnostic.
        :type kind: str or None
        :param selection: "representative" shows one candidate per event; "all" shows every
            selected candidate, including overlapping windows. Applies to the overview.
        :type selection: str
        :param event_index: Zero-based merged-event index for one full-grid conditional fit.
            None shows the overview. Requires conditional kind and representative selection.
        :type event_index: int or None
        :returns: Figure and axes array (one axis for a single-event view, otherwise two).
        :rtype: tuple
        """
        import matplotlib.pyplot as plt

        # Validate the requested diagnostic before creating any figures.
        kind = kind or ("conditional" if self.method == "exhaustive" else "residuals")
        if kind not in ("conditional", "residuals"):
            raise ValueError("kind must be conditional or residuals")
        if selection not in ("representative", "all"):
            raise ValueError("selection must be representative or all")
        if kind == "residuals":
            if event_index is not None:
                raise ValueError("event_index requires conditional plots")
            return self._plot_residuals(axes)
        if self.method != "exhaustive":
            raise ValueError("Conditional candidate plots require exhaustive results")
        if self.intervals and not self.conditional_predictions:
            raise ValueError("This result has no stored conditional predictions")
        # Match each merged event to its strongest scored constituent window.
        smaller = self.settings.get("metric", "pvalue") in ("pvalue", "ll", "mll")
        ranked = sorted(self.conditional_predictions, key=lambda c: (
            c["score"] if smaller else -c["score"], c["start"], c["stop"]))
        representatives = [next(c for c in ranked if a <= c["start"] and c["stop"] <= b)
                           for a, b in self.intervals]
        if event_index is not None:
            if selection != "representative":
                raise ValueError("event_index requires representative selection")
            if not isinstance(event_index, (int, np.integer)) or not 0 <= event_index < len(representatives):
                raise ValueError("event_index is outside the detected event range")
            records = [representatives[event_index]]
        else:
            records = representatives if selection == "representative" else ranked
        # Keep a single full-grid diagnostic separate from the two-panel overview.
        count = 1 if event_index is not None else 2
        if axes is None:
            _, axes = plt.subplots(count, 1, sharex=True, figsize=(10, 4 if count == 1 else 7))
        axes = np.asarray(axes).reshape(-1)
        if len(axes) != count:
            raise ValueError(f"Provide {count} axes")
        for ax in axes:
            ax.plot(self.time, self.flux, ".", ms=3, color="0.6", label="Observations")
            ax.set_ylabel("Working flux")
        if count == 2:
            axes[0].plot(self.time, self.mean, color="C0", label="GP excluding all detections")
            axes[0].fill_between(self.time, self.mean-2*self.std, self.mean+2*self.std, color="C0", alpha=.15)
            for a, b in self.intervals:
                axes[0].axvspan(self.time[a], self.time[b-1], color="C3", alpha=.12)
            axes[0].legend(fontsize="small")
            axes[0].set_title("Overview: condition outside the detected union")
        # Draw exact scored interval predictions or one full-grid conditional prediction.
        ax = axes[-1]
        for i, record in enumerate(records):
            a, b = record["start"], record["stop"]
            # Preserve the event color when displaying its full-grid prediction.
            color = f"C{(event_index if event_index is not None else i) % 10}"
            x = self.time if count == 1 else self.time[a:b]
            mean = record["full_mean"] if count == 1 else record["mean"]
            std = record["full_std"] if count == 1 else record["std"]
            ax.plot(x, mean, color=color, label=f"[{a}, {b}): {self.settings['metric']}={record['score']:.2g}")
            ax.fill_between(x, mean-2*std, mean+2*std, color=color, alpha=.18)
            ax.plot(self.time[a:b], self.flux[a:b], ".", ms=4, color=color)
            if count == 1:
                ax.axvspan(self.time[a], self.time[b-1], color=color, alpha=.07)
        ax.set_title("Each GP excludes only its colored window; bands: ±2 observation std")
        if records:
            ax.legend(fontsize="small")
        else:
            ax.text(.5, .9, "No selected intervals", transform=ax.transAxes, ha="center")
        ax.set_xlabel(f"Time [days; {self.metadata.get('time_format', 'input reference')}]")
        axes[0].figure.tight_layout()
        return axes[0].figure, axes

    def _plot_residuals(self, axes=None):
        """Plot data/fit and residuals; return ``(figure, axes)`` without displaying.

        :param axes: Two axes for the diagnostic panels; None creates a new figure.
        :type axes: sequence of matplotlib.axes.Axes or None
        :returns: Figure and axes handles; no display is triggered.
        :rtype: tuple
        """
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
        axes = np.asarray(axes).ravel()
        if len(axes) != 2:
            raise ValueError("Provide two axes")
        # Show observations, the GP mean, and its predictive uncertainty.
        axes[0].plot(self.time, self.flux, ".", ms=2, color="0.35", label="observations")
        axes[0].plot(self.time, self.mean, color="C0", label="GP mean")
        axes[0].fill_between(
            self.time,
            self.mean - 2 * self.std,
            self.mean + 2 * self.std,
            color="C0",
            alpha=0.2,
            label="±2 predictive std",
        )
        axes[1].plot(self.time, self.residuals, ".", ms=2, color="0.35")
        axes[1].axhline(0, color="C0", lw=1)
        # Highlight the same detected intervals in the fit and residual panels.
        for a, b in self.intervals:
            for ax in axes:
                ax.axvspan(self.time[a], self.time[b - 1], color="C3", alpha=0.18)
        axes[0].set(ylabel="Working flux", title=f"{self.method.title()} detection")
        axes[0].legend(fontsize="small")
        axes[1].set(
            xlabel=f"Time [days; {self.metadata.get('time_format', 'input reference')}]",
            ylabel="Residual",
        )
        axes[0].figure.tight_layout()
        return axes[0].figure, axes

    def export(self, prefix):
        """Write ``prefix.npz``, sample/candidate CSVs and JSON provenance; return paths.

        Conditional arrays use NPZ keys ``conditional_{index}_{field}``; JSON records
        map each index to candidate bounds and scores. Fields are mean, std, covariance,
        full_mean and full_std. No live model or pickle is needed to read these arrays.
        Existing output files with the same prefix are overwritten.

        :param prefix: Output filename prefix; required suffixes are appended and parent directories
            are created.
        :type prefix: str or pathlib.Path
        :returns: Paths for the NPZ, samples CSV, candidates CSV, and JSON exports.
        :rtype: dict
        """
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        # Build consistent output paths for numerical arrays, tabular data, and provenance.
        paths = {
            key: Path(str(prefix) + suffix)
            for key, suffix in [
                ("arrays", ".npz"),
                ("samples", ".csv"),
                ("metadata", ".json"),
                ("candidates", "_candidates.csv"),
            ]
        }
        # Save full-resolution arrays in NPZ and readable sample columns in CSV.
        # Store ragged interval arrays under indexed keys without object arrays or pickle.
        conditional_arrays = {}
        for index, record in enumerate(self.conditional_predictions):
            for key in ("mean", "std", "covariance", "full_mean", "full_std"):
                conditional_arrays[f"conditional_{index}_{key}"] = record[key]
        np.savez_compressed(
            paths["arrays"],
            time=self.time,
            flux=self.flux,
            mean=self.mean,
            std=self.std,
            mask=self.mask,
            residuals=self.residuals,
            **conditional_arrays,
        )
        np.savetxt(
            paths["samples"],
            np.c_[self.time, self.flux, self.mean, self.std, self.mask],
            delimiter=",",
            header="time,flux,mean,std,detected",
            comments="",
        )
        import csv

        # Write candidate scores with a header even when no candidates were tested.
        with paths["candidates"].open("w", newline="") as handle:
            keys = list(self.candidates[0]) if self.candidates else ["start", "stop", "score"]
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.candidates)
        from . import __version__

        # Record software version, search settings, events, and inherited provenance.
        paths["metadata"].write_text(
            json.dumps(
                dict(
                    version=__version__,
                    method=self.method,
                    settings=self.settings,
                    runtime=self.runtime,
                    events=self.events,
                    conditional_predictions=[dict(index=i, start=c["start"], stop=c["stop"],
                        score=c["score"], mean_residual=c["mean_residual"])
                        for i, c in enumerate(self.conditional_predictions)],
                    metadata=self.metadata,
                ),
                indent=2,
                allow_nan=False,
            )
        )
        return paths
