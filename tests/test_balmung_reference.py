"""Verify the default engine against the frozen collaborator-provided implementation."""

import ast
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
from nahiku import Nahiku
from nahiku.balmung import Balmung

REFERENCE = Path(__file__).parent / "reference/balmung_original.py"


def test_engine_matches_collaborator_algorithm_except_documented_boundary_fix():
    baseline = ast.parse(REFERENCE.read_text())
    current = ast.parse(inspect.getsource(Balmung))
    baseline_class = next(n for n in baseline.body if isinstance(n, ast.ClassDef))
    current_class = current.body[0]
    # Documentation/comments may evolve; numerical statements and signatures must not drift.
    for tree in (baseline_class, current_class):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.body:
                first = node.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    node.body.pop(0)
    peak = next(
        n
        for n in baseline_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "find_highest_peak"
    )
    for node in ast.walk(peak):
        if isinstance(node, ast.Assign) and ast.unparse(node) == "nu_peak = p[imax]":
            node.value.value.id = "nu"
    assert ast.dump(baseline_class) == ast.dump(current_class)


@pytest.mark.parametrize("components", [1, 2])
def test_default_matches_frozen_balmung_fit(components):
    spec = importlib.util.spec_from_file_location("balmung_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    time = np.linspace(0, 10, 1001)
    flux = 12 + 0.5 * np.cos(2 * np.pi * 7.3 * time + 0.4)
    if components == 2:
        flux += 0.2 * np.cos(2 * np.pi * 8.1 * time - 0.6)
    reference = module.Balmung(time, flux)
    reference.prewhiten(fmin=5, fmax=10, maxiter=components, minimum_snr=4, diagnose=False)
    lc = Nahiku(time, flux)
    result = lc.prewhiten(fmin=5, fmax=10, max_components=components)
    np.testing.assert_allclose(
        np.c_[result.frequencies, result.amplitudes, result.phases],
        reference.removed,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(result.residuals, reference.residual + np.median(flux), atol=1e-12)
    assert lc.history[-1]["samples_per_peak"] == 5


def test_optimizer_failure_does_not_return_partial_success(monkeypatch):
    time = np.linspace(0, 10, 501)
    flux = np.cos(2 * np.pi * 7.3 * time)
    lc = Nahiku(time, flux)

    def fail(self, theta):
        raise RuntimeError("fit did not converge")

    monkeypatch.setattr(Balmung, "fit", fail)
    with pytest.raises(RuntimeError, match="fit did not converge"):
        lc.prewhiten(fmin=5, fmax=10)
    np.testing.assert_array_equal(lc.flux, flux)
    assert lc.history == []


def test_sampling_override_is_not_silently_applied_to_balmung():
    lc = Nahiku(np.linspace(0, 10, 100), np.ones(100))
    with pytest.raises(ValueError, match="fixed fivefold sampling"):
        lc.prewhiten(fmin=1, fmax=5, samples_per_peak=8)
    result = lc.prewhiten(fmin=1, fmax=5, method="weighted", samples_per_peak=8)
    assert result.method == "weighted"
