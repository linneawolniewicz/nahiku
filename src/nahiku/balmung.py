"""
This code originally developed by Dan Hey at https://github.com/danhey/balmung/blob/master/balmung/balmung.py
This file is used with permission for the development of the Nahiku package.
The numerical algorithm follows the collaborator-provided implementation recorded
in commit dac885e1f48100572a843e639b15a80f8b06b834. Local changes are documentation,
comments, and returning frequency instead of amplitude for a boundary peak.
See THIRD_PARTY_NOTICES.md for attribution and license.
"""

import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import tqdm

class Balmung:
    def __init__(self, time: np.ndarray, flux: np.ndarray):
        """Initialize median-centered observations and empty prewhitening state.

        :param time: Timestamps in days, on the input time reference.
        :type time: array-like
        :param flux: Flux samples in consistent linear units, aligned with time.
        :type flux: array-like
        :returns: None
        :rtype: None
        """
        self.time = time
        self.flux = flux - np.median(flux)
        self.residual = np.copy(self.flux)
        self.removed = []

    def prewhiten(self, fmin=None, fmax=None, minimum_snr=0, maxiter=200, diagnose=True):
        # Calculate initial amplitude spectrum
        """Iteratively remove fitted pulsations using Balmung’s fixed initial amplitude threshold.

        :param fmin: Lower frequency bound; None uses the inverse baseline.
        :type fmin: float or None
        :param fmax: Upper frequency bound; None uses half the inverse median cadence.
        :type fmax: float or None
        :param minimum_snr: Amplitude cutoff multiplier of the initial median spectrum.
        :type minimum_snr: float
        :param maxiter: Maximum number of removal iterations.
        :type maxiter: int
        :param diagnose: Reserved diagnostic flag; currently has no effect.
        :type diagnose: bool
        :returns: None; residuals and removed parameters are updated in place.
        :rtype: None

        The component fit jointly optimizes frequency, amplitude, and phase.
        Frequency spectra use the original fivefold oversampling. Optimizer errors
        propagate to the caller; no partial result is silently returned.
        """
        freq, amp = self.amplitude_spectrum(self.time, self.residual, fmin=fmin, fmax=fmax)

        # # Estimate noise level:
        # bkg = self.estimate_background(freq, amp)
        # snr = amp / bkg

        # Get first guess:
        # idx = np.nanargmax(snr)
        # f0, a0 = freq[idx], amp[idx]
        # phi0 =

        noise_level = minimum_snr * np.median(amp)
        f0, a0, phi0 = self.initialize_guess(fmin=fmin, fmax=fmax)

        for i in tqdm.tqdm(range(maxiter)):
            if a0 > noise_level:
                # Fit theta to lc
                popt = self.fit([f0, a0, phi0])
                self.removed.append(popt.tolist())

                # Subtract off the fitted model
                self.residual -= self.model(self.time, *popt)

                # Get new params for next iteration
                f0, a0, phi0 = self.initialize_guess(fmin=fmin, fmax=fmax)
            else:
                break

        if diagnose:
            pass
            # Diagnostic plotting is available through the plotting methods.

    def fit(self, theta: list) -> np.ndarray:
        """Fit one cosine component to the current residuals.

        :param theta: Initial frequency, amplitude, and phase guesses.
        :type theta: array-like
        :returns: Fitted frequency, nonnegative amplitude, and phase.
        :rtype: numpy.ndarray
        """
        popt, _ = curve_fit(
            self.model, self.time, self.residual, p0=theta, jac=self.grad_model
        )

        # I don't expect this to happen.. but if the amplitude goes negative let's fix it:
        if popt[1] < 0:
            popt[1] *= -1.
            popt[2] += np.pi

        return popt

    def grad_model(
        self, time: np.ndarray, freq: float, amp: float, phi: float
    ) -> np.ndarray:
        """Evaluate the cosine model Jacobian.

        :param time: Timestamps in days, on the input time reference.
        :type time: array-like
        :param freq: Frequency in cycles per day.
        :type freq: float
        :param amp: Cosine amplitude in flux units.
        :type amp: float
        :param phi: Cosine phase in radians.
        :type phi: float
        :returns: One row per sample and columns for frequency, amplitude, and phase.
        :rtype: numpy.ndarray
        """
        factor = 2 * np.pi * freq * time + phi
        return np.array(
            [
                -2 * np.pi * amp * time * np.sin(factor),
                np.cos(factor),
                -1 * amp * np.sin(factor),
            ]
        ).T

    def model(self, time:np.ndarray, freq:float, amp:float, phi:float) -> np.ndarray:
        """Evaluate a single cosine pulsation.

        :param time: Timestamps in days, on the input time reference.
        :type time: array-like
        :param freq: Frequency in cycles per day.
        :type freq: float
        :param amp: Cosine amplitude in flux units.
        :type amp: float
        :param phi: Cosine phase in radians.
        :type phi: float
        :returns: Model flux at each timestamp.
        :rtype: numpy.ndarray
        """
        return amp * np.cos((2 * np.pi * freq * time) + phi)

    def estimate_background(
        self, x: np.ndarray, y: np.ndarray, log_width: float = 0.01
    ) -> np.ndarray:
        """Average overlapping logarithmic-window medians.

        :param x: Positive, increasing frequency grid.
        :type x: numpy.ndarray
        :param y: Amplitude values aligned with x.
        :type y: numpy.ndarray
        :param log_width: Half-width of each window in log10 frequency.
        :type log_width: float
        :returns: Background estimate at each frequency; uncovered bins divide by zero.
        :rtype: numpy.ndarray
        """
        count = np.zeros(len(x), dtype=int)
        bkg = np.zeros_like(x)
        x0 = np.log10(x[0])
        while x0 < np.log10(x[-1]):
            m = np.abs(np.log10(x) - x0) < log_width
            bkg[m] += np.median(y[m])
            count[m] += 1
            x0 += 0.5 * log_width
        return bkg / count

    def find_highest_peak(self, f: np.ndarray, a: np.ndarray) -> float:
        """Interpolate the largest spectral peak using its two neighbors.

        :param f: Frequency grid.
        :type f: numpy.ndarray
        :param a: Amplitude spectrum aligned with f.
        :type a: numpy.ndarray
        :returns: Interpolated peak frequency, or the grid frequency at a boundary peak.
        :rtype: float
        """
        nu, p = f, a
        nu, p = np.atleast_1d(nu, p)

        # Get index of highest peak.
        imax = np.argmax(p)

        # Determine the frequency value by parabolic interpolation
        if imax == 0 or imax == p.size - 1:
            # At a boundary, select its frequency rather than its amplitude.
            nu_peak = nu[imax]
        else:
            # Get the neighboring samples around the spectral maximum.
            frq1 = nu[imax - 1]
            frq2 = nu[imax]
            frq3 = nu[imax + 1]
            y1 = p[imax - 1]
            y2 = p[imax]
            y3 = p[imax + 1]

            # Parabolic interpolation formula.
            t1 = (y2 - y3) * (frq2 - frq1) ** 2 - (y2 - y1) * (frq2 - frq3) ** 2
            t2 = (y2 - y3) * (frq2 - frq1) - (y2 - y1) * (frq2 - frq3)
            nu_peak = frq2 - 0.5 * t1 / t2
        return nu_peak

    def amplitude_spectrum(
        self, t, y, fmin: float = None, fmax: float = None, oversample_factor: float = 5.0,
    ) -> tuple:
        """Calculate a Lomb–Scargle amplitude spectrum.

        :param t: Observation timestamps in days.
        :type t: numpy.ndarray
        :param y: Observed flux aligned with t.
        :type y: numpy.ndarray
        :param fmin: Minimum frequency; None uses the inverse baseline.
        :type fmin: float or None
        :param fmax: Maximum frequency; None uses half the inverse median cadence.
        :type fmax: float or None
        :param oversample_factor: Frequency-grid oversampling relative to the inverse baseline.
        :type oversample_factor: float
        :returns: Frequency and amplitude arrays.
        :rtype: tuple
        """
        # t, y = self.time, self.residual
        tmax = t.max()
        tmin = t.min()
        df = 1.0 / (tmax - tmin)

        if fmin is None:
            fmin = df
        if fmax is None:
            fmax = 0.5 / np.median(np.diff(t))  # *nyq_mult

        freq = np.arange(fmin, fmax, df / oversample_factor)
        model = LombScargle(t, y)
        sc = model.power(freq, method="fast", normalization="psd")

        fct = np.sqrt(4.0 / len(t))
        amp = np.sqrt(sc) * fct

        return freq, amp

    def dft_phase(self, x: np.ndarray, y: np.ndarray, f: float) -> float:
        """Estimate phase from sine and cosine Fourier projections.

        :param x: Timestamps in days.
        :type x: numpy.ndarray
        :param y: Flux aligned with x.
        :type y: numpy.ndarray
        :param f: Frequency in cycles per day.
        :type f: float
        :returns: Fourier phase in radians.
        :rtype: float
        """
        expo = 2.0 * np.pi * f * x
        return np.arctan2(np.sum(y * np.sin(expo)), np.sum(y * np.cos(expo)))

    def initialize_guess(self, fmin: float, fmax: float):
        """Estimate the strongest component frequency, amplitude, and phase.

        :param fmin: Positive lower frequency bound in cycles per day.
        :type fmin: float
        :param fmax: Upper frequency bound in cycles per day; greater than fmin.
        :type fmax: float
        :returns: Initial frequency, amplitude, and phase.
        :rtype: tuple
        """
        time, flux = self.time, self.residual
        f, a = self.amplitude_spectrum(time, flux, fmin=fmin, fmax=fmax, oversample_factor=5.0)

        # Get freq of max power using parabolic interpolation
        f0 = self.find_highest_peak(f, a)
        # Calculate a0 at f0
        a0 = np.sqrt(
            LombScargle(time, flux).power(f0, method="fast", normalization="psd")
        ) * np.sqrt(4.0 / len(time))
        # Calculate phi0, since ASTC needs to be negative
        phi0 = -1 * self.dft_phase(time, flux, f0)
        return f0, a0, phi0

    def plot_lc(self, ax=None):
        """Plot the median-centered input light curve.

        :param ax: Axes to draw on; None creates a new figure and axes.
        :type ax: matplotlib.axes.Axes or None
        :returns: Axes containing the light curve.
        :rtype: matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.time, self.flux)
        ax.set_xlabel("Time")
        ax.set_ylabel("Flux")
        return ax

    def plot_residual(self, ax=None):
        """Plot the current prewhitened residuals.

        :param ax: Axes to draw on; None creates a new figure and axes.
        :type ax: matplotlib.axes.Axes or None
        :returns: Axes containing residual flux.
        :rtype: matplotlib.axes.Axes
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.time, self.residual)
        ax.set_xlabel("Time")
        ax.set_ylabel("Flux")
        return ax

    def plot(self, ax=None):
        """Plot the original amplitude spectrum and removed components.

        :param ax: Axes to draw on; None creates a new figure and axes.
        :type ax: matplotlib.axes.Axes or None
        :returns: Axes containing the spectrum; requires at least one removed component.
        :rtype: matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots()
        f, a = self.amplitude_spectrum(self.time, self.flux)
        ax.plot(f, a, lw=0.7, c="black")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        rem = np.array(self.removed)
        ax.plot(rem[:,0], rem[:,1], 'v', alpha=0.7)
        return ax

