import numpy as np
import matplotlib.pyplot as plt
import Balmung
import lightkurve

from nahiku.src.ExhaustiveSearch import ExhaustiveSearch
from nahiku.src.GreedySearch import GreedySearch
from nahiku.src.utils import freq_idx_to_period_days

from scipy.signal import find_peaks, periodogram, windows, peak_prominences
from scipy.ndimage import gaussian_filter1d

class Dipper:
    def __init__(
            self,
            time,
            flux,
            anomalies={},
            prominence=50,
            plot_dominant_period=False
        ):

        self.time = time
        self.flux = flux
        self.anomalies = anomalies # Will carry lists of true, injected, and identified anomalies, with keys "true", "injected", and "identified"

        self.dominant_period = self.get_dominant_period(prominence=prominence, plot=plot_dominant_period)

    @staticmethod
    def from_synthetic_parameterized(...):
        return Dipper(artificial_time, artificial_flux, ...)

    @staticmethod
    def from_synthetic_sampled(....):
        return Dipper(artificial_time, artificial_flux, ...)
    
    @staticmethod
    def from_lightkurve(
        target, 
        radius=None, 
        exptime=None, 
        cadence=None, 
        mission=('Kepler', 'K2', 'TESS'), 
        author=None, 
        quarter=None, 
        month=None, 
        campaign=None, 
        sector=None, 
        limit=None
    ):
        """
        Load a light curve using the lightkurve.search_lightcurve function, with options to specify various parameters for the search.
        Documentation for lightkurve.search_lightcurve parameters can be found here: https://lightkurve.github.io/lightkurve/reference/api/lightkurve.search_lightcurve.html
        All parameters are copied from lightkurve.search_lightcurve. Only target is required, which is the first parameter of lightkurve.search_lightcurve.
        """
        lc = (
            lightkurve.search_lightcurve(
                target, 
                mission=mission, 
                author=author, 
                exptime=exptime, 
                cadence=cadence, 
                quarter=quarter, 
                month=month, 
                campaign=campaign, 
                sector=sector, 
                radius=radius, 
                limit=limit
            )
            .download_all()
            .stitch()
            .remove_nans()
        )

        # Sort array and initialize variables
        time, flux = lc.time.value, np.array(lc.flux.value) - 1
        m = np.argsort(time)
        time, flux = time[m], flux[m]

        return Dipper(time, flux)

    def greedy_search(self, ...):
        s = GreedySearch(self.time, self.flux, self.dominant_period)

        return s

    def exhaustive_search(self, ...):
        s = ExhaustiveSearch(self.time, self.flux, self.dominant_period)

        return s

    def plot(self):
        pass

    def prewhiten(self, ....):
        prew_t, prew_f = Balmung.()
        return Dipper(prew_t, prew_f)

    def get_dominant_period(self, prominence=50, plot=False):
        """
        Function to calculate the dominant period of a light curve using the periodogram and peak detection. 
        It also includes an option to plot the periodogram and the light curve with the dominant period sinusoid.
        
        :param prominence (int): minimum prominence of peaks to consider in the periodogram (default: 50)
        :param plot (bool): whether to plot the periodogram and light curve (default: False)
        """

        # Check if data is standardized
        if np.std(self.flux) != 1:
            UserWarning(
                "Data is not standardized, so internally this function will standardize it for periodogram calculation."
            )
            self.flux = (self.flux - np.mean(self.flux)) / np.std(self.flux)

        # Get peaks in power spectrum
        freqs, power = periodogram(self.flux)
        peaks, _ = find_peaks(power, prominence=prominence)

        if len(peaks) == 0:
            print(f"No peaks found in power spectrum, using shoulder instead. Maximum dominant period is {self.time[-1]:.2f} days")
            smooth_power = gaussian_filter1d(power, 2)
            slope = np.gradient(smooth_power, freqs)
            shoulder_idx = np.where(slope < 0)[0][0]
            dominant_period = min(freq_idx_to_period_days(freqs[shoulder_idx], self.time), self.time[-1])

        else:
            # Filter to most prominent peak
            prominences, left_bases, right_bases = peak_prominences(power, peaks, wlen=5)

            # If the left_base is 0 or the right_base is the last index, the peak is at the edge of the periodogram. Then we remove it
            valid_peaks = np.where((left_bases != 0) & (right_bases != len(power) - 1))
            if valid_peaks[0].shape[0] == 0:
                print(
                    "No valid peaks found according to criteria that base is not at edge of periodogram. Thus we keep all peaks"
                )
            else:
                peaks = peaks[valid_peaks]
                left_bases = left_bases[valid_peaks]
                right_bases = right_bases[valid_peaks]

            max_peak = np.argmax(power[peaks])
            dominant_period = freq_idx_to_period_days(freqs[peaks[max_peak]], self.time)

        # Plot periodogram
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            axs[0].plot(freq_idx_to_period_days(freqs, self.time), power, label="Periodogram")
            if len(peaks) > 0:
                axs[0].plot(
                    freq_idx_to_period_days(freqs[peaks], self.time),
                    power[peaks],
                    "x",
                    label="Peaks",
                )
                axs[0].plot(
                    freq_idx_to_period_days(freqs[left_bases], self.time),
                    power[left_bases],
                    "o",
                    c="gray",
                    label="Right bases",
                )  # Reversed bc period = 1/frequency
                axs[0].plot(
                    freq_idx_to_period_days(freqs[right_bases], self.time),
                    power[right_bases],
                    "o",
                    c="black",
                    label="Left bases",
                )  # Reversed bc period = 1/frequency
            else:
                axs[0].plot(
                    freq_idx_to_period_days(freqs[shoulder_idx:], self.time),
                    power[shoulder_idx:],
                    "x",
                    label="Shoulder",
                )
            axs[0].legend()
            axs[0].set_xscale("log")
            axs[0].set_xlabel("Period [days]")
            axs[0].set_ylabel("Power")
            axs[0].set_title(f"Periodogram with max peak at {dominant_period:.2f} days")

            # Plot lightcurve with dominant period sinusoid
            axs[1].scatter(self.time, self.flux, s=2, label="Lightcurve")
            axs[1].plot(
                self.time,
                np.sin(2 * np.pi * self.time / dominant_period) + 4,
                c="darkorange",
                label=f"Dominant period: {dominant_period:.2f} days",
            )
            axs[1].set_xlabel("Time [days]")
            axs[1].set_ylabel("Flux")
            axs[1].legend()

            plt.tight_layout()
            plt.show()

        return dominant_period

    def inject_anomaly(self, ...):
        pass

    @staticmethod
    def generate_anomaly(...):
        pass

    def check_identified_anomalies(...):
        pass

