import numpy as np
import matplotlib.pyplot as plt
import lightkurve
import warnings

from nahiku.src.ExhaustiveSearch import ExhaustiveSearch
from nahiku.src.GreedySearch import GreedySearch
from nahiku.src.nahiku_helpers import freq_idx_to_period_days

from balmung import Balmung
from scipy.signal import find_peaks, periodogram, windows, peak_prominences
from scipy.ndimage import gaussian_filter1d

class Nahiku:
    def __init__(
            self,
            time,
            flux,
            anomalies={"true": [], "injected": [], "identified": []},
            prominence=50,
            plot_dominant_period=False
        ):
        """
        Initialize a Nahiku object with time and flux arrays, and an optional anomalies dictionary to keep track of true, injected, and identified anomalies.

        :param time (1D array-like): Array of time points corresponding to the light curve data.
        :param flux (1D array-like): Array of flux values corresponding to the light curve data.
        :param anomalies (dict): Dictionary to keep track of true, injected, and identified anomalies, with keys "true", "injected", and "identified". 
                Each key should map to a list of indices corresponding to the anomalies in the time and flux arrays (default: {"true": [], "injected": [], "identified": []}).
        :param prominence (int): minimum prominence of peaks to consider in the periodogram for calculating the dominant period (default: 50)
        :param plot_dominant_period (bool): whether to plot the periodogram and light curve with the dominant period sinusoid when calculating the dominant period (default: False)
        """

        self.time = time
        self.flux = flux
        self.anomalies = anomalies # Will carry lists of true, injected, and identified anomalies, with keys "true", "injected", and "identified"
        
        self.dominant_period = self.get_dominant_period(prominence=prominence, plot=plot_dominant_period)

    @staticmethod
    def from_lightkurve(
        **kwargs
    ):
        """
        Load a light curve using the lightkurve.search_lightcurve function, with options to specify various parameters for the search.
        Documentation for lightkurve.search_lightcurve parameters can be found here: https://lightkurve.github.io/lightkurve/reference/api/lightkurve.search_lightcurve.html

        :param **kwargs: Additional keyword arguments to pass to lightkurve.search_lightcurve and Nahiku.__init__
        """
        # Create a dict of init args, falling back to defaults if not in kwargs
        init_keys = ['anomalies', 'prominence', 'plot_dominant_period']
        init_args = {k: kwargs.pop(k) for k in init_keys if k in kwargs}

        lc = (
            lightkurve.search_lightcurve(
                **kwargs
            )
            .download_all()
            .stitch()
            .remove_nans()
        )

        # Sort array and initialize variables
        time, flux = lc.time.value, np.array(lc.flux.value) - 1
        m = np.argsort(time)
        time, flux = time[m], flux[m]

        return Nahiku(time, flux, **init_args)

    @staticmethod
    def from_synthetic_parameterized(
        rednoise_amp=1.0,
        whitenoise_amp=1.0,
        num_steps=1000,
        seed=48,
        period=None,
        phase=None,
        amp=None,
        slope=None,
        rednoise_time_scale=None,
        random_noise_step_loc=None,
        **kwargs
    ):
        """
        Generate a synthetic light curve with a sinusoidal signal, white noise, red noise, a step function anomaly, and a linear trend.

        :param rednoise_amp (float): amplitude of red noise (default: 1.0)
        :param whitenoise_amp (float): amplitude of white noise (default: 1.0)
        :param num_steps (int): number of time steps in the light curve (default: 1000)
        :param seed (int): random seed for reproducibility (default: 48)
        :param period (float): period of the sinusoidal signal (default: randomly chosen between 175 and 225)
        :param phase (float): phase of the sinusoidal signal (default: randomly chosen between 0 and 2*pi)
        :param amp (float): amplitude of the sinusoidal signal (default: randomly chosen between 0 and 0.9)
        :param slope (float): slope of the linear trend (default: randomly chosen between -0.001 and 0.001)
        :param rednoise_time_scale (float): correlation time scale of the red noise (default: randomly chosen between 5 and 15)
        :param random_noise_step_loc (float): location of the step function anomaly (default: randomly chosen between 0 and num_steps)
        :param **kwargs: Additional keyword arguments to pass to Nahiku.__init__
        """

        if num_steps < 0:
            warnings.warn("Number of steps must be non-negative. Defaulting to its absolute value.")
            num_steps = abs(num_steps)
        
        x = np.arange(num_steps)
        rng = np.random.default_rng(seed=seed)

        # Synthetic lightcurve
        if period is None:
            period = 175 + 50 * rng.random()  # randomly chosen period of lightcurve

        if phase is None:
            phase = 2 * np.pi * rng.random()  # randomly chosen phase

        if amp is None:
            amp = 0.9 * rng.random()  # randomly chosen amplitude

        lightcurve = amp * np.cos(2 * np.pi * x / period + phase)

        # White noise
        whitenoise = whitenoise_amp * rng.random(num_steps)

        # Red noise
        if rednoise_time_scale is None:
            rednoise_time_scale = rng.integers(5, 15)  # correlation time scale of red noise

        rednoise = np.convolve(
            rng.random(2 * num_steps), windows.gaussian(int(4 * rednoise_time_scale), rednoise_time_scale)
        )

        x1 = int(len(rednoise) / 2) - int(num_steps / 2)
        x2 = x1 + num_steps

        rednoise = rednoise[x1:x2]
        rednoise = rednoise * rednoise_amp / np.std(rednoise)

        # Step parameters
        if random_noise_step_loc is None:
            random_noise_step_loc = num_steps * rng.random()  # location of step

        step_amp = rng.uniform(
            -5 * np.std(rednoise), -1 * np.std(rednoise)
        )  # amplitude of anomaly

        step_width = rng.integers(int(0.001 * num_steps), int(0.01 * num_steps))
        step = step_amp * (x > random_noise_step_loc) * (x < (random_noise_step_loc + step_width))

        # Trend parameters
        if slope is None:
            slope = 0.001 - 0.002 * rng.random()  # slope of trend

        trend = slope * (x - num_steps / 2)

        # Combine
        y = lightcurve + whitenoise + rednoise + step + trend

        return Nahiku(x, y, **kwargs)

    @staticmethod
    def from_synthetic_sampled(....):
        return Nahiku(artificial_time, artificial_flux, ...)

    def greedy_search(self, **kwargs):
        """
        Initialize and run a GreedySearch. 
        Accepts all arguments for GreedySearch.__init__ and GreedySearch.search_for_anomaly.
        """
        
        # Pop out keys that are needed for GreedySearch.__init__, and pass the rest to search_for_anomaly
        init_keys = [
            'device', 'which_grow_metric', 'y_err', 'num_sigma_threshold', 'expansion_param', 'len_deviant'
        ]
        
        # Create a dict of init args, falling back to defaults if not in kwargs
        init_args = {k: kwargs.pop(k) for k in init_keys if k in kwargs}

        search = GreedySearch(
            self.time, 
            self.flux, 
            self.dominant_period, 
            **init_args
        )

        # Pass remaining kwargs to search_for_anomaly, which will handle defaults for those
        search.search_for_anomaly(**kwargs)

        # Add flagged anomalous indices to self.anomalies['identified']
        flagged_indices = np.nonzero(search.flagged_anomalous)
        self.anomalies['identified'].extend(flagged_indices[0].tolist())

        return search

    def exhaustive_search(self, **kwargs):
        """
        Initialize and run a ExhaustiveSearch. 
        Accepts all arguments for ExhaustiveSearch.__init__ and ExhaustiveSearch.search_for_anomaly.
        """
        
        # Pop out keys that are needed for ExhaustiveSearch.__init__, and pass the rest to search_for_anomaly
        init_keys = [
            'device', 'min_anomaly_len', 'max_anomaly_len', 'window_slide_step', 'window_size_step', 'assume_independent', 'which_test_metric'
        ]
        
        # Create a dict of init args, falling back to defaults if not in kwargs
        init_args = {k: kwargs.pop(k) for k in init_keys if k in kwargs}

        search = ExhaustiveSearch(
            self.time, 
            self.flux, 
            self.dominant_period, 
            **init_args
        )

        # Pass remaining kwargs to search_for_anomaly, which will handle defaults for those
        search.search_for_anomaly(**kwargs)

        # Add flagged anomalous indices to self.anomalies['identified']
        flagged_indices = np.nonzero(search.flagged_anomalous)
        self.anomalies['identified'].extend(flagged_indices[0].tolist())

        return search

    def plot(self):
        """
        Plot the light curve using matplotlib, with time on the x-axis and flux on the y-axis.
        """

        plt.figure(figsize=(8, 5))
        plt.plot(self.time, self.flux, c='k', ms=3, alpha=0.5, label="Light Curve")

        # If there are any anomaly indices in self.anomalies, plot them with different colors
        # Plot identified anaomlies as red, and highlight injected anomaly areas in yellow and true anomaly areas in blue
        if self.anomalies['identified']:
            anomaly_locs = self.anomalies['identified']
            plt.scatter(self.time[anomaly_locs], self.flux[anomaly_locs], c='red', ms=5, alpha=0.7, label="Identified Anomaly")

        if self.anomalies['injected']:
            # For each consecutive sequence of injected anomalies, highlight the area in yellow with plt.axvspan
            injected_locs = self.anomalies['injected']
            for i in range(len(injected_locs) - 1):
                if injected_locs[i + 1] != injected_locs[i] + 1:
                    plt.axvspan(self.time[injected_locs[i]], self.time[injected_locs[i + 1]], color='yellow', alpha=0.3)

            # Add label for the legend (plot outside bounds of the light curve so it doesn't show up on the plot)
            plt.axvspan(self.time[-1] + 100, self.time[-1] + 200, color='yellow', alpha=0.3, label="Injected Anomaly")

        if self.anomalies['true']:
            # For each consecutive sequence of true anomalies, highlight the area in blue with plt.axvspan
            true_locs = self.anomalies['true']
            for i in range(len(true_locs) - 1):
                if true_locs[i + 1] != true_locs[i] + 1:
                    plt.axvspan(self.time[true_locs[i]], self.time[true_locs[i + 1]], color='blue', alpha=0.3)

            # Add label for the legend (plot outside bounds of the light curve so it doesn't show up on the plot)
            plt.axvspan(self.time[-1] + 200, self.time[-1] + 300, color='blue', alpha=0.3, label="True Anomaly")
            
        plt.xlim(self.time[0], self.time[-1])
        plt.ylim(min(self.flux) - 0.1 * np.ptp(self.flux), max(self.flux) + 0.1 * np.ptp(self.flux))

        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title("Light Curve")
        plt.legend()
        plt.show()

    def standardize(self):
        """
        Function to standardize the flux of the light curve by subtracting the mean and dividing by the standard deviation.
        This is important for the periodogram calculation and GP modeling, as it ensures that the data is on a consistent scale 
        and that the periodogram is not dominated by the mean flux level.
        """
        self.flux = (self.flux - np.mean(self.flux)) / np.std(self.flux)

    def prewhiten(self, plot=True, **kwargs):
        """
        Prewhiten a light curve using the balmung.prewhiten function, with options to specify various parameters for the removal of frequencies.
        Code for balmung.prewhiten can be found here: https://github.com/danhey/balmung/blob/master/balmung/balmung.py
        
        :param plot (bool): whether to plot the light curve before and after prewhitening (default: True)
        :param **kwargs: Additional keyword arguments to pass to balmung.prewhiten and Nahiku.__init__
        """
        # Create a dict of init args, falling back to defaults if not in kwargs
        init_keys = ['anomalies', 'prominence', 'plot_dominant_period']
        init_args = {k: kwargs.pop(k) for k in init_keys if k in kwargs}

        bm = Balmung(self.time, self.flux)

        if plot:
            print("Light curve before prewhitening:")
            bm.plot_lc()
        
        bm.prewhiten(**kwargs)
        
        if plot:
            print("Light curve after prewhitening:")
            bm.plot_residual()

        return Nahiku(bm.time, bm.residual, **init_args)

    def get_dominant_period(self, prominence=50, plot=False):
        """
        Function to calculate the dominant period of a light curve using the periodogram and peak detection. 
        It also includes an option to plot the periodogram and the light curve with the dominant period sinusoid.
        
        :param prominence (int): minimum prominence of peaks to consider in the periodogram (default: 50)
        :param plot (bool): whether to plot the periodogram and light curve (default: False)
        """

        # Check if data is standardized
        if np.std(self.flux) != 1:
            warnings.warn("Data is not standardized, so it will be standardize it for periodogram calculation.")
            self.standardize()

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

    def inject_anomaly(
        self, 
        num_anomalies,
        seed=48,
        shapes=["gaussian", "saw", "exocomet"],
        period_scale=None,
        snr=None,
        absolute_width=None,
        absolute_depth=None,  
        locs=None,
        alpha=1, 
    ):
        """
        Inject an anomaly into the light curve, with options to specify the number of anomalies, their shapes, widths, depths, and locations.

        :param num_anomalies (int): number of anomalies to inject
        :param seed (int): random seed for reproducibility (default: 48)
        :param shapes (list of str): list of shapes to choose from for the anomalies. 
                Options are "gaussian" for gaussian-shaped anomalies, "saw" for sawtooth-shaped anomalies, and "exocomet" for exocomet-shaped anomalies. 
                Default is ["gaussian", "saw", "exocomet"].
        :param period_scale (float or None): ratio of the dominant period to use as the width of the anomaly. If None, randomly chosen between 0.1 and 5 (default: None)
        :param snr (float or None): signal to noise ratio of the anomaly. If None, randomly chosen between 0.5 and 10 (default: None)
        :param absolute_width (float or None): absolute width of the anomaly. If specified, period_scale is ignored (default: None)
        :param absolute_depth (float or None): absolute depth of the anomaly. If specified, snr is ignored (default: None)
        :param locs (list of float or None): list of locations to inject anomalies at. If None, locations are randomly chosen (default: None)
        :param alpha (float): shape parameter for the exocomet profile, which controls the asymmetry of the anomaly. 
                Higher values of alpha result in a more asymmetric profile with a steeper ingress and a shallower egress (default: 1)
        """

        # Initialize
        rng = np.random.default_rng(seed=seed)
        num_steps = len(self.time)
        time_steps = np.arange(num_steps)
        anomaly = np.zeros(num_steps)
        anomaly_locs = []

        # If absolute_depth is given, use it as the depth of the anomaly
        if absolute_depth is not None:
            if absolute_depth < 0:
                warnings.warn("Absolute depth must be positive. Defaulting to its absolute value.")
                absolute_depth = abs(absolute_depth)

            anomaly_amp = -1 * absolute_depth

        else:
            # Create anomaly with snr if not given
            if snr is None:
                snr = rng.uniform(0.5, 10)  # depth of anomaly
                print(f"Anomaly absolute depth and snr were not specified. Using snr = {snr}")
            
            if snr < 0: 
                warnings.warn("SNR must be positive. Defaulting to its absolute value.")
                snr = abs(snr)

            # Create anomaly of amplitude corresponding to desired snr (using stdev for noise)
            # Note: because snr corresponds to noise, y does not need to be normalized or standardized
            noise = np.std(self.flux)
            signal = snr * noise
            anomaly_amp = -1 * signal

        # If absolute_width is given, use it as the width of the anomaly
        if absolute_width is not None:
            if absolute_width < 0: 
                warnings.warn("Absolute width must be positive. Defaulting to its absolute value.")
                absolute_width = abs(absolute_width)

            anomaly_width = absolute_width
            
        else:
            # Create anomaly period_scale if not given
            if period_scale is None:
                period_scale = rng.uniform(0.1, 5)  # period scaling of anomaly
                print(f"Anomaly absolutel width and period_scale were not specified. Using period_scale = {period_scale}")

            if period_scale < 0: 
                warnings.warn("Period scale must be positive. Defaulting to its absolute value.")
                period_scale = abs(period_scale)

            # Create anomaly_width from period of peak in power spectrum
            # minimum value of 1. Note this is the std dev. of the anomaly (assuming Gaussian)
            anomaly_period = period_scale * self.dominant_period
            anomaly_width = max(
                anomaly_period / (2 * np.sqrt(2 * np.log(2))), 1
            )  

        # Perform some checks of locs list if given
        if locs is not None:
            # Check locs is a list of floats
            if not isinstance(locs, list) or not all(isinstance(loc, (int, float)) for loc in locs):
                warnings.warn("Locs must be a list of floats. Defaulting to random locations.")
                locs = None
            
            # Check that locs are within the range of the time array
            if not all((loc >= self.time[0] and loc <= self.time[-1]) for loc in locs):
                warnings.warn("All locs must be within the range of the time array. Defaulting to random locations.")
                locs = None

        if locs is not None:
            # Check number of locs matches num_anomalies
            if len(locs) != num_anomalies:
                warnings.warn("Length of locs does not match num_anomalies. Defaulting to only using the first num_anomalies values in locs.")
                locs = locs[:num_anomalies]

        # Check that shapes is a list of strings and that all shapes are valid
        if not isinstance(shapes, list) or not all(isinstance(shape, str) for shape in shapes) or not all(shape in ["gaussian", "saw", "exocomet"] for shape in shapes):
            warnings.warn("Shapes must be a list of strings containing only 'gaussian', 'saw', and/or 'exocomet'. Defaulting to ['gaussian', 'saw', 'exocomet']")
            shapes = ["gaussian", "saw", "exocomet"]

        # Inject anomalies of anomaly_width and anomaly_amp at random locations (as many as in locs)
        for i in range(num_anomalies):

            # If locs is given, use the specified location. Otherwise, choose a random location for the anomaly
            if locs is not None:
                anomaly_loc = int(locs[i])
            else:
                anomaly_loc = int(num_steps * rng.random())

            anomaly_locs.append(anomaly_loc)
            shape = rng.choice(shapes)

            if shape == "gaussian":
                # Gaussian-shape anomaly at x0
                anomaly += anomaly_amp * np.exp(
                    -0.5 * ((time_steps - anomaly_loc) / anomaly_width) ** 2
                )
                anomaly_fwhm = 2.355 * anomaly_width  # True for gaussian-shaped anomalies

            elif shape == "saw":
                # Create anomaly that has a quick dip to anomaly_amp, then a slow rise back to 0 based on anomaly_width
                anomaly += anomaly_amp * (
                    1 - np.exp(-np.abs(time_steps - anomaly_loc) / anomaly_width)
                )
                anomaly_fwhm = 2 * anomaly_width  # Approximation for saw-shaped anomalies

            elif shape == "exocomet":
                # -A * exp(-t/tau) * (t/tau)^alpha
                # T = absolute time (in days) in the light curve
                # T0 = injection time (in days) in the light curve
                # t = T - T0 if T >= T0 else 0
                # A is an amplitude parameter, tau a width parameter, and alpha the shape parameter
                t = self.time - self.time[anomaly_loc]
                t = np.where(t >= 0, t, 0)  

                anomaly += anomaly_amp * np.exp((-1 * t) / anomaly_width) * (t / anomaly_width) ** alpha

                anomaly_fwhm = 2.45 * anomaly_width  # The FWHM is 2.45 tau for exocomet-shaped anomalies
                anomaly_amp = 0.37 * anomaly_amp  # Because the minimum of this function is at A/e, or 0.37A

        # Add anomaly_locs to self.anomalies['injected'] and add anomaly to flux
        self.anomalies['injected'].extend(anomaly_locs)
        self.flux += anomaly

        return anomaly_locs, anomaly, anomaly_amp, anomaly_fwhm

    def check_identified_anomalies(self):
        """
        Check the identified anomalies against the true and injected anomalies, and print out the results.
        """

        identified_set = set(self.anomalies['identified'])
        true_set = set(self.anomalies['true'])
        injected_set = set(self.anomalies['injected'])

        detected_true = identified_set.intersection(true_set)
        detected_injected = identified_set.intersection(injected_set)
        detected_true_injected = identified_set.intersection(true_set.intersection(injected_set))
        false_positives = identified_set - true_set - injected_set

        true_anomaly_tpr = len(detected_true) / len(true_set) if len(true_set) > 0 else 0
        injected_anomaly_tpr = len(detected_injected) / len(injected_set) if len(injected_set) > 0 else 0
        total_tpr = len(detected_true_injected) / len(true_set.union(injected_set)) if len(true_set.union(injected_set)) > 0 else 0
        false_positive_rate = len(false_positives) / (len(self.time) - len(true_set) - len(injected_set)) if (len(self.time) - len(true_set) - len(injected_set)) > 0 else 0

        print(f"True anomaly indices detected: {len(detected_true)} out of {len(true_set)} (TPR: {true_anomaly_tpr:.2f})")
        print(f"Injected anomaly indices detected: {len(detected_injected)} out of {len(injected_set)} (TPR: {injected_anomaly_tpr:.2f})")
        print(f"Total anomaly indices detected: {len(detected_true_injected)} out of {len(true_set.union(injected_set))} (TPR: {total_tpr:.2f})")
        print(f"False positive rate: {false_positive_rate:.2f}")

        return {
            "detected_true": detected_true,
            "detected_injected": detected_injected,
            "detected_true_injected": detected_true_injected,
            "false_positives": false_positives,
            "true_anomaly_tpr": true_anomaly_tpr,
            "injected_anomaly_tpr": injected_anomaly_tpr,
            "total_tpr": total_tpr,
            "false_positive_rate": false_positive_rate
        }

        
