import gpytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
import lightkurve
import warnings

from .exhaustive_search import ExhaustiveSearch
from .greedy_search import GreedySearch
from .gp_helpers import QuasiPeriodicKernel
from .balmung import Balmung


from scipy.signal import find_peaks, periodogram, windows, peak_prominences
from scipy.ndimage import gaussian_filter1d
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel
from gpytorch.distributions import MultivariateNormal


class Nahiku:
    """
    This class represents a light curve and provides methods for anomaly detection using greedy and exhaustive search, as well as methods for plotting,
    standardizing, prewhitening, calculating the dominant period, injecting anomalies, and checking the accuracy of identified anomalies against true and injected anomalies.
    """

    def __init__(
        self, time, flux, anomalies=None, prominence=50, plot_dominant_period=False
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

        # self.anomalies carry lists of true, injected, and identified anomalies, with keys "true", "injected", and "identified"
        if anomalies is None:
            self.anomalies = {"true": [], "injected": [], "identified": []}
        else:
            # Ensures that the anomalies are not carried around by multiple objects
            self.anomalies = anomalies.copy()

        self.dominant_period = self.get_dominant_period(
            prominence=prominence, plot=plot_dominant_period
        )

    @staticmethod
    def from_lightkurve(**kwargs):
        """
        Load a light curve using the lightkurve.search_lightcurve function, with options to specify various parameters for the search.
        Documentation for lightkurve.search_lightcurve parameters can be found here: https://lightkurve.github.io/lightkurve/reference/api/lightkurve.search_lightcurve.html

        :param kwargs: Additional keyword arguments to pass to lightkurve.search_lightcurve and Nahiku.__init__
        """
        # Create a dict of init args, falling back to defaults if not in kwargs
        init_keys = ["anomalies", "prominence", "plot_dominant_period"]
        init_args = {k: kwargs.pop(k) for k in init_keys if k in kwargs}

        lc = (
            lightkurve.search_lightcurve(**kwargs).download_all().stitch().remove_nans()
        )

        # Sort array and initialize variables
        time, flux = lc.time.value, np.array(lc.flux.value) - 1
        m = np.argsort(time)
        time, flux = time[m], flux[m]

        return Nahiku(time, flux, **init_args)

    @staticmethod
    def from_synthetic_parameterized_noise(
        num_days=100,
        num_steps=1000,
        seed=48,
        rednoise_amp=1.0,
        whitenoise_amp=1.0,
        period=None,
        phase=None,
        amp=None,
        slope=None,
        rednoise_time_scale=None,
        random_noise_step_loc=None,
        **kwargs,
    ):
        """
        Generate a synthetic light curve with a sinusoidal signal, white noise, red noise, a step function anomaly, and a linear trend.

        :param num_days (float): total duration of the light curve in days (default: 100)
        :param num_steps (int): number of time steps in the light curve (default: 1000)
        :param seed (int): random seed for reproducibility (default: 48)
        :param rednoise_amp (float): amplitude of red noise (default: 1.0)
        :param whitenoise_amp (float): amplitude of white noise (default: 1.0)
        :param period (float): period of the sinusoidal signal (default: randomly chosen between 175 and 225)
        :param phase (float): phase of the sinusoidal signal (default: randomly chosen between 0 and 2*pi)
        :param amp (float): amplitude of the sinusoidal signal (default: randomly chosen between 0 and 0.9)
        :param slope (float): slope of the linear trend (default: randomly chosen between -0.001 and 0.001)
        :param rednoise_time_scale (float): correlation time scale of the red noise (default: randomly chosen between 5 and 15)
        :param random_noise_step_loc (float): location of the step function anomaly (default: randomly chosen between 0 and num_steps)
        :param kwargs: Additional keyword arguments to pass to Nahiku.__init__
        """

        if num_steps < 0:
            warnings.warn(
                "Number of steps must be non-negative. Defaulting to its absolute value."
            )
            num_steps = abs(num_steps)

        x = np.linspace(0, num_days, num_steps)
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
            rednoise_time_scale = rng.integers(
                5, 15
            )  # correlation time scale of red noise

        rednoise = np.convolve(
            rng.random(2 * num_steps),
            windows.gaussian(int(4 * rednoise_time_scale), rednoise_time_scale),
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
        step = (
            step_amp
            * (x > random_noise_step_loc)
            * (x < (random_noise_step_loc + step_width))
        )

        # Trend parameters
        if slope is None:
            slope = 0.001 - 0.002 * rng.random()  # slope of trend

        trend = slope * (x - num_steps / 2)

        # Combine
        y = lightcurve + whitenoise + rednoise + step + trend

        return Nahiku(x, y, **kwargs)

    @staticmethod
    def from_synthetic_parameterized_gp(
        num_days=100,
        num_steps=1000,
        seed=48,
        add_high_residuals=False,
        device="cpu",
        mean_constant=None,
        outputscale=None,
        periodic_lengthscale=None,
        period=None,
        rbf_lengthscale=None,
        noise_std=None,
        num_high_residuals=None,
        mean_high_residuals=None,
        var_high_residuals=None,
        **kwargs,
    ):
        """
        Sample a function from a Gaussian Process with a scaled quasi-periodic kernel, constant mean, and Gaussian likelihood,
        with options to specify various parameters for the kernel, mean function, likelihood noise, and number high residuals to add in.

        :param num_days (float): total duration of the light curve in days (default: 100)
        :param num_steps (int): number of time steps in the light curve (default: 1000)
        :param seed (int): random seed for reproducibility (default: 48)
        :param add_high_residuals (bool): whether to add high residuals to the sampled function to create more challenging anomalies (default: False)
        :param device (str): device to use for GP sampling, either "cpu" or "cuda" (default: "cpu")
        :param mean_constant (float or None): constant value for the mean function.
                If not provided, randomly chosen between -1 and 1 (Optional)
        :param outputscale (float or None): output scale for the scaled quasi-periodic kernel.
                If not provided, randomly chosen between 0.1 and 10 (Optional)
        :param periodic_lengthscale (float or None): length scale for the periodic component of the quasi-periodic kernel.
                If not provided, randomly chosen between 0.5 and num_days/4 (Optional)
        :param period (float or None): period for the periodic component of the quasi-periodic kernel.
                If not provided, randomly chosen between 0.5 and num_days (Optional)
        :param rbf_lengthscale (float or None): length scale for the RBF component of the quasi-periodic kernel.
                If not provided, randomly chosen between 0.5 and num_days/2 (Optional)
        :param noise_std (float or None): standard deviation of the Gaussian noise to add to the sampled function.
                If not provided, randomly chosen between 0.1 and 1 (Optional)
        :param num_high_residuals (int or None): number of high residuals to add to the sampled function if add_high_residuals is True.
                If not provided, randomly chosen between 5 and 25 (Optional)
        :param mean_high_residuals (float or None): mean of the Gaussian distribution to sample the high residuals from if add_high_residuals is True.
                If not provided, randomly chosen between -1 and 1 (Optional)
        :param var_high_residuals (float or None): variance of the Gaussian distribution to sample the high residuals from if add_high_residuals is True.
                If not provided, randomly chosen between 0.1 and 10 (Optional)
        :param kwargs: Additional keyword arguments to pass to Nahiku.__init__
        """

        rng = np.random.default_rng(seed=seed)

        # Sample missing parameters if not given
        if mean_constant is None:
            mean_constant = rng.uniform(-1, 1)

        if outputscale is None:
            outputscale = rng.uniform(0.1, 10)

        if periodic_lengthscale is None:
            periodic_lengthscale = rng.uniform(0.5, num_days / 4)

        if period is None:
            period = rng.uniform(0.5, num_days)

        if rbf_lengthscale is None:
            rbf_lengthscale = rng.uniform(0.5, num_days / 2)

        if noise_std is None:
            noise_std = rng.uniform(0.1, 1)

        if num_high_residuals is None:
            num_high_residuals = rng.integers(5, 25)

        if mean_high_residuals is None:
            mean_high_residuals = rng.uniform(-1, 1)

        if var_high_residuals is None:
            var_high_residuals = rng.uniform(0.1, 10)

        # Define timesteps, y as Gaussian noise, and noise
        x_sample = torch.linspace(0, num_days, num_steps).to(device)

        # Initialize a scaled quasi-periodic kernel with the specified parameters
        kernel = ScaleKernel(QuasiPeriodicKernel())
        kernel.outputscale = outputscale
        kernel.base_kernel.periodic_kernel.period_length = period
        kernel.base_kernel.periodic_kernel.lengthscale = periodic_lengthscale
        kernel.base_kernel.rbf_kernel.lengthscale = rbf_lengthscale

        # Initialize a constant mean function with the specified mean constant
        mean = ConstantMean()
        mean.constant = mean_constant

        # Sample from the MultivariateNormal defined by the parameterized kernel and mean
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean_x = mean(x_sample).cpu()
            covar_x = kernel(x_sample).cpu()
            mvn = MultivariateNormal(mean_x, covar_x)

        # Sample from the MultivariateNormal
        sample = mvn.sample()

        # Add uncorrelated gaussian noise with noise_std
        noisy_sample = sample.cpu().numpy()
        noisy_sample += np.random.normal(0, noise_std, size=noisy_sample.shape)

        # Convert to numpy for further processing
        x = x_sample.detach().cpu().numpy()
        sample = noisy_sample

        if add_high_residuals:
            # Sample num_residuals from a normal distribution with mean mean_residuals and std of sqrt(var_residuals)
            num_high_residuals = int(num_high_residuals)
            residuals = np.random.normal(
                loc=mean_high_residuals,
                scale=np.sqrt(var_high_residuals),
                size=num_high_residuals,
            )

            # Randomly flip signs with 50% probability
            signs = np.random.choice([1, -1], size=num_high_residuals)
            residuals *= signs

            high_residual_indices = np.random.choice(
                len(x), num_high_residuals, replace=False
            )

            # Add the high residuals to the sample at the randomly chosen indices
            for idx_res, idx_sample in enumerate(high_residual_indices):
                sample[idx_sample] += residuals[idx_res]

        return Nahiku(x, sample, **kwargs)

    def greedy_search(self, **kwargs):
        """
        Initialize and run a GreedySearch.
        Accepts all arguments for GreedySearch.__init__ and GreedySearch.search_for_anomaly.
        """

        # Pop out keys that are needed for GreedySearch.__init__, and pass the rest to search_for_anomaly
        init_keys = [
            "device",
            "which_grow_metric",
            "y_err",
            "num_sigma_threshold",
            "expansion_param",
            "len_deviant",
        ]

        # Create a dict of init args, falling back to defaults if not in kwargs
        init_args = {k: kwargs.pop(k) for k in init_keys if k in kwargs}

        search = GreedySearch(self.time, self.flux, self.dominant_period, **init_args)

        # Pass remaining kwargs to search_for_anomaly, which will handle defaults for those
        search.search_for_anomaly(**kwargs)
        print(
            f"The greedy search took {search.runtime:.2f} seconds to run, and found {search.num_detected_anomalies} anomalous intervals."
        )

        # Add flagged anomalous indices to self.anomalies['identified']
        flagged_indices = np.nonzero(search.flagged_anomalous)[0].tolist()

        if flagged_indices not in self.anomalies["identified"]:
            self.anomalies["identified"].extend(flagged_indices)
            self.anomalies[
                "identified"
            ].sort()  # Sort the list of injected anomaly locations for easier visualization and analysis later

        return search

    def exhaustive_search(self, **kwargs):
        """
        Initialize and run a ExhaustiveSearch.
        Accepts all arguments for ExhaustiveSearch.__init__ and ExhaustiveSearch.search_for_anomaly.
        """

        # Pop out keys that are needed for ExhaustiveSearch.__init__, and pass the rest to search_for_anomaly
        init_keys = [
            "device",
            "min_anomaly_len",
            "max_anomaly_len",
            "window_slide_step",
            "window_size_step",
            "assume_independent",
            "which_test_metric",
        ]

        # Create a dict of init args, falling back to defaults if not in kwargs
        init_args = {k: kwargs.pop(k) for k in init_keys if k in kwargs}

        search = ExhaustiveSearch(
            self.time, self.flux, self.dominant_period, **init_args
        )

        # Pass remaining kwargs to search_for_anomaly, which will handle defaults for those
        search.search_for_anomaly(**kwargs)
        print(
            f"The exhaustive search took {search.runtime:.2f} seconds to run, and found {search.num_detected_anomalies} anomalous intervals."
        )

        # Add flagged anomalous indices to self.anomalies['identified']
        flagged_indices = np.nonzero(search.flagged_anomalous)[0].tolist()

        if flagged_indices not in self.anomalies["identified"]:
            self.anomalies["identified"].extend(flagged_indices)
            self.anomalies[
                "identified"
            ].sort()  # Sort the list of injected anomaly locations for easier visualization and analysis later

        return search

    def plot(self, show_identified_points=True):
        """
        Plot the light curve with shaded regions for injected/true anomalies
        and optional red x's for identified anomalies.

        :param show_identified_points (bool): whether to plot the identified anomalous points as red x's (default: True)
        """
        plt.figure(figsize=(10, 5))

        # Base Light Curve
        plt.scatter(self.time, self.flux, c="k", s=3, alpha=0.5, label="Light Curve")

        # Shaded Regions (Injected & True)
        regions = [
            ("injected", "gold", "Injected Anomaly", 0.6),
            ("true", "blue", "True Anomaly", 0.6),
        ]

        for key, color, label, alpha in regions:
            events = self.get_events(self.anomalies.get(key, []))
            for i, (start_idx, end_idx) in enumerate(events):
                plt.axvspan(
                    self.time[start_idx],
                    self.time[end_idx],
                    color=color,
                    alpha=alpha,
                    # Only add the label to the first event for the legend
                    label=label if i == 0 else "",
                )

        # Identified Points
        if show_identified_points and self.anomalies["identified"]:
            idx = self.anomalies["identified"]
            plt.scatter(
                self.time[idx],
                self.flux[idx],
                c="red",
                s=10,
                marker="x",
                alpha=0.8,
                label="Identified Anomaly",
            )

        # Formatting
        plt.xlim(self.time[0], self.time[-1])
        # Auto-scale Y with some padding
        y_padding = np.ptp(self.flux) * 0.1
        plt.ylim(np.min(self.flux) - y_padding, np.max(self.flux) + y_padding)

        plt.xlabel("Time [units of time array]")
        plt.ylabel("Normalized Flux")
        plt.title(
            f"Light Curve | Dominant Period: {self.dominant_period:.2f} [units of time array]"
        )

        # Place legend outside or adjust to avoid covering data
        plt.legend(loc="upper right", frameon=True, fontsize="small")
        plt.tight_layout()
        plt.show()

    def standardize(self):
        """
        Function to standardize the flux of the light curve by subtracting the mean and dividing by the standard deviation.
        This is important for the periodogram calculation and GP modeling, as it ensures that the data is on a consistent scale
        and that the periodogram is not dominated by the mean flux level.
        """
        self.flux = (self.flux - np.mean(self.flux)) / np.std(self.flux)

        return

    def prewhiten(self, plot=False, **kwargs):
        """
        Prewhiten a light curve using the balmung.prewhiten function, with options to specify various parameters for the removal of frequencies.
        Code for balmung.prewhiten can be found here: https://github.com/danhey/balmung/blob/master/balmung/balmung.py

        :param plot (bool): whether to plot the light curve before and after prewhitening (default: True)
        :param kwargs: Additional keyword arguments to pass to balmung.prewhiten and Nahiku.__init__
        """
        # Create a dict of init args, falling back to defaults if not in kwargs
        init_keys = ["anomalies", "prominence", "plot_dominant_period"]
        init_args = {k: kwargs.pop(k) for k in init_keys if k in kwargs}

        bm = Balmung(self.time, self.flux)

        if plot:
            print("Light curve before prewhitening:")
            self.plot()

        bm.prewhiten(**kwargs)

        # Update inplace
        self.flux = bm.residual
        self.dominant_period = self.get_dominant_period(
            prominence=init_args.get("prominence", 50),
            plot=init_args.get("plot_dominant_period", False),
        )
        self.standardize()

        if plot:
            print("Light curve after prewhitening:")
            self.plot()

        return

    @staticmethod
    def freq_idx_to_period_days(freqs_idx, times):
        """
        Function to convert frequency indices from a periodogram to periods in days, using the time points of the original data to calculate the scaling factor for the conversion.

        :param freqs_idx (1D array-like): Array of frequency indices to convert to periods in days.
        :param times (1D array-like): Array of time points corresponding to the original data, used to calculate the scaling factor for converting frequency indices to periods in days.
        """
        idx_day_scale_factor = (times[-1] - times[0]) / len(times)
        periods = 1 / freqs_idx
        periods_days = periods * idx_day_scale_factor

        return periods_days

    def get_dominant_period(self, prominence=50, plot=False):
        """
        Function to calculate the dominant period of a light curve using the periodogram and peak detection.
        It also includes an option to plot the periodogram and the light curve with the dominant period sinusoid.

        :param prominence (int): minimum prominence of peaks to consider in the periodogram (default: 50)
        :param plot (bool): whether to plot the periodogram and light curve (default: False)
        """

        # Check if data is standardized
        if np.std(self.flux) != 1:
            warnings.warn(
                "Data is not standardized, and will be standardized for estimating the dominant period."
            )
            self.standardize()

        # Get peaks in power spectrum
        freqs, power = periodogram(self.flux)
        peaks, _ = find_peaks(power, prominence=prominence)

        if len(peaks) == 0:
            print(
                f"No peaks found in power spectrum, using shoulder instead. Maximum dominant period is {self.time[-1]:.2f} days"
            )
            smooth_power = gaussian_filter1d(power, 2)
            slope = np.gradient(smooth_power, freqs)
            shoulder_indices = np.where(slope < 0)[0]
            if len(shoulder_indices) > 0:
                shoulder_idx = shoulder_indices[0]
                dominant_period = min(
                    self.freq_idx_to_period_days(freqs[shoulder_idx], self.time),
                    self.time[-1],
                )
            else:
                dominant_period = self.time[-1] - self.time[0]

        else:
            # Filter to most prominent peak
            prominences, left_bases, right_bases = peak_prominences(
                power, peaks, wlen=5
            )

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
            dominant_period = self.freq_idx_to_period_days(
                freqs[peaks[max_peak]], self.time
            )

        # Plot periodogram
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            axs[0].plot(
                self.freq_idx_to_period_days(freqs, self.time),
                power,
                label="Periodogram",
            )
            if len(peaks) > 0:
                axs[0].plot(
                    self.freq_idx_to_period_days(freqs[peaks], self.time),
                    power[peaks],
                    "x",
                    label="Peaks",
                )
                axs[0].plot(
                    self.freq_idx_to_period_days(freqs[left_bases], self.time),
                    power[left_bases],
                    "o",
                    c="gray",
                    label="Right bases",
                )  # Reversed bc period = 1/frequency
                axs[0].plot(
                    self.freq_idx_to_period_days(freqs[right_bases], self.time),
                    power[right_bases],
                    "o",
                    c="black",
                    label="Left bases",
                )  # Reversed bc period = 1/frequency
            else:
                axs[0].plot(
                    self.freq_idx_to_period_days(freqs[shoulder_idx:], self.time),
                    power[shoulder_idx:],
                    "x",
                    label="Shoulder",
                )
            axs[0].legend()
            axs[0].set_xscale("log")
            axs[0].set_xlabel("Period [units of time array]")
            axs[0].set_ylabel("Power")
            axs[0].set_title(
                f"Periodogram with max peak at {dominant_period:.2f} [units of time array]"
            )

            # Plot lightcurve with dominant period sinusoid
            axs[1].scatter(self.time, self.flux, s=2, label="Lightcurve")
            axs[1].plot(
                self.time,
                np.sin(2 * np.pi * self.time / dominant_period) + 4,
                c="darkorange",
                label=f"Dominant period: {dominant_period:.2f} [units of time array]",
            )
            axs[1].set_xlabel("Time [units of time array]")
            axs[1].set_ylabel("Flux")
            axs[1].legend()

            plt.tight_layout()
            plt.show()

        return dominant_period

    def inject_anomaly(
        self,
        num_anomalies,
        absolute_width=None,
        absolute_depth=None,
        idxs=None,
        seed=48,
        shapes=["gaussian", "saw", "exocomet"],
        period_scale=None,
        snr=None,
        alpha=1,
    ):
        """
        Inject an anomaly into the light curve, with options to specify the number of anomalies, their shapes, widths, depths, and locations.

        :param num_anomalies (int): number of anomalies to inject
        :param absolute_width (float or None): absolute width of the anomaly. If specified, period_scale is ignored (default: None)
        :param absolute_depth (float or None): absolute depth of the anomaly. If specified, snr is ignored (default: None)
        :param idxs (list of float or None): list of indices to inject anomalies at. If None, indices are randomly chosen (default: None)
        :param seed (int): random seed for reproducibility (default: 48)
        :param shapes (list of str): list of shapes to choose from for the anomalies.
                Options are "gaussian" for gaussian-shaped anomalies, "saw" for sawtooth-shaped anomalies, and "exocomet" for exocomet-shaped anomalies.
                Default is ["gaussian", "saw", "exocomet"].
        :param period_scale (float or None): ratio of the dominant period to use as the width of the anomaly. If None, randomly chosen between 0.1 and 5 (default: None)
        :param snr (float or None): signal to noise ratio of the anomaly. If None, randomly chosen between 0.5 and 10 (default: None)
        :param alpha (float): shape parameter for the exocomet profile, which controls the asymmetry of the anomaly.
                Higher values of alpha result in a more asymmetric profile with a steeper ingress and a shallower egress (default: 1)
        """

        # Initialize
        rng = np.random.default_rng(seed=seed)
        num_steps = len(self.time)
        time_steps = np.arange(num_steps)
        anomaly = np.zeros(num_steps)
        anomaly_idxs = []

        # If absolute_depth is given, use it as the depth of the anomaly
        if absolute_depth is not None:
            if absolute_depth < 0:
                warnings.warn(
                    "Absolute depth must be positive. Defaulting to its absolute value."
                )
                absolute_depth = abs(absolute_depth)

            anomaly_amp = -1 * absolute_depth

        else:
            # Create anomaly with snr if not given
            if snr is None:
                snr = rng.uniform(0.5, 10)  # depth of anomaly
                print(
                    f"Anomaly absolute depth and snr were not specified. Using snr = {snr}"
                )

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
                warnings.warn(
                    "Absolute width must be positive. Defaulting to its absolute value."
                )
                absolute_width = abs(absolute_width)

            anomaly_width = absolute_width

        else:
            # Create anomaly period_scale if not given
            if period_scale is None:
                period_scale = rng.uniform(0.1, 5)  # period scaling of anomaly
                print(
                    f"Anomaly absolute width and period_scale were not specified. Using period_scale = {period_scale}"
                )

            if period_scale < 0:
                warnings.warn(
                    "Period scale must be positive. Defaulting to its absolute value."
                )
                period_scale = abs(period_scale)

            # Create anomaly_width from period of peak in power spectrum
            # minimum value of 1. Note this is the std dev. of the anomaly (assuming Gaussian)
            anomaly_period = period_scale * self.dominant_period
            anomaly_width = max(anomaly_period / (2 * np.sqrt(2 * np.log(2))), 1)

        # Perform some checks of idxs list if given
        if idxs is not None:
            # Check idxs is a list of floats
            if not isinstance(idxs, list) or not all(
                isinstance(idx, (int, float)) for idx in idxs
            ):
                warnings.warn(
                    "Idxs must be a list of floats. Defaulting to random indices."
                )
                idxs = None

            # Check that idxs are within the range of the time array
            if not all((idx >= 0 and idx < len(self.time)) for idx in idxs):
                warnings.warn(
                    "All idxs must be within the range of the time array. Defaulting to random indices."
                )
                idxs = None

        if idxs is not None:
            # Check number of idxs matches num_anomalies
            if len(idxs) != num_anomalies:
                warnings.warn(
                    "Length of idxs does not match num_anomalies. Defaulting to only using the first num_anomalies values in idxs."
                )
                idxs = idxs[:num_anomalies]

        # Check that shapes is a list of strings and that all shapes are valid
        if (
            not isinstance(shapes, list)
            or not all(isinstance(shape, str) for shape in shapes)
            or not all(shape in ["gaussian", "saw", "exocomet"] for shape in shapes)
        ):
            warnings.warn(
                "Shapes must be a list of strings containing only 'gaussian', 'saw', and/or 'exocomet'. Defaulting to ['gaussian', 'saw', 'exocomet']"
            )
            shapes = ["gaussian", "saw", "exocomet"]

        # Inject anomalies of anomaly_width and anomaly_amp at random locations (as many as in idxs)
        for i in range(num_anomalies):

            # If idxs is given, use the specified index. Otherwise, choose a random index for the anomaly
            if idxs is not None:
                anomaly_idx = int(idxs[i])
            else:
                anomaly_idx = int(num_steps * rng.random())

            anomaly_idxs.append(anomaly_idx)
            shape = rng.choice(shapes)

            if shape == "gaussian":
                # Gaussian-shape anomaly at x0
                anomaly += anomaly_amp * np.exp(
                    -0.5 * ((time_steps - anomaly_idx) / anomaly_width) ** 2
                )
                anomaly_fwhm = (
                    2.355 * anomaly_width
                )  # True for gaussian-shaped anomalies

            elif shape == "saw":
                # Create anomaly that has a quick dip to anomaly_amp, then a slow rise back to 0 based on anomaly_width
                anomaly += anomaly_amp * (
                    1 - np.exp(-np.abs(time_steps - anomaly_idx) / anomaly_width)
                )
                anomaly_fwhm = (
                    2 * anomaly_width
                )  # Approximation for saw-shaped anomalies

            elif shape == "exocomet":
                # -A * exp(-t/tau) * (t/tau)^alpha
                # T = absolute time (in days) in the light curve
                # T0 = injection time (in days) in the light curve
                # t = T - T0 if T >= T0 else 0
                # A is an amplitude parameter, tau a width parameter, and alpha the shape parameter
                t = self.time - self.time[anomaly_idx]
                t = np.where(t >= 0, t, 0)

                anomaly += (
                    anomaly_amp
                    * np.exp((-1 * t) / anomaly_width)
                    * (t / anomaly_width) ** alpha
                )

                anomaly_fwhm = (
                    2.45 * anomaly_width
                )  # The FWHM is 2.45 tau for exocomet-shaped anomalies
                anomaly_amp = (
                    0.37 * anomaly_amp
                )  # Because the minimum of this function is at A/e, or 0.37A

            print(
                f"Injected {shape}-shaped anomaly with amplitude {anomaly_amp:.2f}, width {anomaly_width:.2f}, and FWHM {anomaly_fwhm:.2f} at index {anomaly_idx} (time {self.time[anomaly_idx]:.2f} [units of time array])"
            )

            # Add anomaly_idxs to self.anomalies['injected'] and add anomaly to flux
            if anomaly_idx not in self.anomalies["injected"]:
                self.anomalies["injected"].append(anomaly_idx)
                self.anomalies[
                    "injected"
                ].sort()  # Sort the list of injected anomaly locations for easier visualization and analysis later

        self.flux += anomaly

        return anomaly_idxs, anomaly, anomaly_amp, anomaly_fwhm

    @staticmethod
    def get_events(indices):
        """
        Helper to turn a list of indices into a list of (start, end) tuples.

        :param indices (list of int): list of indices to group into events
        """
        if not indices:
            return []
        indices = sorted(list(set(indices)))
        events = []
        start = indices[0]

        # Group consecutive indices into events.
        # If the next index is more than 1 away from the current index, we consider it a new event
        for i in range(1, len(indices)):
            if indices[i] > indices[i - 1] + 1:
                events.append((start, indices[i - 1]))
                start = indices[i]

        events.append((start, indices[-1]))
        return events

    def check_identified_anomalies(self, buffer=5):
        """
        Check the identified anomalies against the true and injected anomalies, and print out the results.

        :param buffer (int): number of indices on either side of the true and injected anomaly indices to consider as a match for an identified anomaly (default: 5)
        """
        # Group indices into events
        true_events = self.get_events(self.anomalies["true"])
        injected_events = self.get_events(self.anomalies["injected"])
        identified_events = self.get_events(self.anomalies["identified"])

        # Combine true and injected for total ground truth
        all_ground_truth = true_events + injected_events

        # Check for matches
        detected_count = 0
        for gt_start, gt_end in all_ground_truth:
            # Check if any identified event overlaps with this ground truth event (including buffer)
            for id_start, id_end in identified_events:
                # Overlap logic: (StartA <= EndB + buffer) and (EndA >= StartB - buffer)
                if (id_start <= gt_end + buffer) and (id_end >= gt_start - buffer):
                    detected_count += 1
                    break  # Found a match for this GT event, move to next

        # False Positives: Identified events that hit no ground truth
        false_positives_count = 0
        for id_start, id_end in identified_events:
            hit = False
            for gt_start, gt_end in all_ground_truth:
                if (id_start <= gt_end + buffer) and (id_end >= gt_start - buffer):
                    hit = True
                    break

            if not hit:
                false_positives_count += 1

        # Calculate metrics
        total_gt_events = len(all_ground_truth)
        tpr = detected_count / total_gt_events if total_gt_events > 0 else 0
        precision = (
            detected_count / len(identified_events) if len(identified_events) > 0 else 0
        )

        print(f"Events Detected: {detected_count} / {total_gt_events}")
        print(f"Event-wise TPR: {tpr:.2f}")
        print(f"False Positive Events: {false_positives_count}")
        print(f"Event-wise Precision: {precision:.2f}")

        return {
            "tpr": tpr,
            "precision": precision,
            "detected_count": detected_count,
            "false_positives": false_positives_count,
        }
