import numpy as np
import matplotlib.pyplot as plt
import Balmung
import lightkurve

from scipy.signal import find_peaks, periodogram, windows, peak_prominences
from scipy.ndimage import gaussian_filter1d

class Dipper:
    def __init__(self, time, flux, anomalies={}):
        self.anomalies = anomalies
        pass

    @staticmethod
    def from_synthetic_parameterized(...):
        return Dipper(artificial_time, artificial_flux, ...)

    @staticmethod
    def from_synthetic_sampled(....):
        return Dipper(artificial_time, artificial_flux, ...)
    
    @staticmethod
    def from_lightkurve(lc_name):
        lc = lightkurve.load_lc(lc_name)
        return Dipper(artificial_time, artificial_flux, ...)

    def greedy_search(self, n_params=100):
        s = Search()

        return s

    def exhaustive_search(self, ...):
        pas = Search()

        return s

    def plot(self):
        pass

    def prewhiten(self, ....):
        prew_t, prew_f = Balmung.()
        return Dipper(prew_t, prew_f)

    def get_dominant_period(self, ...):
        pass

    def inject_anomaly(self, ...):
        pass

    @staticmethod
    def generate_anomaly(...):
        pass

    def check_identified_anomalies(...):
        pass

