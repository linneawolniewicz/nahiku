

def freq_idx_to_period_days(freqs_idx, times):
    """
    Docstring for freq_idx_to_period_days
    
    :param freqs_idx (1D array-like): Array of frequency indices to convert to periods in days.
    :param times (1D array-like): Array of time points corresponding to the original data, used to calculate the scaling factor for converting frequency indices to periods in days.
    """
    idx_day_scale_factor = (times[-1] - times[0]) / len(times)
    periods = 1 / freqs_idx
    periods_days = periods * idx_day_scale_factor

    return periods_days