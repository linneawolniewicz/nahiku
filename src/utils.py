
def freq_idx_to_period_days(freqs_idx, times):
    idx_day_scale_factor = (times[-1] - times[0]) / len(times)
    periods = 1 / freqs_idx
    periods_days = periods * idx_day_scale_factor

    return periods_days