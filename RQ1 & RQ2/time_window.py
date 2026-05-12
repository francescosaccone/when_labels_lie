import pandas as pd
from datetime import timedelta
import numpy as np
from collections import Counter


def convert_to_seconds(time_str):
    """
    Convert a time window string into seconds.

    The function supports strings with days, such as
    "2 days, 03:04:05", and strings without days, such as
    "03:04:05".

    Parameters
    ----------
    time_str : str
        Time window string to convert.

    Returns
    -------
    float
        Total number of seconds represented by the input string.
    """
    if "day" in time_str:
        days, _, time_part = time_str.partition(", ")
        day_count = int(days.split()[0])
        h, m, s = map(int, time_part.split(":"))
        total_seconds = timedelta(
            days=day_count,
            hours=h,
            minutes=m,
            seconds=s
        ).total_seconds()
    else:
        h, m, s = map(int, time_str.split(":"))
        total_seconds = timedelta(
            hours=h,
            minutes=m,
            seconds=s
        ).total_seconds()

    return total_seconds


def convert_seconds_to_time(seconds):
    """
    Convert seconds into a readable time window string.

    Parameters
    ----------
    seconds : float
        Number of seconds to convert.

    Returns
    -------
    str
        Readable time string in either "X days, HH:MM:SS" or "HH:MM:SS" format.
    """
    time_delta = timedelta(seconds=seconds)
    days = time_delta.days
    hours, remainder = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days > 0:
        return f"{days} days, {hours:02}:{minutes:02}:{seconds:02}"

    return f"{hours:02}:{minutes:02}:{seconds:02}"


df = pd.read_csv("miss_class_filtered.csv")

time_strings = df["window"]

time_in_seconds = [convert_to_seconds(time_str) for time_str in time_strings]

mean_seconds = np.mean(time_in_seconds)
median_seconds = np.median(time_in_seconds)
mode_seconds = Counter(time_in_seconds).most_common(1)[0][0]

mean_time = convert_seconds_to_time(mean_seconds)
median_time = convert_seconds_to_time(median_seconds)
mode_time = convert_seconds_to_time(mode_seconds)

print(f"Mean: {mean_time}")
print(f"Median: {median_time}")
print(f"Mode: {mode_time}")