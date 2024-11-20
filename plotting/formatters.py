import numpy as np
from matplotlib.ticker import FuncFormatter


def seconds_to_hours_tick_formatter(seconds, pos):
    return f"{seconds / 3600:.1f}"


def seconds_to_fraction_of_tide_tick_formatter(seconds, pos):
    return f"{seconds / 12.42*60*60:.1f}"


def meters_to_kilometers_tick_formatter(meters, pos):
    return f"{meters / 1000:.1f}"


hours_formatter = FuncFormatter(seconds_to_hours_tick_formatter)
kilometers_formatter = FuncFormatter(meters_to_kilometers_tick_formatter)
tide_formatter = FuncFormatter(seconds_to_fraction_of_tide_tick_formatter)
