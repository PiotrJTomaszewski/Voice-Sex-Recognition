import sys
import scipy.io.wavfile as wav
import scipy.signal as sig
import warnings
from numpy.fft import fft
import numpy as np

# DEBUG_MODE = False

# Voice frequency (according to wiki)
# Male 85 to 180Hz
# Female 165 to 255Hz
# Also from 40 to 600Hz for all human beings
male_voice_averaged = 132.5
female_voice_averaged = 210
min_voice_freq = 40
max_voice_freq = 600


def downsample_multiply_signal(signal):
    signal_fd = abs(fft(signal))
    signal_fd_org = signal_fd.copy()
    for i in range(2, 5):
        decimated_signal_fd = sig.decimate(x=signal_fd_org, q=i)
        signal_fd[:len(decimated_signal_fd)] *= decimated_signal_fd
    return np.argmax(signal_fd[1:]) + 1


def analyze(signal, window, window_length):
    window_overlap = window_length // 2
    start_point = 0
    argmaxes = []
    while start_point <= len(signal) - window_length:
        signal_in_window = signal[start_point:start_point + window_length] * window
        argmaxes.append(downsample_multiply_signal(signal_in_window))
        start_point += window_overlap
    return argmaxes


def generate_weights(number_of_values):
    weights = list(range(1, (number_of_values // 2) + 1))
    if number_of_values % 2 != 0:
        weights.append(weights[-1])
    weights += reversed(weights)
    return weights


def recognize(file_path):
    # Read the file
    sample_rate, signal = wav.read(file_path)
    # Take only one channel
    if type(signal[0]) is not np.int16:  # Check if signal is not already mono
        signal = [x[0] for x in signal]

    window_length = 8192
    window = np.kaiser(M=window_length, beta=14)
    analyzed_signal = analyze(signal, window, window_length)
    frequency_scale = np.linspace(0, sample_rate, window_length)
    fundamental_freq_candidates = [frequency_scale[x] for x in analyzed_signal]
    # Filter signals
    filtered_fund_freq_candidates = [x for x in fundamental_freq_candidates if min_voice_freq < x < max_voice_freq]
    if len(filtered_fund_freq_candidates) == 0:
        # if DEBUG_MODE:
        #     print('No voice found')
        return 'K'
    # Calculate weighted average
    weights = generate_weights(len(filtered_fund_freq_candidates))
    fundamental_voice_frequency = sum([a * b for a, b in zip(filtered_fund_freq_candidates, weights)]) / sum(weights)
    # Calculate distance from the average frequency for both sexes
    distance_from_male = abs(fundamental_voice_frequency - male_voice_averaged)
    distance_from_female = abs(fundamental_voice_frequency - female_voice_averaged)
    if distance_from_female < distance_from_male:
        return 'K'
    else:
        return 'M'


def main():
    # Disable warnings because they mess with the result checking script
    # if DEBUG_MODE:
    #     file_path = sys.argv[1]
    #     result = recognize(file_path)
    #     print(result)
    # else:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            try:
                result = recognize(file_path)
                print(result)
            except BaseException:
                print('K')


if __name__ == '__main__':
    # DEBUG_MODE = True
    main()
