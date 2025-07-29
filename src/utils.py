import pyfar as pf
import numpy as np
import soundfile as sf
import scipy
from numpy.typing import NDArray, ArrayLike
from typing import Any, Optional, Tuple, Union


def cart2sph(x: Union[NDArray, float],
             y: Union[NDArray, float],
             z: Union[NDArray, float],
             axis: int = -1,
             degrees: bool = True) -> NDArray:
    """
    Convert cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x : Union[NDArray, float]
        X coordinate(s).
    y : Union[NDArray, float]
        Y coordinate(s).
    z : Union[NDArray, float]
        Z coordinate(s).
    axis : int, optional
        Axis along which to stack the result (default is -1).
    degrees : bool, optional
        If True, return angles in degrees (default is True).

    Returns
    -------
    NDArray
        Spherical coordinates (azimuth, elevation, r).
    """
    x, y, z = np.broadcast_arrays(x, y, z)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)

    if degrees:
        azimuth = np.degrees(azimuth)
        elevation = np.degrees(elevation)

    return np.stack((azimuth, elevation, r), axis=axis)


def sph2cart(azimuth: Union[NDArray, float],
             elevation: Union[NDArray, float],
             r: Union[NDArray, float],
             axis: int = -1,
             degrees: bool = True) -> NDArray:
    """
    Convert spherical coordinates to cartesian coordinates.

    Parameters
    ----------
    azimuth : Union[NDArray, float]
        Azimuth angle(s) in degrees or radians.
    elevation : Union[NDArray, float]
        Elevation angle(s) in degrees or radians.
    r : Union[NDArray, float]
        Radius value(s).
    axis : int, optional
        Axis along which to stack the result (default is -1).
    degrees : bool, optional
        If True, input angles are in degrees (default is True).

    Returns
    -------
    NDArray
        Cartesian coordinates (x, y, z).
    """
    azimuth, elevation, r = np.broadcast_arrays(azimuth, elevation, r)
    if degrees:
        azimuth = np.radians(azimuth)
        elevation = np.radians(elevation)
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.stack((x, y, z), axis=axis)


def unpack_coordinates(
        coord_matrix: NDArray,
        axis: int = -1) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Unpack a coordinate matrix into separate arrays for each dimension.

    Parameters
    ----------
    coord_matrix : NDArray
        Coordinate matrix of shape (..., 3).
    axis : int, optional
        Axis along which to unpack (default is -1).

    Returns
    -------
    tuple of ArrayLike
        Tuple of arrays (dim1, dim2, dim3) corresponding to the three coordinate dimensions.
    """
    if axis != -1:
        coord_matrix = coord_matrix.T

    dim1 = coord_matrix[:, 0]
    dim2 = coord_matrix[:, 1]
    dim3 = coord_matrix[:, 2]

    return dim1, dim2, dim3


def rir_from_sweep(meas_sweep: Union[ArrayLike, NDArray],
                   dry_sweep: ArrayLike,
                   fs: float,
                   start_time_ms: float = 1000,
                   end_time_ms: float = 10000,
                   extra_padding_ms: float = 1000) -> NDArray:
    """
    Deconvolution to get RIRs from measured and dry sweep signals
    Args:
        meas_sweep (ArrayLike/NDArray): measured sine sweep signals in a room, of shape 
                                        num_time_samples x num_channels, or num_time_samples,
        dry_sweep (ArrayLike): dry sine sweep signal, of shape num_time_samples,
        fs (float): sampling rate of sweep signals
        start_time_ms (float): trim silence before dry sweep
        end_time_ms (float): length of sweep in ms
        extra_padding (float): length of the measured signal should be 
                               longer than the sweep length by this much
    Returns:
        NDArray: Deconvolved RIRs of shape (num_time_samples x num_channels) or num_time_samples,
    """
    dry_sweep = dry_sweep[:, np.newaxis]
    if meas_sweep.ndim > 1:
        num_channels = meas_sweep.shape[1]
        dry_sweep = np.repeat(dry_sweep, num_channels, axis=-1)
    else:
        meas_sweep = meas_sweep[:, np.newaxis]

    #### WRITE YOUR CODE HERE ####

    # truncate sweep signals from start_time_ms to start_time_ms + end_time_ms
    start_time_samps = ms_to_samps(start_time_ms, fs)
    end_time_samps = start_time_samps + ms_to_samps(end_time_ms, fs)
    extra_padding_samps = ms_to_samps(extra_padding_ms, fs)

    # add some extra padding at the end to the measured sweep signal
    meas_sweep_trunc = meas_sweep[start_time_samps:end_time_samps +
                                  extra_padding_samps, :]
    dry_sweep_trunc = dry_sweep[start_time_samps:end_time_samps, :]

    #### WRITE YOUR CODE HERE ####

    # time flip the dry sweep signal
    dry_sweep_flipped = np.flip(dry_sweep_trunc)
    # convolve the measured sweep signals and the time flipped dry sweep signal
    rirs = scipy.signal.fftconvolve(meas_sweep_trunc, dry_sweep_flipped, axes = 0, mode = 'same')
    # return the RIRs
    return rirs


def audioread(rir_path: str, to_mono: bool = True) -> tuple[np.ndarray, int]:
    """
    Read an audio file and optionally convert it to mono.

    Parameters
    ----------
    rir_path : str
        Path to the audio file.
    to_mono : bool, optional
        If True, convert stereo to mono by averaging channels (default: True).

    Returns
    -------
    tuple[np.ndarray, int]
        Tuple containing the audio data and sampling rate.
    """
    rir, fs = sf.read(rir_path)
    if rir.ndim > 1 and to_mono:
        rir = rir.mean(axis=1)
    return rir, fs


def find_onset(rir: NDArray) -> int:
    """
    Find the onset in a room impulse response (RIR) by extracting a local energy envelope and locating its maximum.

    Parameters
    ----------
    rir : np.ndarray
        Room impulse response of shape num_time_samples x num_channels

    Returns
    -------
    int
        Index of the detected onsets in the RIRs.
    """
    win_len = 64
    overlap = 0.75
    win = np.hanning(win_len)[:, np.newaxis]

    if len(rir.shape) == 1:
        rir = np.expand_dims(rir, -1)
    # pad rir
    pad_width = int(win_len * overlap)
    rir = np.pad(rir, ((pad_width, pad_width), (0, 0)))
    hop = 1 - overlap
    hop_len = int(win_len * hop)
    n_wins = int(np.floor(rir.shape[0] / hop_len - 1 / (2 * hop)))

    local_energy = []
    for i in range(1, n_wins - 1):
        start = (i - 1) * hop_len
        end = start + win_len
        segment = rir[start:end, :]
        if segment.shape[0] != win_len:
            continue
        energy_per_channel = np.sum((segment**2) * win,
                                    axis=0)  # shape num_channels,
        local_energy.append(energy_per_channel)

    # convert to 2D array of shape (num_windows, num_channels)
    local_energy = np.stack(local_energy, axis=0)

    # discard trailing points
    n_win_discard = int((overlap / hop) - (1 / (2 * hop)))
    local_energy = local_energy[n_win_discard:, :]
    if len(local_energy) == 0:
        return 0
    onset_idx = np.argmax(local_energy, axis=0)
    return (win_len * hop * (onset_idx - 1)).astype(int)


def filterbank(x: np.ndarray,
               n_fractions: int = 1,
               f_min: int = 63,
               f_max: int = 16000,
               sample_rate: int = 48000,
               compensate_energy: bool = True,
               filter_type: str = 'pyfar') -> np.ndarray:
    """
    Apply a fractional octave filterbank to a signal.

    Parameters
    ----------
    x : np.ndarray
        Input time-domain signal (can be 1D or multi-dimensional).
    n_fractions : int, optional
        Number of fractions per octave (default: 1).
    f_min : int, optional
        Minimum frequency of the filterbank (default: 63).
    f_max : int, optional
        Maximum frequency of the filterbank (default: 16000).
    sample_rate : int, optional
        Sampling rate of the signal (default: 48000).
    compensate_energy : bool, optional
        If True, compensate for energy loss in filtering (default: True).
    filter_type : str, optional
        Type of filterbank to use: 'pyfar' or 'sos' (default: 'pyfar').

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Filtered signal (shape depends on filter_type and input) and center frequencies.
    """
    # Create impulse for filter design
    impulse = np.zeros(x.shape[0])
    impulse[0] = 1.0

    # Get center frequencies for fractional octave bands
    center_freqs = pf.dsp.filter.fractional_octave_frequencies(
        num_fractions=n_fractions,
        frequency_range=(f_min, f_max),
        return_cutoff=False)[0]

    if filter_type == 'sos':
        # Design SOS filters for each band
        order = 5
        sos_filters = []
        for band_idx, center_freq in enumerate(center_freqs):
            if abs(center_freq) < 1e-6:
                # Lowpass for lowest band
                f_cutoff = (1 / np.sqrt(2)) * center_freqs[band_idx + 1]
                sos = scipy.signal.butter(N=order,
                                          Wn=f_cutoff,
                                          fs=sample_rate,
                                          btype='lowpass',
                                          output='sos')
            elif abs(center_freq - sample_rate / 2) < 1e-6:
                # Highpass for highest band
                f_cutoff = np.sqrt(2) * center_freqs[band_idx - 1]
                sos = scipy.signal.butter(N=order,
                                          Wn=f_cutoff,
                                          fs=sample_rate,
                                          btype='highpass',
                                          output='sos')
            else:
                # Bandpass for intermediate bands
                f_cutoff = center_freq * np.array([1 / np.sqrt(2), np.sqrt(2)])
                sos = scipy.signal.butter(N=order,
                                          Wn=f_cutoff,
                                          fs=sample_rate,
                                          btype='bandpass',
                                          output='sos')
            sos_filters.append(sos)

        # Apply each filter to the signal
        filtered = [
            scipy.signal.sosfilt(sos, x, axis=-1) for sos in sos_filters
        ]
        y = np.stack(filtered, axis=-2)

    elif filter_type == 'pyfar':
        # Get frequency responses for each band
        f_bank = pf.dsp.filter.fractional_octave_bands(
            pf.Signal(impulse, sample_rate),
            num_fractions=n_fractions,
            frequency_range=(f_min, f_max),
        ).freq.T  # shape: (filter_length, n_bands)
        f_bank = np.squeeze(f_bank)
        # Prepare output array
        y = np.zeros((f_bank.shape[1], *x.shape))

        # FFT of input signal
        X = np.fft.rfft(x, n=x.shape[0], axis=0)
        for i_band in range(f_bank.shape[1]):
            filt = np.pad(f_bank[:, i_band], (0, X.shape[0] - f_bank.shape[0]))
            if compensate_energy:
                norm = np.sqrt(np.sum(np.abs(filt)**2))
                Y_band = X * filt / norm
            else:
                Y_band = X * filt
            y[i_band, ...] = np.fft.irfft(Y_band, n=x.shape[0], axis=0)

    return y, center_freqs


def discard_last_n_percent(x: np.ndarray,
                           n_percent: float = 5.0) -> np.ndarray:
    """
    Discard the last n_percent of a 1D array.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    n_percent : float, optional
        Percentage of the array to discard from the end (default: 5.0).

    Returns
    -------
    np.ndarray
        Array with the last n_percent removed.
    """
    last_id = int(np.round((1 - n_percent / 100) * x.shape[0]))
    out = x[0:last_id]
    return out


def ms_to_samps(ms: Union[float, ArrayLike], fs: float) -> Union[int, NDArray]:
    """
    Convert milliseconds to samples.

    Parameters
    ----------
    ms : float or ArrayLike
        Time in milliseconds.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    int or np.ndarray
        Time in samples.
    """
    samp = ms * 1e-3 * fs
    if np.isscalar(samp):
        return int(samp)
    else:
        return samp.astype(np.int32)


def db2lin(dB: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Convert decibels to linear scale.

    Parameters
    ----------
    dB : float or NDArray
        Value(s) in decibels.

    Returns
    -------
    float or NDArray
        Value(s) in linear scale.
    """
    return 10**(dB / 20)


def lin2db(linear: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Convert linear scale to decibels.

    Parameters
    ----------
    linear : float or NDArray
        Value(s) in linear scale.

    Returns
    -------
    float or NDArray
        Value(s) in decibels.
    """
    return 20 * np.log10(
        linear + 1e-32)  # Avoid log(0) by adding a small constant