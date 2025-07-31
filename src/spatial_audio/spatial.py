from abc import ABC, abstractmethod
from typing import Tuple
from loguru import logger
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.fft import irfft, rfft
from scipy.spatial import ConvexHull
import spaudiopy as spa
from tqdm import tqdm

from utils import cart2sph, sph2cart, unpack_coordinates


def convert_A2B_format_tetramic(rirs_Aformat: NDArray) -> NDArray:
    """
    Convert A format tetramic RIRs to B-format using SN3D normalisation and ACN ordering.

    Parameters
    ----------
    rirs_Aformat : NDArray
        RIRs in A-format, of shape (num_time_samples, num_channels).

    Returns
    -------
    NDArray
        RIRs in B format of shape (num_time_samples, num_channels) [w, y, z, x] ordering.
    """
    # Assume 4 unit vectors for tetrahedral mic (each row is [x, y, z])
    dirs = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    #### WRITE YOUR CODE HERE ####
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    theta = np.acos(z)
    phi = np.atan2(y, x)

    # Create SN3D-normalized real SH basis functions (ACN order)
    # Order: [Y_0^0, Y_1^-1, Y_1^0, Y_1^1] => [W, Y, Z, X]
    Y_00 = 1 / np.sqrt(4 * np.pi) * np.ones_like(theta)
    Y_1m1 = np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.sin(phi)
    Y_10 = np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
    Y_11 = np.sqrt(3 / (4 * np.pi)) * np.sin(theta) * np.cos(phi)

    # Stack SH functions into shape (num_mic_dirs, num_channels)
    Y = np.column_stack((Y_00, Y_1m1, Y_10, Y_11))

    # Invert to get A → B transform
    Y_inv = np.linalg.pinv(Y)

    # Multiply wth inverted matrix with A-format RIRs to get B-format RIRs of
    # shape (num_time_samples, num_channels). Use einsum
    rirs_Bformat = (Y_inv @ rirs_Aformat.T).T

    # Return B-format RIRs of shape: (num_time_samples, num_channels) in ACN/SN3D
    return rirs_Bformat


def convert_srir_to_brir(srirs: NDArray, hrir_sh: NDArray,
                         head_orientations: ArrayLike) -> NDArray:
    """
    Convert SRIRs to BRIRs for specific orientations using spherical harmonic transformation.

    Parameters
    ----------
    srirs : NDArray
        SRIRs of shape (num_pos, num_ambi_channels, num_time_samp).
    hrir_sh : NDArray
        HRIRs encoded in SH domain of shape (num_ambi_channels, 2, num_time_samps).
    head_orientations : ArrayLike
        Head orientations of shape (num_ori, 2).

    Returns
    -------
    NDArray
        BRIRs of shape (num_pos, num_ori, num_time_samples, 2).
    """
    ambi_order = int(np.sqrt(srirs.shape[1] - 1))
    num_receivers = srirs.shape[0]
    num_freq_bins = 2**int(np.ceil(np.log2(srirs.shape[-1])))

    # take FFT of SRIRs - size is num_ambi_channels x num_receivers x num_time_samples
    ambi_rtfs = rfft(srirs, num_freq_bins, axis=-1)

    # take FFT of SH-HRIRs these are of shape num_ambi_channels x 2 x num_freq_samples
    ambi_hrtfs = rfft(hrir_sh, n=num_freq_bins, axis=-1)
    logger.info("Done calculating FFTs")

    num_orientations = head_orientations.shape[0]
    brirs = np.zeros((num_receivers, num_orientations, num_freq_bins, 2))

    #### WRITE YOUR CODE HERE ####

    # loop through receiver positions
    for rec_pos_idx in tqdm(range(num_receivers)):

        # get current SRIR FFT = shape is num_ambi_channels x num_freqs
        srir_fft = ambi_rtfs[rec_pos_idx, :, :]

        # loop through head orientations
        for ori_idx in range(num_orientations):

            # get current head orientation
            head_ori = head_orientations[ori_idx, :]
            yaw, pitch, roll = np.deg2rad(head_ori)

            # get rotation matrix in the opposite direction - size num_freq_bins x num_ambi_channels
            rotation_matrix = spa.sph.sh_rotation_matrix(N_sph=ambi_order, yaw=yaw, pitch=pitch, roll=0, sh_type='real')
            
            rotated_hrtfs = np.einsum('ij, jfk -> ifk', rotation_matrix, ambi_hrtfs)

            # get the binaural room transfer function by conjugating
            # freq-domain SRIRs and multiplying them with SH-HRTFs
            brir_fft = np.einsum('cha, ca -> ha', rotated_hrtfs.conj(), srir_fft) # c = ambi_channels, h = ears, a = freq

            # get the BRIR by taking an inverse FFT and save it to current
            # receiver position and orientation index
            brirs[rec_pos_idx, ori_idx, :, :] = irfft(brir_fft, n=num_freq_bins, axis=-1).T

    return brirs
