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

    # Create SN3D-normalized real SH basis functions (ACN order)
    # Order: [Y_0^0, Y_1^-1, Y_1^0, Y_1^1] => [W, Y, Z, X]
    
    # az = cart2sph(dirs[:, 0], dirs[:, 1], dirs[:, 2])[0]  # azimuth
    # el = cart2sph(dirs[:, 0], dirs[:, 1], dirs[:, 2])[1]  # elevation
    # sh_basis = spa.sph.sh_matrix(1, dirs[:, 0], dirs[:, 1], type='real')
    # spaudio uses zenith so 90 - elevation
    
    x = dirs[:, 0]
    y = dirs[:, 1]
    z = dirs[:, 2]
    
    theta = np.arccos(z)
    phi = np.arctan2(y, x)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    Y0 = np.ones_like(theta) / np.sqrt(4 * np.pi)  # Y_0^0
    Y1 = np.sqrt(3 / (4 * np.pi)) * sin_theta * sin_phi # Y_1^-1
    Y2 = np.sqrt(3 / (4 * np.pi)) * cos_theta # Y_1^0
    Y3 = np.sqrt(3 / (4 * np.pi)) * sin_theta * cos_phi # Y_1^1

    # Stack SH functions into shape (num_mic_dirs, num_channels)
    sh_basis = np.column_stack((Y0, Y1, Y2, Y3))

    # Invert to get A → B transform
    sh_basis_inv = np.linalg.pinv(sh_basis)

    # Multiply wth inverted matrix with A-format RIRs to get B-format RIRs of
    # shape (num_time_samples, num_channels). Use einsum
    # h = sh_basis_inv @ rirs_Aformat
    h = np.einsum('mn,tn->tm', sh_basis_inv, rirs_Aformat)

    # Return B-format RIRs of shape: (num_time_samples, num_channels) in ACN/SN3D
    return h
