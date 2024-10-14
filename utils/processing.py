####################################################################################################
#                                           processing.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 21/06/22                                                                                #
#                                                                                                  #
# Purpose: Some helpful functions for processing MRS data are defined here.                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import torch

from fsl_mrs.utils.preproc.nifti_mrs_proc import update_processing_prov, DimensionsDoNotMatch
from suspect.processing.denoising import sliding_gaussian


#*******************************#
#   process the basis spectra   #
#*******************************#
def processBasis(basis):
    """
        Processes the basis spectra.
        @param basis -- The basis spectra.
        @return -- The processed basis spectra.
    """
    # conjugate basis if necessary
    specBasis = np.abs(np.fft.fft(basis[:, 0]))
    if np.max(specBasis[:basis.shape[0] // 2]) > np.max(specBasis[basis.shape[0] // 2:]):
        basis = np.conjugate(basis)
    return basis # torch.from_numpy(basis).cfloat()


#*********************************#
#   process model input spectra   #
#*********************************#
def processSpectra(spectra, basis=None, conj=False):
    """
        Processes the input spectra.

        @param spectra -- The input spectra.
        @param basis -- The basis spectra.
        @param conj -- If True the spectra are conjugated (if necessary).

        @returns -- The processed spectra.
    """
    # conjugate spectra if necessary
    if conj and spectra[0, :spectra.shape[1] // 2].abs().max() > \
            spectra[0, spectra.shape[1] // 2:].abs().max():
        spectra = torch.conj(spectra)

    spectra = torch.stack((spectra.real, spectra.imag), dim=1)
    return spectra


#*********************************#
#   phase correction of spectra   #
#*********************************#
def phaseCorrection(spectra):
    """
        Phase correction of the spectra.

        @param spectra -- The spectra.

        @returns -- The phase corrected spectra. Aligned to the maximum peak.
    """
    spectra = spectra[:, 0] + 1j * spectra[:, 1]
    maxIdx = np.argmax(np.abs(spectra), axis=1)
    phase = np.angle(np.take_along_axis(spectra, maxIdx[:, None], axis=1))
    spectra = spectra * np.exp(-1j * phase)
    return np.stack((np.real(spectra), np.imag(spectra)), axis=1)


#***************************************#
#   own nifti eddy current correction   #
#***************************************#
def own_nifti_ecc(data, reference):
    """
        Eddy current correction for MRS data in the NIfTI format. Using the code from suspect.

        @param data -- The MRS data to be corrected.
        @param reference -- The reference data for the correction.

        @returns -- The corrected MRS data.
    """

    if data.shape != reference.shape \
            and reference.ndim > 4:
        raise DimensionsDoNotMatch('Reference and data shape must match'
                                   ' or reference must be single FID.')

    corrected_obj = data.copy()
    for dd, idx in data.iterate_over_dims(iterate_over_space=True):

        if data.shape == reference.shape:
            # reference is the same shape as data, voxel-wise and spectrum-wise iteration
            ref = reference[idx]
        else:
            # only one reference FID, only iterate over spatial voxels.
            ref = reference[idx[0], idx[1], idx[2], :]

        ec_smooth = sliding_gaussian(np.unwrap(np.angle(ref)), 32)
        ecc = np.exp(-1j * ec_smooth)
        corrected_obj[idx] = dd * ecc

    # update processing prov
    processing_info = f'{__name__}.ecc, '
    processing_info += f'reference={reference.filename}.'

    update_processing_prov(corrected_obj, 'Eddy current correction', processing_info)

    return corrected_obj