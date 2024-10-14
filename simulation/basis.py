####################################################################################################
#                                             basis.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/02/23                                                                                #
#                                                                                                  #
# Purpose: Defines a main structure for MRS metabolite basis sets. Encapsulates all information    #
#          and holds definitions to compute various aspects.                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

# own
from loading.loadBasis import loadBasisAsFSL

import numpy as np
from scipy.signal import resample, butter, filtfilt



#**************************************************************************************************#
#                                           Class Basis                                            #
#**************************************************************************************************#
#                                                                                                  #
# The main structure for MRS metabolite basis sets.                                                #
#                                                                                                  #
#**************************************************************************************************#
class Basis():

    #*************************#
    #   initialize instance   #
    #*************************#
    def __init__(self, path2basis, fmt=''):
        """
        Main init for the MyMRS class.

        @param path2basis -- The path to the basis set folder.
        @param fmt -- The data format to be selected.
        """
        basis = loadBasisAsFSL(path2basis)

        if fmt.lower() == 'cha_philips':
            basis = self.rescale(basis, mets=['Mac'], metScales=[0.40929292785161303])
        elif fmt.lower() == 'biggaba':
            basis = self.reformat(basis, 2000, 2048,
                                  ignore=['Cit', 'EtOH', 'Phenyl', 'Ser', 'Tyros', 'bHB', 'bHG'])
            basis = self.rescale(basis)
        elif fmt.lower() == 'fmrsinpain':
            basis = self.reformat(basis, 2000, 2048, ignore=['CrCH2', 'EA', 'H2O', 'Ser'])
            # basis = self.rescale(basis, mets=['Mac', 'Glc'],
            #                       metScales=[0.40929292785161303, 0.3657869584081066])

        self.basisFSL = basis
        self.names = [n.split('.')[0] for n in basis.names]
        self.n_metabs = len(self.names)
        self.fids = basis._raw_fids
        self.bw = basis.original_bw
        self.dwelltime = float(basis.original_dwell)
        self.n = basis.original_points
        self.t = np.arange(self.dwelltime, self.dwelltime * (self.n + 1), self.dwelltime)
        self.f = np.arange(- self.bw / 2, self.bw / 2, self.bw / self.n)
        self.ppm = np.fft.ifftshift(basis.original_ppm_shift_axis)
        self.cf = float(basis.cf)


    #*************************#
    #   get formatted basis   #
    #*************************#
    def reformat(self, basis, bw, points, ignore=[]):
        """
        Reformat the basis set to a given bandwidth and number of points.

        @param basis -- The basis set to reformat.
        @param bw -- The bandwidth to reformat to.
        @param points -- The number of points to reformat to.
        @param ignore -- The metabolites to ignore.

        @returns -- The reformatted basis set.
        """
        basis._raw_fids = basis.get_formatted_basis(bw, points, ignore=ignore)
        basis._dt = 1. / bw
        basis._names = basis.get_formatted_names(ignore=ignore)
        return basis


    #***********************#
    #   rescale basis set   #
    #***********************#
    def rescale(self, basis, scale=0.7772748317447753, mets=[], metScales=[]):
        """
        Rescale the basis set with separate metabolite scaling (to match predefined values of
        the concentration ranges).

        @param basis -- The basis set to rescale.
        @param scale -- The scaling factor (default is the ISMRM 2016 challenge data).
        @param mets -- The metabolites to scale.
        @param metScales -- The scaling factors for the metabolites. Compute as:
                            mean(abs((basis / mean(abs(basis._raw_fids[:, nonMetIdx])))[:, metIdx])).
        """
        for i, met in enumerate(mets):
            metIdx = basis.names.index(met)
            nonMetIdx = np.arange(len(basis.names)) != metIdx
            basis._raw_fids /= np.mean(np.abs(basis._raw_fids[:, nonMetIdx]))
            basis._raw_fids[:, metIdx] *= metScales[i] / np.mean(np.abs(basis._raw_fids[:, metIdx]))
        basis._raw_fids *= scale / np.mean(np.abs(basis._raw_fids))
        return basis