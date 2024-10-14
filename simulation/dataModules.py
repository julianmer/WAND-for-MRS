####################################################################################################
#                                         dataModules.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 10/08/22                                                                                #
#                                                                                                  #
# Purpose: Definition of data modules, taking care of loading and processing of the data.          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch

from fsl_mrs.utils import mrs_io

from spec2nii.Philips.philips import read_sdat, read_spar
from spec2nii.Philips.philips_data_list import _read_list

from torch.utils.data import DataLoader


# own
from loading.loadData import loadDataAsFSL, loadDataSetsAsFSL
from loading.loadBasis import loadBasisAsFSL
from loading.loadConc import loadConcsDir, load_EXCEL_conc
from loading.philips import read_Philips_data
from simulation.basis import Basis
from simulation.sigModels import VoigtModel
from simulation.simulation import simulateParam, simulateRW, simulatePeaks, params, stdConcs
from utils.components import randomWalk, randomPeak
from utils.processing import processBasis, processSpectra



#**************************************************************************************************#
#                                       Class SynthDataModule                                      #
#**************************************************************************************************#
#                                                                                                  #
# The data module to load synthetic data.                                                          #
#                                                                                                  #
#**************************************************************************************************#
class SynthDataModule(pl.LightningDataModule):

    #*************************#
    #   initialize instance   #
    #*************************#
    def __init__(self, basis_dir, nums_test, sigModel=None, params=params, concs=stdConcs,
                 basisFmt='', specType='synth'):
        super().__init__()
        self.basisObj = Basis(basis_dir, fmt=basisFmt)
        self.basis = processBasis(self.basisObj.fids)

        self.nums = nums_test

        self.params = params
        self.concs = concs

        if sigModel:
            self.sigModel = sigModel
        else:
            self.sigModel = VoigtModel(basis=self.basis, first=0, last=self.basisObj.n,
                                       t=self.basisObj.t, f=self.basisObj.f)


    #******************************#
    #   simulate a batch of data   #
    #******************************#
    def get_batch(self, batch):
        theta, noise = simulateParam(self.basisObj, batch, self.params, self.concs)
        spec, bl = self.sigModel.forward(torch.from_numpy(theta), sumOut=False, baselineOut=True)

        cleanSpec = processSpectra(spec, self.basis)
        spec = spec.sum(-1) + torch.from_numpy(noise) + bl
        spec = processSpectra(spec, self.basis)

        # add baseline signal as a training target
        bl = torch.from_numpy(np.stack((np.real(bl), np.imag(bl)), axis=1))
        cleanSpec = torch.cat((cleanSpec, bl[..., np.newaxis]), dim=-1)

        # add noise signal as a training target (if no more unpredictable signals are added)
        noise = torch.from_numpy(np.stack((np.real(noise), np.imag(noise)), axis=1))

        # add all unpredictable signals as one training target
        cleanSpec = torch.cat((cleanSpec, noise[..., np.newaxis]), dim=-1)

        # add a random walk to the spectra and as a training target
        if 'scale' in self.params and 'smooth' in self.params and 'limits' in self.params:
            # TODO: implement random walk as a PyTorch function and vectorize
            scale, smooth, lowLim, highLim = simulateRW(batch, self.params)
            rw = [randomWalk(waveLength=self.basis.shape[0], scale=scale[i], smooth=smooth[i],
                             ylim=[lowLim[i], highLim[i]]) for i in range(batch)]
            scale, smooth, lowLim, highLim = simulateRW(batch, self.params)
            rwCplx = [randomWalk(waveLength=self.basis.shape[0], scale=scale[i], smooth=smooth[i],
                                 ylim=[lowLim[i], highLim[i]]) for i in range(batch)]
            rw = torch.from_numpy(np.stack((np.array(rw), np.array(rwCplx)), axis=1))
            spec += rw   # add to signal
            cleanSpec[..., -1] += rw   # add to artificial channel

        # add random peaks with amplitude, linewidth, and phasing
        if 'numPeaks' in self.params and 'peakAmp' in self.params and \
                'peakWidth' in self.params and 'peakPhase' in self.params:
            nums = np.random.randint(self.params['numPeaks'][0], self.params['numPeaks'][1], batch)

            peaks = np.zeros((batch, self.basis.shape[0]), dtype=np.complex128)
            for i in range(batch):
                if nums[i] > 0:
                    pos = np.random.randint(0, self.basis.shape[0], (nums[i], 1))
                    amps, widths, phases = simulatePeaks(self.basis, nums[i], self.params)

                    peaks[i] = randomPeak(waveLength=self.basis.shape[0], batch=nums[i],
                                          amp=amps, pos=pos, width=widths, phase=phases).sum(0)

            peaks = torch.from_numpy(np.stack((np.real(peaks), np.imag(peaks)), axis=1))
            spec += peaks   # add to signal
            cleanSpec[..., -1] += peaks   # add to artificial channel

        return spec, cleanSpec, theta


    #**************************#
    #   create test data set   #
    #**************************#
    def test_dataloader(self):
        x, y, t = self.get_batch(self.nums)
        return DataLoader(list(zip(x, y, t)), num_workers=4, batch_size=self.nums)



#**************************************************************************************************#
#                                     Class ChallengeDataModule                                    #
#**************************************************************************************************#
#                                                                                                  #
# The data module to load synthetic data of the ISMRM 2016 Fitting Challenge.                      #
#                                                                                                  #
#**************************************************************************************************#
class ChallengeDataModule(pl.LightningDataModule):

    #*************************#
    #   initialize instance   #
    #*************************#
    def __init__(self,
                 data_dir='../Data/DataSets/ISMRM_challenge/datasets_JMRUI_WS/',
                 basis_dir='../Data/BasisSets/ISMRM_challenge/basisset_JMRUI/',
                 truth_dir='../Data/DataSets/ISMRM_challenge/ground_truth/',
                 nums_cha=None, pre_pro=False):
        super().__init__()
        self.nums_cha = nums_cha
        self.pre_pro = pre_pro

        self.basis = loadBasisAsFSL(basis_dir)
        self.concs, self.crlbs = loadConcsDir(truth_dir)

        # ground truths
        concs = [[c[met] for met in self.basis._names] for c in self.concs]  # sort by basis names
        self.concs = torch.Tensor(concs)[:, :self.basis.n_metabs]

        # load data
        files = os.listdir(data_dir)[:nums_cha]
        data = np.array([mrs_io.read_FID(data_dir + file).mrs().FID for file in files])
        data = torch.from_numpy(data)

        # to frequency domain (stack real and imaginary part)
        data = torch.fft.fft(data, dim=-1)
        self.data = torch.stack((data.real, data.imag), dim=1)


    #**************************#
    #   create test data set   #
    #**************************#
    def test_dataloader(self):
        data = list(zip(self.data[:self.nums_cha],
                        self.data[:self.nums_cha].unsqueeze(-1),
                        self.concs[:self.nums_cha]))
        return DataLoader(data, num_workers=4, batch_size=self.nums_cha)


    #**********************#
    #   scenario indices   #
    #**********************#
    def scenario_indices(self):
        return {
            'Lorentzian': [0, 11, 21],  # 1-3
            'Gaussian': [22, 23, 24],  # 4-6
            'No MM': [2],  # 11
            'SNR 1': [3],  # 12
            'SNR 2': [6],  # 15
            'SNR 3': [9],  # 18
            'SNR 4': [0],  # 1
            'SNR 5': [13],  # 21
            'Eddy Currents': [14],  # 22
            'Residual Water': [15, 16],  # 23, 24
            'Tumor-Like': [17, 18, 19, 20],  # 25-28
            'General': [25, 26, 27, 1, 4, 5, 7, 8, 10, 12],  # 7-10, 13, 14, 16, 17, 19, 20
            'Normal': list(range(14)) + list(range(21, 28))   # 0-13, 21-27
        }


    #**************************#
    #   indices mapping dict   #
    #**************************#
    def indices_mapping(self):
        return {1:0, 2:11, 3:21, 4:22, 5:23, 6:24, 7:25, 8:26, 9:27, 10:1, 11:2, 12:3,
                13:4, 14:5, 15:6, 16:7, 17:8, 18:9, 19:10, 20:12, 21:13, 22:14, 23:15,
                24:16, 25:17, 26:18, 27:19, 28:20}



#**************************************************************************************************#
#                                      Class InVivoDataModule                                      #
#**************************************************************************************************#
#                                                                                                  #
# The data module to load in-vivo data.                                                            #
#                                                                                                  #
#**************************************************************************************************#
class InVivoDataModule(pl.LightningDataModule):

        #*************************#
        #   initialize instance   #
        #*************************#
        def __init__(self,
                    data_dir='../Data/DataSets/InVivo/',
                    basis_dir='../Data/BasisSets/ISMRM_challenge/basisset_JMRUI/',
                    nums_test=None, pre_pro=False, verbose=0):
            super().__init__()
            self.nums_test = nums_test
            self.pre_pro = pre_pro

            self.data_dir = data_dir

            self.basis_dir = basis_dir
            self.basis = loadBasisAsFSL(basis_dir)

            self.data = []
            self.refs = []
            # go through files in folder
            for file in os.listdir(data_dir):
                # if file is a folder
                if os.path.isdir(data_dir + '/' + file):
                    # go through files in sub-folder
                    for sub_file in os.listdir(data_dir + '/' + file):
                        # load fid
                        fid = loadDataAsFSL(data_dir + '/' + file + '/' + sub_file)
                        if not isinstance(fid, type(None)):
                            if 'act' in sub_file.lower():
                                if verbose: self.visualize(fid)
                                self.data.append(fid)
                            elif 'ref' in sub_file.lower():
                                self.refs.append(fid)
                else:   # load fid
                    fid = loadDataAsFSL(data_dir + '/' + file)
                    if not isinstance(fid, type(None)):
                        if 'act' in file.lower():
                            if verbose: self.visualize(fid)
                            self.data.append(fid)
                        elif 'ref' in file.lower():
                            self.refs.append(fid)


        #**************************#
        #   create test data set   #
        #**************************#
        def test_dataloader(self):
            data = torch.tensor(np.array([d.FID for d in self.data]))
            data = torch.fft.fft(data, dim=-1)
            data = torch.stack((data.real, data.imag), dim=1)
            return DataLoader(data[:self.nums_test], num_workers=4, batch_size=self.nums_test)


        #*******************************#
        #   create reference data set   #
        #*******************************#
        def ref_dataloader(self):
            refs = torch.tensor(np.array([r.FID for r in self.refs]))
            refs = torch.fft.fft(refs, dim=-1)
            refs = torch.stack((refs.real, refs.imag), dim=1)
            return DataLoader(refs[:self.nums_test], num_workers=4, batch_size=self.nums_test)




#**************************************************************************************************#
#                                       Class InVivoNSAModule                                      #
#**************************************************************************************************#
#                                                                                                  #
# The data module to load in-vivo data with individual NSA.                                        #
#                                                                                                  #
#**************************************************************************************************#
class InVivoNSAModule(pl.LightningDataModule):

        #*************************#
        #   initialize instance   #
        #*************************#
        def __init__(self,
                    data_dir='../Data/DataSets/fMRSinPain/SUBJECTS/SUBJECTS/',
                    basis_dir='../Data/BasisSets/ISMRM_challenge/basisset_JMRUI/',
                    basis_fmt='', nums_test=None, fMRS=False, coil_comb=False, verbose=0):
            super().__init__()
            self.nums_test = nums_test
            self.fMRS = fMRS
            self.coil_comb = coil_comb

            self.data_dir = data_dir
            self.basis = Basis(basis_dir, fmt=basis_fmt)

            self.data = []
            self.refs = []

            # go through files in folder
            for file in os.listdir(data_dir):
                # if file is a folder
                if os.path.isdir(data_dir + '/' + file):
                    # go through files in sub-folder
                    for sub_file in os.listdir(data_dir + '/' + file):
                        # load fid
                        if sub_file.lower().split('.')[-1] == 'list':
                            fids, h2o = self.load_DATALIST_data(data_dir + '/' + file + '/' + sub_file)
                            if self.fMRS and len(fids.shape) > 2 + int(not self.coil_comb):  # fMRS dim: (samples, [coils], NSA, f-blocks)
                                self.data.append(fids)
                                self.refs.append(h2o)
                            if not self.fMRS and len(fids.shape) <= 2 + int(not self.coil_comb):   # (samples, [coils], NSA)
                                self.data.append(fids)
                                self.refs.append(h2o)

                        elif sub_file.lower().split('.')[-1] == 'sdat' or sub_file.lower().split('.')[-1] == 'spar':
                            if 'ref' in sub_file.lower():
                                h2o = self.load_SDATSPAR_data(data_dir + '/' + file + '/' + sub_file)
                                self.refs.append(h2o)
                            else:
                                fids = self.load_SDATSPAR_data(data_dir + '/' + file + '/' + sub_file)
                                self.data.append(fids)
                else:
                    # load fid
                    if file.lower().split('.')[-1] == 'list':
                        fids, h2o = self.load_DATALIST_data(data_dir + '/' + file)
                        if self.fMRS and len(fids.shape) > 2 + int(not self.coil_comb):
                            self.data.append(fids)
                            self.refs.append(h2o)
                        if not self.fMRS and len(fids.shape) <= 2 + int(not self.coil_comb):
                            self.data.append(fids)
                            self.refs.append(h2o)

                    elif file.lower().split('.')[-1] == 'sdat' or file.lower().split('.')[-1] == 'spar':
                        if 'ref' in file.lower():
                            h2o = self.load_SDATSPAR_data(data_dir + '/' + file)
                            self.refs.append(h2o)
                        else:
                            fids = self.load_SDATSPAR_data(data_dir + '/' + file)
                            self.data.append(fids)


        #****************************#
        #   loading DATA LIST data   #
        #****************************#
        def load_DATALIST_data(self, path2data):
            df, num_dict, coord_dict, os_dict = _read_list(path2data[:-4] + 'list')
            sorted_data_dict = read_Philips_data(path2data[:-4] + 'data', df)

            fids = sorted_data_dict['STD_0'].squeeze()
            h2o = sorted_data_dict['STD_1'].squeeze()

            # naively combine channels
            if self.coil_comb:
                fids = fids.sum(1)
                h2o = h2o.sum(1)
            return fids, h2o


        #****************************#
        #   loading SDAT SPAR data   #
        #****************************#
        def load_SDATSPAR_data(self, path2data):
            params = read_spar(path2data[:-4] + 'SPAR')
            data = read_sdat(path2data[:-4] + 'SDAT',
                             params['samples'],
                             params['rows'])
            return data


        #**************************#
        #   create test data set   #
        #**************************#
        def test_dataloader(self):
            data = torch.tensor(np.array(self.data))
            data = torch.fft.fft(data, dim=-2 - int(self.fMRS) - int(not self.coil_comb))
            data = torch.stack((data.real, data.imag), dim=1)
            return DataLoader(data[:self.nums_test], num_workers=4, batch_size=self.nums_test)


        #*******************************#
        #   create reference data set   #
        #*******************************#
        def ref_dataloader(self):
            refs = torch.tensor(np.array(self.refs))
            refs = torch.fft.fft(refs, dim=-2 - int(self.fMRS) - int(not self.coil_comb))
            refs = torch.stack((refs.real, refs.imag), dim=1)
            return DataLoader(refs[:self.nums_test], num_workers=4, batch_size=self.nums_test)