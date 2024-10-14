####################################################################################################
#                                          framework.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/12/22                                                                                #
#                                                                                                  #
# Purpose: Optimization framework.                                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import torch

# own
from simulation.basis import Basis
from simulation.sigModels import VoigtModel
from simulation.simulationDefs import *
from utils.auxiliary import general_minimize
from utils.processing import processBasis



#**************************************************************************************************#
#                                          Class Framework                                         #
#**************************************************************************************************#
#                                                                                                  #
# The framework allowing to define, train, and test models.                                        #
#                                                                                                  #
#**************************************************************************************************#
class Framework():
    def __init__(self, basis_dir, basisFmt='', specType='auto', dataType='none', ppmlim=None):
        self.basis_dir = basis_dir
        self.basisFmt = basisFmt
        self.specType = specType
        self.ppmlim = ppmlim

        # selection of the parameters and concentrations to be used for simulation
        if dataType == 'clean': self.ps, self.concs = cleanParams, stdConcsNorm
        elif dataType == 'clean_rw': self.ps, self.concs = cleanParamsRW, stdConcsNorm
        elif dataType == 'clean_rw_p': self.ps, self.concs = cleanParamsRWP, stdConcsNorm
        elif dataType == 'norm': self.ps, self.concs = normParams, normConcs
        elif dataType == 'norm_rw': self.ps, self.concs = normParamsRW, normConcs
        elif dataType == 'norm_rw_p': self.ps, self.concs = normParamsRWP, normConcs
        elif dataType == 'norm_rw_p+': self.ps, self.concs = paramsRWP, normConcs
        elif dataType == 'std': self.ps, self.concs = params, stdConcs
        elif dataType == 'std_rw': self.ps, self.concs = paramsRW, stdConcs
        elif dataType == 'std_rw_p': self.ps, self.concs = paramsRWP, stdConcs
        elif dataType == 'custom': self.ps, self.concs = customParams, customConcs
        else: self.ps, self.concs = None, None

        self.basisObj = Basis(basis_dir, fmt=basisFmt)
        self.basisObj.fids = processBasis(self.basisObj.fids)
        self.basisObj.basisFSL._raw_fids = self.basisObj.fids
        self.lossType = None

        # set ppm range to be used, 'auto' <-> use ppmlim to infer indices
        if specType.lower() == 'auto':
            assert ppmlim is not None, 'ppmlim needs to be specified for specType "auto"'
            self.first = (np.abs(self.basisObj.ppm - ppmlim[0])).argmin()
            self.last = (np.abs(self.basisObj.ppm - ppmlim[1])).argmin()
            self.skip = 1  # 1 -> no skip, 2 -> every second, ...
        elif specType.lower() == 'synth':
            self.first, self.last = self.basisObj.n - 256, self.basisObj.n - 12
            self.skip = 1  # 1 -> no skip, 2 -> every second, ...
        elif specType.lower() == 'invivo':  # in vivo data
            self.first, self.last = self.basisObj.n - 264, self.basisObj.n - 21
            self.skip = 1  # 1 -> no skip, 2 -> every second, ...
        elif specType.lower() == 'biggaba':  # big gaba data
            self.first, self.last = self.basisObj.n - 2 * 264, self.basisObj.n - 2 * 21
            self.skip = 2  # 1 -> no skip, 2 -> every second, ...
        elif specType.lower() == 'fmrsinpain':  # fMRS in pain data
            self.first, self.last = self.basisObj.n - 548, self.basisObj.n - 48
            self.skip = 1  # 1 -> no skip, 2 -> every second, ...
        elif specType.lower() == 'custom':
            self.first, self.last = self.basisObj.n - 412, self.basisObj.n - 2
            self.skip = 1  # 1 -> no skip, 2 -> every second, ...
        elif specType.lower() == 'all':
            self.first, self.last = 0, self.basisObj.n
            self.skip = 1
        else:
            raise ValueError('specType not recognized')

        self.sigModel = VoigtModel(basis=self.basisObj.fids, first=self.first, last=self.last,
                                   t=self.basisObj.t, f=self.basisObj.f)

    #*********************#
    #   input to output   #
    #*********************#
    def forward(self, x):
        pass


    #**********************#
    #   loss calculation   #
    #**********************#
    def loss(self, x, y, y_hat, type='mse'):
        pass


    #************************#
    #   scale the estimate   #
    #************************#
    def scaleEstimate(self, y, y_hat):
        y[y < 0] = 0
        scale = y.sum(-1).unsqueeze(-1)
        y_hat[y_hat < 0] = 0
        y_hat /= y_hat.sum(-1).unsqueeze(-1)
        return scale * y_hat


    #*************************#
    #   optimal referencing   #
    #*************************#
    def optimalReference(self, y, y_hat, type='scipy_l1'):
        # y_hat = self.scaleEstimate(y, y_hat)   # roughly scale the estimate

        if type == 'ls_l1':   # minimal l1 loss
            def err(w):
                w = torch.clamp(w, min=0)
                return torch.nn.L1Loss(reduction='none')(y, w * y_hat).mean()
            w = general_minimize(err, torch.ones(y.shape[0], 1))

        if type == 'scipy_l1':
            from scipy.optimize import minimize
            y_np, y_hat_np = y.detach().cpu().numpy(), y_hat.detach().cpu().numpy()
            w = np.ones(y.shape[0])
            for i in range(y.shape[0]):
                def err(w):
                    w = np.clip(w, 0, None)
                    return np.abs(y_np[i] - w * y_hat_np[i]).mean()
                w[i] = minimize(err, w[i], bounds=[(0, None)]).x
            w = torch.tensor(w).to(y.device).unsqueeze(-1)

        elif type == 'scale':
            y[y < 0] = 0
            w = y.sum(-1).unsqueeze(-1) / y_hat.sum(-1).unsqueeze(-1)

        else:
            raise ValueError(f'type "{type}" not recognized...')
        return w


    #****************************#
    #   loss on concentrations   #
    #****************************#
    def concsLoss(self, t, t_hat, type='smae'):
        t = t[:, :self.basisObj.n_metabs]
        t_hat = t_hat[:, :self.basisObj.n_metabs]

        if type == 'mae':    # mean absolute error
            return torch.nn.L1Loss(reduction='none')(t, t_hat)

        elif type == 'smae':   # scaled mean absolute error
            t_hat = self.scaleEstimate(t, t_hat)
            return torch.nn.L1Loss(reduction='none')(t, t_hat)

        elif type == 'msmae':   # minimal scaled mean absolute error
            w = self.optimalReference(t, t_hat, type='scipy_l1')
            return torch.nn.L1Loss(reduction='none')(t, w * t_hat)

        elif type == 'mse':   # mean squared error
            return torch.nn.MSELoss(reduction='none')(t, t_hat)

        elif type == 'smse':   # scaled mean squared error
            t_hat = self.scaleEstimate(t, t_hat)
            return torch.nn.MSELoss(reduction='none')(t, t_hat)

        elif type == 'mape':   # mean absolute percentage error
            mape = self.concsLoss(t, t_hat, type='mae')
            return 100 * mape / torch.clamp(t, min=torch.finfo(mape.dtype).eps)

        elif type == 'smape':   # scaled mean absolute percentage error
            smae = self.concsLoss(t, t_hat, type='smae')
            return 100 * smae / torch.clamp(t, min=torch.finfo(smae.dtype).eps)

        elif type == 'diff':   # difference
            return t - t_hat

        elif type == 'pe':   # percentage error
            return 100 * (t - t_hat) / torch.clamp(t, min=torch.finfo(t.dtype).eps)

        elif type == 'spe':   # scaled percentage error
            t_hat = self.scaleEstimate(t, t_hat)
            return 100 * (t - t_hat) / torch.clamp(t, min=torch.finfo(t.dtype).eps)

        elif type == 'mspe':   # minimal scaled percentage error
            t_hat = self.scaleEstimate(t, t_hat)
            def err(w):
                w = torch.nn.functional.relu(w)
                return self.concsLoss(t, w * t_hat, type='pe').abs().mean()
            w = general_minimize(err, torch.ones(t.shape[0], 1))
            return self.concsLoss(t, w * t_hat, type='pe')

        elif type == 'ccc':   # lin's concordance correlation coefficient
            from torchmetrics.regression import ConcordanceCorrCoef
            return ConcordanceCorrCoef(num_outputs=t.shape[1], reduction='none')(t, t_hat)

        elif type == 'sccc':   # scaled concordance correlation coefficient
            t_hat = self.scaleEstimate(t, t_hat)
            from torchmetrics.regression import ConcordanceCorrCoef
            return ConcordanceCorrCoef(num_outputs=t.shape[1])(t, t_hat)

        elif type == 'msccc':   # minimal scaled concordance correlation coefficient
            t_hat = self.scaleEstimate(t, t_hat)
            from torchmetrics.regression import ConcordanceCorrCoef
            def err(w):
                w = torch.nn.functional.relu(w)
                return ConcordanceCorrCoef(num_outputs=t.shape[1])(t, w * t_hat)
            w = general_minimize(err, torch.ones(t.shape[0], 1))
            return ConcordanceCorrCoef(num_outputs=t.shape[1])(t, w * t_hat)

        elif type == 'ktau':   # kendall's tau
            from torchmetrics.regression import KendallRankCorrCoef
            return KendallRankCorrCoef(num_outputs=t.shape[1])(t, t_hat)

        elif type == 'sktau':   # scaled kendall's tau
            t_hat = self.scaleEstimate(t, t_hat)
            from torchmetrics.regression import KendallRankCorrCoef

        else:
            raise ValueError('type not recognized')


    #********************#
    #   test the model   #
    #********************#
    def test_model(self, data):
        pass