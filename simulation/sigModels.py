####################################################################################################
#                                           sigModels.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/07/22                                                                                #
#                                                                                                  #
# Purpose: Definitions of various MRS signal models.                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import torch

from fsl_mrs.core import MRS
from fsl_mrs.models import getModelFunctions, getModelJac
from fsl_mrs.utils.misc import calculate_lap_cov



#**************************************************************************************************#
#                                          Class SigModel                                          #
#**************************************************************************************************#
#                                                                                                  #
# The base class for the MRS signal models. Defines the necessary attributes and methods a signal  #
# model should implement.                                                                          #
#                                                                                                  #
#**************************************************************************************************#
class SigModel():

    #*************************#
    #   initialize instance   #
    #*************************#
    def __init__(self, basis, baseline, order, first, last, t, f):
        self.basis = basis
        self.first, self.last = first, last

        if not baseline:
            self.baseline = \
            torch.from_numpy(self.baseline_init(order, first, last))
        else:
            self.baseline = baseline

        self.t = torch.from_numpy(t)
        self.f = torch.from_numpy(f)
        self.basis = torch.from_numpy(basis)


    #********************#
    #   parameter init   #
    #********************#
    def initParam(self, specs):
        pass


    #*******************#
    #   forward model   #
    #*******************#
    def forward(self, theta):
        pass


    #***************#
    #   regressor   #
    #***************#
    def regress_out(self, x, conf, keep_mean=True):
        """
        Linear deconfounding

        Ref: Clarke WT, Stagg CJ, Jbabdi S. FSL-MRS: An end-to-end spectroscopy analysis package.
        Magnetic Resonance in Medicine 2021;85:2950–2964 doi: https://doi.org/10.1002/mrm.28630.
        """
        if isinstance(conf, list):
            confa = np.squeeze(np.asarray(conf)).T
        else:
            confa = conf
        if keep_mean:
            m = np.mean(x, axis=0)
        else:
            m = 0
        return x - confa @ (np.linalg.pinv(confa) @ x) + m


    #*******************#
    #   baseline init   #
    #*******************#
    def baseline_init(self, order, first, last):
        x = np.zeros(self.basis.shape[0], complex)
        x[first:last] = np.linspace(-1, 1, last - first)
        B = []
        for i in range(order + 1):
            regressor = x ** i
            if i > 0:
                regressor = self.regress_out(regressor, B, keep_mean=False)

            B.append(regressor.flatten())
            B.append(1j * regressor.flatten())

        B = np.asarray(B).T
        tmp = B.copy()
        B = 0 * B
        B[first:last, :] = tmp[first:last, :].copy()
        return B



#**************************************************************************************************#
#                                         Class VoigtModel                                         #
#**************************************************************************************************#
#                                                                                                  #
# Implements a signal model as mentioned in [1] (a Voigt signal model).                            #
#                                                                                                  #
# [1] Clarke, W.T., Stagg, C.J., and Jbabdi, S. (2020). FSL-MRS: An end-to-end spectroscopy        #
#     analysis package. Magnetic Resonance in Medicine, 85, 2950 - 2964.                           #
#                                                                                                  #
#**************************************************************************************************#
class VoigtModel(SigModel):

    #*************************#
    #   initialize instance   #
    #*************************#
    def __init__(self, basis, first, last, t , f, baseline=None, order=2):
        """
        Main init for the VoigtModel class.

        @param basis -- The basis set of metabolites.
        @param baseline -- The baseline used to fit the spectra.
        """
        super(VoigtModel, self).__init__(basis, baseline, order,
                                         first=first, last=last, t=t, f=f)

    #********************#
    #   parameter init   #
    #********************#
    def initParam(self, specs, mode='fsl', basisFSL=None):
        """
        Initializes the optimization parameters.

        @param specs -- The batch of specs to get initializations for.
        @param mode -- The initialization mode (default: 'fsl').
        @param basisFSL -- The basis set of metabolites as (FSL) MRS object (default: None).

        @returns -- The optimization parameters.
        """
        if mode.lower() == 'fsl':
            specs = specs[:, 0] + 1j * specs[:, 1]
            theta = np.zeros((specs.shape[0], self.basis.shape[1] + 11))
            for i, spec in enumerate(specs):
                specFSL = MRS(FID=np.fft.ifft(spec.cpu().numpy()),
                              basis=basisFSL,
                              cf=basisFSL.cf,
                              bw=basisFSL.original_bw)
                specFSL.processForFitting()
                theta[i, :] = init(specFSL,
                                   metab_groups=[0],
                                   baseline=self.baseline.numpy(),
                                   ppmlim=[0.2, 4.2])
        elif mode.lower() == 'random':
            theta = np.zeros((specs.shape[0], self.basis.shape[1] + 11))
            theta[:, :self.basis.shape[1]] = np.random.rand(specs.shape[0], self.basis.shape[1])
        else:
            raise ValueError('Unknown initialization mode: ' + mode)
        return torch.Tensor(theta)


    #*******************#
    #   forward model   #
    #*******************#
    def forward(self, theta, sumOut=True, baselineOut=False, phase1=True):
        """
        The (forward) signal model.

        @param theta -- The optimization parameters.
        @param sumOut -- Whether to sum over the metabolites (default: True).
        @param baselineOut -- Whether to return the baseline (default: False).
        @param phase1 -- Whether to include the first-order phase (default: True).

        @returns -- The forward model function.
        """
        self.t , self.f = self.t.to(theta.device), self.f.to(theta.device)
        self.basis = self.basis.to(theta.device)

        n = self.basis.shape[1]
        g = 1

        con = theta[:, :n]  # concentrations
        gamma = theta[:, n:n + g]  # lorentzian blurring
        sigma = theta[:, n + g:n + 2 * g]  # gaussian broadening
        eps = theta[:, n + 2 * g:n + 3 * g]  # frequency shift
        phi0 = theta[:, n + 3 * g]  # global phase shift
        phi1 = theta[:, n + 3 * g + 1]  # global phase ramp
        b = theta[:, n + 3 * g + 2:]  # baseline params

        # compute m(t) * exp(- (1j * eps + gamma + sigma ** 2 * t) * t)
        lin = torch.exp(- (1j * eps + gamma + (sigma ** 2) * self.t) * self.t)
        ls = lin[..., None] * self.basis
        S = torch.fft.fft(ls, dim=1)

        # compute exp(-1j * (phi0 + phi1 * nu)) * con * S(nu)
        if phase1: ex = torch.exp(-1j * (phi0[..., None] + phi1[..., None] * self.f))
        else: ex = torch.exp(-1j * phi0[..., None])
        fd = ex[:, :, None] * con[:, None, :] * S

        # add baseline
        if self.baseline is not None:
            self.baseline = self.baseline.to(theta.device)

            # compute baseline
            if len(self.baseline.shape) > 2:
                ba = torch.einsum("ij, ikj -> ik", b.cfloat(), self.baseline)
            else:
                ba = torch.einsum("ij, kj -> ik", b.cdouble(), self.baseline)

        if sumOut: fd = fd.sum(-1) + ba
        if baselineOut: return fd, ba
        return fd


    #*********************#
    #   gradient vector   #
    #*********************#
    def gradient(self, theta, specs=None, constr=False):
        """
        The gradient of the signal model.

        @returns -- The gradient.
        """
        n = self.basis.shape[1]
        g = 1

        # # ! make sure specs are processed, otherwise call: processSpec(specs)
        # specs = specs[:, 0] + 1j * specs[:, 1]

        con = theta[:, :n]  # concentrations
        gamma = theta[:, n:n + g]  # lorentzian blurring
        sigma = theta[:, n + g:n + 2 * g]  # gaussian broadening
        eps = theta[:, n + 2 * g:n + 3 * g]  # frequency shift
        phi0 = theta[:, n + 3 * g]  # global phase shift
        phi1 = theta[:, n + 3 * g + 1]  # global phase ramp
        b = theta[:, n + 3 * g + 2:]  # baseline params

        # compute m(t) * exp(- (1j * eps + gamma + sigma ** 2 * t) * t)
        lin = torch.exp(- (1j * eps + gamma + (sigma ** 2) * self.t) * self.t)
        ls = lin[..., None] * self.basis
        S = torch.fft.fft(ls, dim=1)

        # compute exp(-1j * (phi0 + phi1 * nu))
        ex = torch.exp(-1j * (phi0[..., None] + phi1[..., None] * self.f))
        fd = ex * torch.sum(con[:, None, :] * S, -1)
        ea = ex[..., None] * con[:, None, :]

        Sg = torch.fft.fft(- self.t[None, :, None] * ls, dim=1)
        Ss = torch.fft.fft(- 2 * sigma[..., None] * self.t[None, :, None] ** 2 * ls, dim=1)
        Se = torch.fft.fft(- 1j * self.t[None, :, None] * ls, dim=1)

        dc = ex[..., None] * S
        dg = torch.sum(ea * Sg, -1)
        ds = torch.sum(ea * Ss, -1)
        de = torch.sum(ea * Se, -1)
        dp0 = - 1j * torch.sum(ea * S, -1)
        dp1 = - 1j * self.f * torch.sum(ea * S, -1)

        if not len(self.baseline.shape) > 2:
            fd += torch.einsum("ij, kj -> ik", b.cdouble(), self.baseline)
            db = self.baseline.repeat((specs.shape[0], 1, 1))
        elif type(self.baseline) is torch.Tensor:
            fd += torch.einsum("ij, ikj -> ik", b.cfloat(), self.baseline)
            db = self.baseline

        dS = torch.cat((dc, dg.unsqueeze(-1), ds.unsqueeze(-1), de.unsqueeze(-1),
                        dp0.unsqueeze(-1), dp1.unsqueeze(-1), db), dim=-1)

        return dS

        # grad = torch.real((fd[..., None] * torch.conj(dS) + torch.conj(fd)[..., None] * dS -
        #                    torch.conj(specs)[..., None] * dS - specs[..., None] * torch.conj(dS)))
        #
        # return grad.sum(1)


    #**********************#
    #   CRLB computation   #
    #**********************#
    def crlb(self, theta, data, grad=None, sigma=None):
        """
        Computes the Cramer-Rao lower bound.

        @param theta -- The optimization parameters.
        @param data -- The data to compute the CRLB for.
        @param grad -- The gradient of the signal model (optional).
        @param sigma -- The noise std (optional).

        @returns -- The CRLB.
        """
        data = data[:, 0] + 1j * data[:, 1]

        if grad is None: grad = self.gradient(theta, data)
        if sigma is None:
            spec = self.forward(theta)
            if spec.shape[1] > data.shape[1]: spec = spec[:, self.first:self.last]
            sigma = torch.std(data - spec, dim=-1)[:, None, None]

        # compute the Fisher information matrix
        F0 = torch.diag(torch.ones(theta.shape[1]) * 1e-10)   # creates non-zeros on the diag
                                                              # (inversion can otherwise fail)
        F0 = F0.to(theta.device).unsqueeze(0)
        F = 1 / sigma **2 * torch.real(torch.permute(grad, [0, 2, 1]) @ torch.conj(grad))

        # compute the CRLB
        crlb = torch.sqrt(torch.linalg.inv(F + F0).diagonal(dim1=1, dim2=2))
        return crlb


    #********************************#
    #   CRLB computation using fsl   #
    #********************************#
    def crlb_fsl(self, theta, data, basis=None, grad=None, sigma=None):
        """
        Computes the Cramer-Rao lower bound.

        @param theta -- The optimization parameters.
        @param data -- The data to compute the CRLB for.
        @param basisFSL -- The basis set.
        @param grad -- The gradient of the signal model (optional).
        @param sigma -- The noise std (optional).

        @returns -- The CRLB.
        """
        data = data[:, 0] + 1j * data[:, 1]

        if basis is None: basis = self.basis.detach().cpu().numpy()

        _, _, forward, _, _ = getModelFunctions('voigt')
        jac = getModelJac('voigt')

        def forward_lim(theta):
            return forward(
                theta,
                self.f.detach().cpu().numpy(),
                self.t.detach().cpu().numpy(),
                basis,
                self.baseline.detach().cpu().numpy(),
                G=[0] * basis.shape[1],
                g=1,
            )[self.first:self.last]

        def jac_lim(theta):
            return jac(
                theta,
                self.f.unsqueeze(-1).detach().cpu().numpy(),
                self.t.unsqueeze(-1).detach().cpu().numpy(),
                basis,
                self.baseline.detach().cpu().numpy(),
                G=[0] * basis.shape[1],
                g=1, first=self.first, last=self.last)

        crlbs = np.zeros(theta.shape)
        for i in range(data.shape[0]):
            C = calculate_lap_cov(theta[i].detach().cpu().numpy(),
                                  forward_lim,
                                  data[i].detach().cpu().numpy(),
                                  jac_lim(theta[i].detach().cpu().numpy()).T)

            # scaling according to FSL-MRS
            crlbs[i] = np.sqrt(np.diag(C / 2))
        return torch.from_numpy(crlbs).to(theta.device)

