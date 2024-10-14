####################################################################################################
#                                         frameworkFSL.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 08/10/23                                                                                #
#                                                                                                  #
# Purpose: FSL-MRS optimization framework for least-squares fitting of spectra.                    #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import multiprocessing
import numpy as np
import os
import pickle
import torch

from fsl_mrs.core import MRS
from fsl_mrs.utils import fitting, mrs_io

# own
from frameworks.framework import Framework



#**************************************************************************************************#
#                                        Class FrameworkFSL                                        #
#**************************************************************************************************#
#                                                                                                  #
# The framework wrapper for the FSL-MRS fitting tool.                                              #
#                                                                                                  #
#**************************************************************************************************#
class FrameworkFSL(Framework):
    def __init__(self, basis_dir, method='Newton', basisFmt='', specType='synth', dataType='none',
                 multiprocessing=False, ppmlim=(0.5, 4.2), baseline_order=2, conj=False,
                 unc='perc', save_path='', device='cpu'):
        Framework.__init__(self, basis_dir,
                           basisFmt=basisFmt, specType=specType, dataType=dataType, ppmlim=ppmlim)

        self.method = method   # 'Newton', 'MH'
        self.basisFSL = self.basisObj.basisFSL
        self.multiprocessing = multiprocessing
        self.baseline_order = baseline_order
        self.conj = conj
        self.unc = unc
        self.save_path = save_path

        self.device = torch.device(device)


    #**********************#
    #   forward function   #
    #**********************#
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


    #*********************#
    #   input to output   #
    #*********************#
    def forward(self, x, x0=None):
        theta = self.fsl_minimize(x, x0)
        return theta


    #*********************#
    #   FSL-MRS fitting   #
    #*********************#
    def fsl_minimize(self, x, x0=None):
        thetas, uncs = [], []
        x = x[:, 0] + 1j * x[:, 1]
        x = x.detach().cpu().numpy()   # push to cpu and numpy
        fids = np.fft.ifft(x, axis=-1)   # to time domain
        if self.conj: fids = np.conjugate(fids)   # conjugate if necessary

        # multi threading
        if self.multiprocessing:
            tasks = [(fid, self.basisFSL, self.method, x0, i) for i, fid in enumerate(fids)]
            with multiprocessing.Pool(None) as pool:
                thetas, uncs = zip(*pool.starmap(self.fsl_forward, tasks))

        else:  # loop
            for i, fid in enumerate(fids):
                theta, unc = self.fsl_forward(fid, self.basisFSL, self.method, x0, i)
                thetas.append(theta)
                uncs.append(unc)
        return torch.from_numpy(np.array(thetas)).to(self.device)[:, :self.basisFSL.n_metabs], \
               torch.from_numpy(np.array(uncs)).to(self.device)[:,:self.basisFSL.n_metabs]


    #*************************#
    #   FSL-MRS optimization   #
    #*************************#
    def fsl_forward(self, fid, basis, method, x0=None, idx=0):
        specFSL = MRS(FID=fid,
                      basis=basis,
                      cf=basis.cf,
                      bw=basis.original_bw)
        specFSL.processForFitting()

        init = x0 if x0 else None
        opt = fitting.fit_FSLModel(specFSL, method=method, x0=init, ppmlim=self.ppmlim,
                                   baseline_order=self.baseline_order)

        if self.save_path != '':
            if not os.path.exists(self.save_path): os.makedirs(self.save_path)
            opt.to_file(f'{self.save_path}/summary{idx}.csv', what='summary')
            opt.to_file(f'{self.save_path}/concs{idx}.csv', what='concentrations')
            opt.plot(specFSL, out=f'{self.save_path}/fit{idx}.png')
            pickle.dump(opt, open(f'{self.save_path}/opt{idx}.pkl', 'wb'))

            from fsl_mrs.utils import plotting
            plotting.plotly_fit(specFSL, opt).write_html(f'{self.save_path}/residuals{idx}.html')

        if self.unc == 'perc':
            params = opt.params
            with np.errstate(divide='ignore', invalid='ignore'):
                perc_SD = np.sqrt(opt.crlb[:len(params)]) / params * 100
                perc_SD[perc_SD > 999] = 999
                perc_SD[np.isnan(perc_SD)] = 999
            return params, perc_SD
        elif self.unc == 'crlb':
            return opt.params, opt.crlb
        else: raise ValueError('Invalid uncertainty type!')