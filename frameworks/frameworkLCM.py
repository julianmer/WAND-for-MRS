####################################################################################################
#                                         frameworkLCM.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 20/06/24                                                                                #
#                                                                                                  #
# Purpose: Python wrapper for the LCModel optimization framework for least-squares fitting         #
#          of spectra.                                                                             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import multiprocessing
import numpy as np
import os
import shutil
import subprocess
import time
import torch

from fsl_mrs.utils import mrs_io

# own
from frameworks.framework import Framework
from loading.lcmodel import read_LCModel_coord



#**************************************************************************************************#
#                                      Class FrameworkLCModel                                      #
#**************************************************************************************************#
#                                                                                                  #
# The framework wrapper for the LCModel fitting tool.                                              #
#                                                                                                  #
#**************************************************************************************************#
class FrameworkLCModel(Framework):
    def __init__(self, basis_dir, basisFmt='', specType='synth', dataType='none', control=None,
                 multiprocessing=True, ppmlim=(0.5, 4.2), conj=True, ignore='default',
                 save_path='', device='cpu'):
        Framework.__init__(self, basis_dir,
                           basisFmt=basisFmt, specType=specType, dataType=dataType, ppmlim=ppmlim)

        self.basisFSL = self.basisObj.basisFSL
        self.control = control
        self.multiprocessing = multiprocessing
        self.save_path = save_path
        self.conj = conj

        if isinstance(ignore, str):
            if ignore.lower() == 'default': ignore = ['Lip13a', 'Lip13b', 'Lip09', 'Lip20',
                                                      'MM09', 'MM12', 'MM14', 'MM17', 'MM20',
                                                      '-CrCH2', 'CrCH2']
            elif ignore.lower() == 'default_wmm': ignore = ['Lip13a', 'Lip13b', 'Lip09', 'Lip20',
                                                            '-CrCH2', 'CrCH2']
            elif ignore.lower() == 'default_wlip': ignore = ['MM09', 'MM12', 'MM14', 'MM17',
                                                             'MM20', '-CrCH2', 'CrCH2']
            elif ignore.lower() == 'default_wmm_wlip': ignore = ['-CrCH2', 'CrCH2']
            elif ignore.lower() == 'none': ignore = []
            elif ignore.lower() == 'custom':
                # ignore = ['-CrCH2', 'CrCH2', 'EA', 'H2O', 'Ser']
                ignore = ['MM09', 'MM12', 'MM14', 'MM17', 'MM20']
            else: raise ValueError(f'Unknown preset string "{ignore}"... Please use one of '
                                   f'the predefined or provide a list of metabolite names!')
        elif not isinstance(ignore, list):
            raise ValueError('Ignore must be a list of metabolite names or a string!')

        self.device = torch.device(device)

        # # TODO: convert basis to LCModel format (if necessary)
        # if not basis_dir.lower().endswith('.basis'):
        #     raise ValueError('Basis file must be in .basis format')
        # # ATTENTION: for now, we use hardcoded paths as below!

        # parse control file
        if control is not None:
            self.control = control.split('\n')

            # adjust ppm limits
            for i, line in enumerate(self.control):
                if line.startswith('ppmst='):
                    self.control[i] = f'ppmst={ppmlim[1]}'
                if line.startswith('ppmend='):
                    self.control[i] = f'ppmend={ppmlim[0]}'

            # overwrite ignore metabolites
            for i, line in enumerate(self.control):
                if line.startswith('nomit='):
                    self.control[i] = f'nomit={len(ignore)}'
                    for j, met in enumerate(ignore):
                        self.control.insert(i+j+1, f'chomit({j+1})=\'{met}\'')
                    break
        else:
            lines = []
            lines.append(f"$LCMODL")

            # lines.append(f"nunfil={self.basisFSL.original_points}")  # data points
            lines.append(f"nunfil=1536")  # hardcoded (basis set might differ from data) -> TODO: infer from data
            lines.append(f"deltat={self.basisFSL.original_dwell}")  # dwell time
            lines.append(f"hzpppm={self.basisFSL.cf}")  # field strength in MHz
            lines.append(f"ppmst={ppmlim[1]}")
            lines.append(f"ppmend={ppmlim[0]}")

            lines.append(f"dows=F")   # 'T' <-> do water scaling
            lines.append(f"doecc=F")   # 'T' <-> do eddy current correction
            lines.append(f"neach=99")   # number of metabolites to plot fit individually

            # remember path is from cwd set in subprocess
            if basisFmt.lower() == 'fmrsinpain':
                lines.append(f"filbas='../../Data/BasisSets/basis_fMRSinPain/"
                             f"LCModel_Universal_Philips_UnEdited_PRESS_22_.BASIS'")
            elif basisFmt.lower() == 'cha_philips':
                lines.append(f"filbas='../../Data/BasisSets/ISMRM_challenge/press3T_30ms_Philips.BASIS'")
            else:
                lines.append(f"filbas='../../Data/BasisSets/ISMRM_challenge/press3T_30ms.BASIS'")

            lines.append(f"filraw='example.raw'")
            lines.append(f"filps='example.ps'")
            lines.append(f"filcoo='example.coord'")

            lines.append(f"lcoord=9")  # 0 <-> surpress creation of coord file, 9 <-> don't surpress
            lines.append(f"nomit={len(ignore)}")
            for i, met in enumerate(ignore):
                lines.append(f"chomit({i+1})=\'{met}\'")
            lines.append(f"namrel='Cr+PCr'")

            # lines.append(f"nratio=0")   # number of soft constraints (default 12, see manual 11.8)
            # lines.append(f"sddegz=6")   # 6 <-> eddy current correction (see manual 5.3.4)
            # lines.append(f"dkntmn=0.5")   # limit knot spacing of baseline (max 1/3 of ppm range)
            lines.append(f"$END")

            self.control = lines


    #**********************#
    #   forward function   #
    #**********************#
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


    #*********************#
    #   input to output   #
    #*********************#
    def forward(self, x, x0=None):
        assert x0 is None, 'Initial values not supported... (please remove x0)'

        theta = self.lcmodel_minimize(x)
        return theta


    #*********************#
    #   LCModel fitting   #
    #*********************#
    def lcmodel_minimize(self, x):
        thetas, crlbs = [], []
        x = x[:, 0] + 1j * x[:, 1]
        x = x.detach().cpu().numpy()   # push to cpu and numpy
        fids = np.fft.ifft(x, axis=-1)   # to time domain
        if self.conj: fids = np.conjugate(fids)   # conjugate if necessary

        # create temporary directory
        if self.save_path == '': path = os.getcwd() + '/tmp/'
        else: path = os.getcwd() + '/' + self.save_path
        if not os.path.exists(path): os.makedirs(path)

        # run
        if self.multiprocessing:   # multi threading
            tasks = [(fids[i], i, path) for i in range(fids.shape[0])]
            with multiprocessing.Pool(None) as pool:
                thetas, crlbs = zip(*pool.starmap(self.lcm_forward, tasks))

        else:  # loop
            for i, fid in enumerate(fids):
                theta, crlb = self.lcm_forward(fid, i, path)
                thetas.append(theta)
                crlbs.append(crlb)

        # remove temporary folder
        if self.save_path == '':
            shutil.rmtree(os.getcwd() + '/tmp/', ignore_errors=True)
        else:
            # ... or save control file to save path
            with open(f'{path}/control', 'w') as file:
                file.write('\n'.join(self.control))

        return torch.from_numpy(np.array(thetas)).to(self.device), \
               torch.from_numpy(np.array(crlbs)).to(self.device)


    #************************#
    #   write to .raw file   #
    #************************#
    def to_raw(self, fid, file_path, header=" $NMID\n  id='', fmtdat='(2E15.6)'\n $END\n"):
        with open(file_path, 'w') as file:
            file.write(header)
            for num in fid:
                file.write(f"  {num.real:.6E} {num.imag:.6E}\n")


    #*************************#
    #   read from .raw file   #
    #*************************#
    def from_raw(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.split()[0] == '$END': break
            fid = [complex(float(line.split()[0]),
                           float(line.split()[1])) for line in lines[i+1:]]
        return np.array(fid)


    #*************************#
    #   run LCModel wrapper   #
    #*************************#
    def lcm_forward(self, fid, idx=0, path=os.getcwd() + '/tmp/'):
        # transform to a .raw file
        self.to_raw(fid, f'{path}/temp{idx}.raw')

        # run LCModel
        self.initiate(f'{path}/temp{idx}.raw')

        # wait for .coord file
        while not os.path.exists(f'{path}/temp{idx}.coord'): time.sleep(1e-3)   # 1ms

        # read .coord file
        metabs, concs, crlbs, tcr = read_LCModel_coord(f'{path}/temp{idx}.coord',
                                                       meta=False)
        # sort concentrations by basis names
        concs = [concs[metabs.index(met)] if met in metabs else 0.0
                 for met in self.basisFSL._names]
        crlbs = [crlbs[metabs.index(met)] if met in metabs else 999.0
                 for met in self.basisFSL._names]
        return concs, crlbs


    #******************************#
    #   initiate routine on .raw   #
    #******************************#
    def initiate(self, file_path):
        # write control file
        for i, line in enumerate(self.control):
            if line.startswith('filraw='): self.control[i] = f'filraw=\'{file_path}\''
            if line.startswith('filps='): self.control[i] = f'filps=\'{file_path[:-4]}.ps\''
            if line.startswith('filcoo='): self.control[i] = f'filcoo=\'{file_path[:-4]}.coord\''

        msg = '\n'.join(self.control)
        msg = msg.encode('utf-8')

        # run LCModel
        proc = subprocess.Popen(
            ['LCModel.exe', ],
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd() + '/frameworks/',
        )
        stdout_value, stderr_value = proc.communicate(msg)

        # error handling
        if not stdout_value == b'': print(stdout_value)
        if not stderr_value == b'': print(stderr_value)


    #**************************#
    #   setter for save path   #
    #**************************#
    def set_save_path(self, path):
        self.save_path = path