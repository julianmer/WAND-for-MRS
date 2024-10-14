####################################################################################################
#                                           visInVivo.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/02/24                                                                                #
#                                                                                                  #
# Purpose: Visualizes the in-vivo results.                                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np
import shutup; shutup.please()   # shut up warnings
import torch

from fsl_mrs.core.nifti_mrs import gen_nifti_mrs, split
from fsl_mrs.utils.preproc import nifti_mrs_proc as proc

from tqdm import tqdm

# own
from frameworks.frameworkFSL import FrameworkFSL
from frameworks.frameworkLCM import FrameworkLCModel
from loading.lcmodel import read_LCModel_fit
from loading.loadConc import loadConcs, loadConcsDir
from simulation.dataModules import *
from train import Pipeline
from utils.auxiliary import computeTotalMetabolites, renameMetabolites
from utils.processing import own_nifti_ecc
from visualizations.plotFunctions import *



#****************#
#   initialize   #
#****************#
config = {
    # path to a trained model
    'path2trained': './lightning_logs/ay9ctg9o/checkpoints/epoch=0-step=260608.ckpt',   # norm rw (fmrsinpain)

    # path to basis set
    'path2basis': '../Data/BasisSets/basis_fMRSinPain/',

    # for nn models
    'arch': 'unet',   # 'mlp', 'cnn', 'unet', 'aspp'
    'specType': 'fMRSinPain',  # 'synth', 'invivo', 'biggaba', 'fMRSinPain'
    'basisFmt': 'fMRSinPain',  # '', 'cha_philips', 'fMRSinPain'

    'dataType': 'invivo_ua',   # 'cha', 'invivo', 'invivo_ua', 'synth'
    'note': 'norm_rw',   # additional note

    # for lcm model
    'method': ['Newton', 'LCModel'],  # 'Newton', 'MH', 'LCModel'
    'ppmlim': (0.5, 4.2),  # ppm limits for the spectra
    'lcmodel_ignore': 'default',  # 'default', 'none', 'custom', ...
    'mm': True,  # include model internal macromolecules
    'lipids': True,  # include lipids in the basis set (only LCModel)
    'averages': -1,  # averages to run, i % averages == 0 (or -1 <-> 1, 2, 4, 8, ...)
    'shift_ref': 'cr',  # cr, naa
    'constrain_baseline': False,  # constrain LCModel baseline (for WAND)

    # data settings
    'test_size': 32,   # number of test samples
    'load_model': True,  # load model from path2trained
    'skip_train': True,  # skip the training procedure

    # visual settings
    'run': True,  # run the inference (will try to load results if False)
    'save': True,  # save the plots
    'error': 'mae',  # 'mae', 'mse', 'mape', ...
    'scale': 'tcr',  # 'gts', 'tcr', 'max'
    'linewidth': 5,  # linewidth for the plots
    'idx': [],  # batch index (all if empty)
    'metIdx': [],  # metabolite index (all if empty)
    'imag': False,  # plot imaginary part
    'plot_dec': False,   # plot decomposition
    'plot_conc': True,   # plot concentrations
    'plot_corrs': True,   # plot correlations
    'plot_fits': True,   # plot fits
    'plot_conc_metabs': 'both',   # 'metabs', 'total', 'both'
}



if __name__ == '__main__':

    # initialize pipeline
    pipeline = Pipeline()
    pipeline.default_config.update(config)
    pipeline.main({'online': False})

    config, model = pipeline.config, pipeline.model
    save_path = f'../Imgs/{config.dataType}/{config.note}/'

    if not isinstance(config.method, list):
        methods = [config.method]
    else:
        methods = config.method

    if config.run:
        for method in tqdm(methods):

            # lcm method
            if method.lower() == 'newton' or method.lower() == 'mh':
                lcm = FrameworkFSL(config['path2basis'], method=method,
                                   specType=config.specType, dataType=config.dataType,
                                   basisFmt=config.basisFmt, device=model.device,
                                   baseline_order=4)

                # add macromolecules
                if config.mm: lcm.basisFSL.add_default_MM_peaks()

            elif method.lower() == 'lcmodel':
                # add macromolecules and lipids
                ignore = config.lcmodel_ignore
                if config.mm: ignore += '_wmm'
                if config.lipids: ignore += '_wlip'

                lcm = FrameworkLCModel(config['path2basis'], specType=config.specType,
                                       dataType=config.dataType, basisFmt=config.basisFmt,
                                       ignore=ignore, device=model.device)
            else:
                raise ValueError('Method not recognized.')

            # remove mean from basis
            lcm.basisObj.basisFSL._raw_fids -= lcm.basisObj.basisFSL._raw_fids.mean(-2, keepdims=True)

            # test data
            if config.dataType == 'cha':
                dataloader = ChallengeDataModule(basis_dir=config.path2basis,
                                                 nums_cha=config.test_size)
            elif config.dataType == 'invivo':
                dataloader = InVivoDataModule(data_dir='../Data/DataSets/InVivo/volunteer1/SPARSDAT/',
                                              # data_dir='../Data/DataSets/fMRSinPain/SUBJECTS/BASELINE/',
                                              basis_dir=config.path2basis,
                                              nums_test=config.test_size)
            elif config.dataType == 'invivo_ua':
                dataloader = InVivoNSAModule(# data_dir='../Data/DataSets/InVivo/volunteer2/listdata/NSA128/',
                                             data_dir='../Data/DataSets/fMRSinPain/SUBJECTS/RAW/',
                                             basis_dir=config.path2basis,
                                             basis_fmt=config.basisFmt,
                                             nums_test=config.test_size,
                                             fMRS=False)
            else:
                dataloader = SynthDataModule(basis_dir=config.path2basis,
                                             nums_test=config.test_size,
                                             sigModel=model.sigModel,
                                             params=model.ps,
                                             concs=model.concs,
                                             basisFmt=config.basisFmt,
                                             specType=config.specType)

            data = next(iter(dataloader.test_dataloader()))[:config.test_size]
            if isinstance(data, list): x, y, t = data
            else: x, y, t = data, None, None
            model = model.to(x.device)

            if config.dataType == 'invivo_ua':

                # to nifti for processing.py
                x_proc = (x[:, 0] + 1j * x[:, 1]).detach().cpu().numpy()
                x_proc = np.fft.ifft(x_proc, axis=-3)
                x_proc = gen_nifti_mrs(data=x_proc.reshape(x_proc.shape[0:1] + (1, 1,) + x_proc.shape[1:]),
                                       dwelltime=lcm.basisFSL.original_dwell,
                                       spec_freq=lcm.basisFSL.cf,
                                       nucleus='1H',
                                       dim_tags=['DIM_COIL', 'DIM_DYN', None],
                                       no_conj=False)

                x_ref = next(iter(dataloader.ref_dataloader()))[:config.test_size]
                x_ref = (x_ref[:, 0] + 1j * x_ref[:, 1]).detach().cpu().numpy()
                x_ref = np.fft.ifft(x_ref, axis=-3)
                x_ref = gen_nifti_mrs(data=x_ref.reshape(x_ref.shape[0:1] + (1, 1,) + x_ref.shape[1:]),
                                      dwelltime=lcm.basisFSL.original_dwell,
                                      spec_freq=lcm.basisFSL.cf,
                                      nucleus='1H',
                                      dim_tags=['DIM_COIL', 'DIM_DYN', None],
                                      no_conj=False)

                # coilcombine
                if config.specType == 'fMRSinPain': x_proc = proc.coilcombine(x_proc, no_prewhiten=True)   # coil combine
                else: x_proc = proc.coilcombine(x_proc)   # coil combine
                x_proc = proc.align(x_proc, 'DIM_DYN', ppmlim=(0, 8))   # align phases

                if config.specType == 'fMRSinPain': x_ref = proc.coilcombine(x_ref, no_prewhiten=True)   # coil combine
                else: x_ref = proc.coilcombine(x_ref)   # coil combine
                x_ref = proc.align(x_ref, 'DIM_DYN', ppmlim=(0, 8))   # align phases

                # average
                x = proc.average(x_proc.copy(), 'DIM_DYN')
                x_ref = proc.average(x_ref, 'DIM_DYN')

                # process
                x = proc.ecc(x, x_ref.copy())   # eddy current correction
                # x = own_nifti_ecc(x, x_ref.copy())   # eddy current correction
                x_proc = proc.remove_peaks(x_proc, [-0.15, 0.15], limit_units='ppm')  # remove residual water
                if config.shift_ref.lower() == 'cr':
                    x = proc.shift_to_reference(x, 3.027, (2.9, 3.1))  # shift to reference
                    x = proc.phase_correct(x, (2.9, 3.1))   # phase correction
                elif config.shift_ref.lower() == 'naa':
                    x = proc.shift_to_reference(x, 2.02, (1.9, 2.1))
                    x = proc.phase_correct(x, (1.9, 2.1))

                print(x.hdr_ext)

                # back to initial spectral from
                x = np.fft.fft(x[:, 0, 0], axis=-1)
                x = torch.stack((torch.tensor(x.real), torch.tensor(x.imag)), dim=1).to(model.device)

            elif config.dataType == 'invivo':

                # to nifti for processing.py
                x_proc = (x[:, 0] + 1j * x[:, 1]).detach().cpu().numpy()
                x_proc = np.fft.ifft(x_proc, axis=-1)
                x_proc = gen_nifti_mrs(data=x_proc.reshape(x_proc.shape[0:1] + (1, 1,) + x_proc.shape[1:]),
                                        dwelltime=lcm.basisFSL.original_dwell,
                                        spec_freq=lcm.basisFSL.cf,
                                        nucleus='1H',
                                        dim_tags=[None, None, None],
                                        no_conj=False)

                # process
                x_proc = proc.ecc(x_proc, x_ref.copy())   # eddy current correction
                # x_proc = own_nifti_ecc(x_proc, x_ref.copy())   # eddy current correction
                x_proc = proc.remove_peaks(x_proc, [-0.15, 0.15], limit_units='ppm')  # remove residual water
                if config.shift_ref.lower() == 'cr':
                    x_proc = proc.shift_to_reference(x_proc, 3.027, (2.9, 3.1))  # shift to reference
                    x_proc = proc.phase_correct(x_proc, (2.9, 3.1))  # phase correction
                elif config.shift_ref.lower() == 'naa':
                    x_proc = proc.shift_to_reference(x_proc, 2.02, (1.9, 2.1))
                    x_proc = proc.phase_correct(x_proc, (1.9, 2.1))

                # back to initial spectral from
                x = np.fft.fft(x_proc[:, 0, 0], axis=-1)
                x = torch.stack((torch.tensor(x.real), torch.tensor(x.imag)), dim=1).to(model.device)

            # inference
            xn, yn, t = model.prepareData(x.clone(), y, t)
            y_hat_s, ms = model.forward(xn)
            cwtmatr, masks = ms

            # pad back to original length
            y_hat = torch.nn.functional.pad(y_hat_s, (0, 0, model.first, model.basisObj.n - model.last),
                                            mode='replicate')

            # cut and pad back for other signals to remove issues with boundary regions
            xs = xn[:, :, model.first:model.last:model.skip]
            xl = torch.nn.functional.pad(xs, (model.first, model.basisObj.n - model.last), mode='replicate')

            # get the normal lcm quantification
            if method.lower() == 'lcmodel': lcm.control[1] = 'nunfil=2048'
            lcm.save_path = f'{save_path}quant_{method}/lcm{x_proc.shape[4]}/'
            t_lcm, u_lcm = lcm.forward(x)
            if method.lower() == 'lcmodel': lcm.control[1] = 'nunfil=1536'

            lcm.save_path = f'{save_path}quant_{method}/lcm_l{x_proc.shape[4]}/'
            t_lcm_l, u_lcm_l = lcm.forward(xl)

            # get lcm quantification without artifact signal
            xc = xl - y_hat[..., -1]  # remove artifact signal
            lcm.save_path = f'{save_path}quant_{method}/lcm_c{x_proc.shape[4]}/'
            if config.constrain_baseline and method.lower() == 'lcmodel': lcm.control.insert(-2, 'dkntmn=1.5')
            t_lcm_c, u_lcm_c = lcm.forward(xc)
            if config.constrain_baseline and method.lower() == 'lcmodel': lcm.control.pop(-2)

            # visualize specs
            linewidth = config.linewidth
            idx, metIdx, imag = config.idx, config.metIdx, config.imag

            if config.plot_dec:   # plot/save the scalograms, masks, and metabolite spectra
                visualize_pred(xs, cwtmatr, masks, y_hat_s, model.names, imag=imag, batch_idx=idx, metab_idx=metIdx,
                               save_path=f'{save_path}x{"Real" if not imag else "Imag"}/')

                # plot decomposition in one figure
                plot_decomposition(xs, y_hat_s, model.names, imag=imag, batch_idx=idx, metab_idx=metIdx, linewidth=2,
                                   save_path=f'{save_path}d{"Real" if not imag else "Imag"}/', yLim=False)

            concs = {x_proc.shape[-1]: t_lcm[:, :lcm.basisObj.n_metabs]}
            concs_l = {x_proc.shape[-1]: t_lcm_l[:, :lcm.basisObj.n_metabs]}
            concs_c = {x_proc.shape[-1]: t_lcm_c[:, :lcm.basisObj.n_metabs]}

            uncs = {x_proc.shape[-1]: u_lcm[:, :lcm.basisObj.n_metabs]}
            uncs_l = {x_proc.shape[-1]: u_lcm_l[:, :lcm.basisObj.n_metabs]}
            uncs_c = {x_proc.shape[-1]: u_lcm_c[:, :lcm.basisObj.n_metabs]}

            # run with less and less averages to see model performance with low-quality data
            if config.dataType == 'invivo_ua':
                for i in tqdm(range(1, x_proc.shape[4])):
                    if i % config.averages == 0 and config.averages > 0 or \
                       x_proc.shape[4] - i in [1, 2, 4, 8, 16, 32, 64, 128] and config.averages < 0:
                        _, x = split(x_proc, 'DIM_DYN', i-1)   # split on dynamics

                        # average
                        if x.shape[4] > 1: x = proc.average(x, 'DIM_DYN')

                        # process
                        x = proc.ecc(x, x_ref.copy())  # eddy current correction
                        # x = own_nifti_ecc(x, x_ref.copy())  # eddy current correction
                        x = proc.remove_peaks(x, [-0.15, 0.15], limit_units='ppm')  # remove residual water
                        if config.shift_ref.lower() == 'cr':
                            x = proc.shift_to_reference(x, 3.027, (2.9, 3.1))  # shift to reference
                            x = proc.phase_correct(x, (2.9, 3.1))  # phase correction
                        elif config.shift_ref.lower() == 'naa':
                            x = proc.shift_to_reference(x, 2.02, (1.9, 2.1))
                            x = proc.phase_correct(x, (1.9, 2.1))

                        # back to initial spectral from
                        x = np.fft.fft(x[:, 0, 0], axis=-1)
                        x = torch.stack((torch.tensor(x.real), torch.tensor(x.imag)), dim=1).to(model.device)

                        # inference
                        xn, yn, t = model.prepareData(x.clone(), y, t)
                        y_hat_s, ms = model.forward(xn)
                        cwtmatr, masks = ms

                        # pad back to original length
                        y_hat = torch.nn.functional.pad(y_hat_s, (0, 0, model.first, model.basisObj.n - model.last),
                                                        mode='replicate')

                        # cut and pad back for other signals to remove issues with boundary regions
                        xs = xn[:, :, model.first:model.last:model.skip]
                        xl = torch.nn.functional.pad(xs, (model.first, model.basisObj.n - model.last), mode='replicate')

                        # get the normal lcm quantification
                        if method.lower() == 'lcmodel': lcm.control[1] = 'nunfil=2048'
                        lcm.save_path = f'{save_path}quant_{method}/lcm{x_proc.shape[4] - i}/'
                        t_lcm, u_lcm = lcm.forward(x)
                        if method.lower() == 'lcmodel': lcm.control[1] = 'nunfil=1536'

                        lcm.save_path = f'{save_path}quant_{method}/lcm_l{x_proc.shape[4] - i}/'
                        t_lcm_l, u_lcm_l = lcm.forward(xl)

                        # get lcm quantification without artifact signal
                        xc = xl - y_hat[..., -1]  # remove artifact signal
                        lcm.save_path = f'{save_path}quant_{method}/lcm_c{x_proc.shape[4] - i}/'
                        t_lcm_c, u_lcm_c = lcm.forward(xc)

                        # concs
                        concs[x_proc.shape[4] - i] = t_lcm[:, :lcm.basisObj.n_metabs]
                        concs_l[x_proc.shape[4] - i] = t_lcm_l[:, :lcm.basisObj.n_metabs]
                        concs_c[x_proc.shape[4] - i] = t_lcm_c[:, :lcm.basisObj.n_metabs]

                        # uncs
                        uncs[x_proc.shape[4] - i] = u_lcm[:, :lcm.basisObj.n_metabs]
                        uncs_l[x_proc.shape[4] - i] = u_lcm_l[:, :lcm.basisObj.n_metabs]
                        uncs_c[x_proc.shape[4] - i] = u_lcm_c[:, :lcm.basisObj.n_metabs]

                        if config.plot_dec:   # visualize specs
                            visualize_pred(xs, cwtmatr, masks, y_hat_s, model.names, imag=imag, batch_idx=idx, metab_idx=metIdx,
                                           save_path=f'{save_path}x{f"Real{x_proc.shape[4] - i}" if not imag else f"Imag{x_proc.shape[4] - i}"}/')

                            # plot decomposition in one figure
                            plot_decomposition(xs, y_hat_s, model.names, imag=imag, batch_idx=idx, metab_idx=metIdx, linewidth=2, yLim=False,
                                               save_path=f'{save_path}d{f"Real{x_proc.shape[4] - i}" if not imag else f"Imag{x_proc.shape[4] - i}"}/')

            # save estimated concentrations
            torch.save(concs, f'{save_path}{method.lower()}_concs.pth')
            torch.save(concs_l, f'{save_path}{method.lower()}_concs_l.pth')
            torch.save(concs_c, f'{save_path}{method.lower()}_concs_c.pth')

            # save estimated uncertainties
            torch.save(uncs, f'{save_path}{method.lower()}_uncs.pth')
            torch.save(uncs_l, f'{save_path}{method.lower()}_uncs_l.pth')
            torch.save(uncs_c, f'{save_path}{method.lower()}_uncs_c.pth')

        # check if folder exists and create json file of config
        if os.path.exists(save_path):
            with open(f'{save_path}config.json', 'w') as f:
                f.write(str(config))


    # testing (load and visualize)
    if not isinstance(config.method, list): methods = [config.method]
    else: methods = config.method

    # get metab names
    tot_names = list(computeTotalMetabolites(torch.zeros(1, model.basisObj.n_metabs), model.basisObj.names).keys())

    colors = {'Newton': ['royalblue', 'mediumpurple'],
              'LCModel': ['darkslategray', 'turquoise'],
              'MH': ['brown', 'coral']}
    color_list = []
    for m in methods: color_list.extend(colors[m])

    concs, concs_l, concs_c = {}, {}, {}
    tot_concs, tot_concs_l, tot_concs_c = {}, {}, {}

    # load concentrations
    for method in methods:
        concs[method] = torch.load(f'{save_path}{method.lower()}_concs.pth')
        concs_l[method] = torch.load(f'{save_path}{method.lower()}_concs_l.pth')
        concs_c[method] = torch.load(f'{save_path}{method.lower()}_concs_c.pth')

        # compute total concentrations
        tot_concs[method] = {k: torch.stack(list(computeTotalMetabolites(v, model.basisObj.names).values()), dim=1)
                             for k, v in concs[method].items()}
        tot_concs_l[method] = {k: torch.stack(list(computeTotalMetabolites(v, model.basisObj.names).values()), dim=1)
                               for k, v in concs_l[method].items()}
        tot_concs_c[method] = {k: torch.stack(list(computeTotalMetabolites(v, model.basisObj.names).values()), dim=1)
                               for k, v in concs_c[method].items()}

    # ground truth concentrations
    if config.specType == 'fMRSinPain':
        path2conc = '../Data/DataSets/fMRSinPain/GLUPI/GLUPI/'
        gt_concs, gt_uncs, tot_gt_concs, tot_gt_uncs = [], [], [], []
        fwhms, snrs, shifts, phases = [], [], [], []
        for i, sub in enumerate(os.listdir(path2conc)):
            if i == config.test_size: break
            path = f'{path2conc}{sub}/bsl_32s/{sub}_bsl_32.table'
            if os.path.exists(path):
                from loading.lcmodel import read_LCModel_coord
                mets, cons, crlbs, tcr, fwhm, snr, shift, phase = read_LCModel_coord(path)

                if config.scale == 'gts': tcr = cons

                m_tcr = {
                    'Ace': 0.0,
                    'Ala': tcr[mets.index('Ala')],
                    'Asc': 0.0,
                    'Asp': tcr[mets.index('Asp')],
                    'Cr': tcr[mets.index('Cr')],
                    'GABA': tcr[mets.index('GABA')],
                    'Glc': tcr[mets.index('Glc')],
                    'Gln': tcr[mets.index('Gln')],
                    'Glu': tcr[mets.index('Glu')],
                    'Gly': 0.0,
                    'GPC': tcr[mets.index('GPC')],
                    'GSH': tcr[mets.index('GSH')],
                    'mI': tcr[mets.index('Ins')],
                    'Lac': tcr[mets.index('Lac')],
                    'Mac': tcr[mets.index('MM09')] + tcr[mets.index('MM12')] +
                           tcr[mets.index('MM14')] + tcr[mets.index('MM17')],
                    'NAA': tcr[mets.index('NAA')],
                    'NAAG': tcr[mets.index('NAAG')],
                    'PCh': tcr[mets.index('PCh')],
                    'PCr': tcr[mets.index('PCr')],
                    'PE': 0.0,
                    'sI': tcr[mets.index('Scyllo')],
                    'Tau': tcr[mets.index('Tau')],
                }

                # sort
                m_tcr = {k: m_tcr[k] for k in model.basisObj.names[:model.basisObj.n_metabs]}

                tot_tcr = {
                    'NAA+NAAG': tcr[mets.index('NAA+NAAG')],
                    'Cr+PCr': tcr[mets.index('Cr+PCr')],
                    'Gln+Glu': tcr[mets.index('Glu+Gln')],
                    'Ins+Gly': tcr[mets.index('Ins')] + tcr[mets.index('Scyllo')],
                    'GPC+PCho': tcr[mets.index('GPC+PCh')],
                    'GABA': tcr[mets.index('GABA')],
                    'GSH': tcr[mets.index('GSH')],
                }

                gt_concs.append(m_tcr)
                tot_gt_concs.append(tot_tcr)
                fwhms.append(fwhm)
                snrs.append(snr)
                shifts.append(shift)
                phases.append(phase)

        print(f'FWHM: {np.min(fwhms):.2f} - {np.max(fwhms):.2f}')
        print(f'SNR: {np.min(snrs):.2f} - {np.max(snrs):.2f}')
        print(f'Shift: {np.min(shifts):.2f} - {np.max(shifts):.2f}')

        gt_concs = {32: torch.tensor([list(gt_con.values()) for gt_con in gt_concs])}
        tot_gt_concs = {32: torch.tensor([list(tot_gt_con.values()) for tot_gt_con in tot_gt_concs])}

    else:
        gt_concs = None
        tot_gt_concs = {'NAA+NAAG': 1.3685, 'Cr+PCr': 1.0, 'Gln+Glu': 1.4357, 'Ins+Gly': 0.9883,
                        'GPC+PCho': 0.27961, 'GABA': 0.24077341, 'GSH': 0.423337856}
        tot_gt_std = {'NAA+NAAG': 0.14956, 'Cr+PCr': 0.0, 'Gln+Glu': 0.295167, 'Ins+Gly': 0.1375,
                      'GPC+PCho': 0.032389, 'GABA': 0.04409769, 'GSH': 0.07717096}

    def scaleTo(concs, type='gts', scale=None):
        if type.lower() == 'gts':
            return {k: concs[k] * model.optimalReference(scale[list(gt_concs.keys())[0]], concs[k])
                    for k in concs.keys()}
        elif type.lower() == 'tcr':
            return {k: concs[k] / scale[k][:, tot_names.index('Cr+PCr')].unsqueeze(1) for k in concs.keys()}
        elif type.lower() == 'max':
            return {k: concs[k] / concs[k].max(1, keepdim=True).values for k in concs.keys()}

    if config.scale == 'gts':  # scale to ground truth concentrations
        for method in methods:
            concs[method] = scaleTo(concs[method], 'gts', gt_concs)
            concs_l[method] = scaleTo(concs_l[method], 'gts', gt_concs)
            concs_c[method] = scaleTo(concs_c[method], 'gts', gt_concs)

            tot_concs[method] = scaleTo(tot_concs[method], 'gts', tot_gt_concs)
            tot_concs_l[method] = scaleTo(tot_concs_l[method], 'gts', tot_gt_concs)
            tot_concs_c[method] = scaleTo(tot_concs_c[method], 'gts', tot_gt_concs)

    elif config.scale == 'tcr':  # scale to total creatine
        for method in methods:
            concs[method] = scaleTo(concs[method], 'tcr', tot_concs[method])
            concs_l[method] = scaleTo(concs_l[method], 'tcr', tot_concs_l[method])
            concs_c[method] = scaleTo(concs_c[method], 'tcr', tot_concs_c[method])

            tot_concs[method] = scaleTo(tot_concs[method], 'tcr', tot_concs[method])
            tot_concs_l[method] = scaleTo(tot_concs_l[method], 'tcr', tot_concs_l[method])
            tot_concs_c[method] = scaleTo(tot_concs_c[method], 'tcr', tot_concs_c[method])


    # plot concentrations
    if config.plot_conc:

        # bar plot
        def plot_bar(gt_concs, concs, concs_l, concs_c, names, k=None, idx=None, total=False):

            if idx is None: idx = range(concs[methods[0]][list(concs[methods[0]].keys())[0]].shape[0])
            elif isinstance(idx, int): idx = [idx]

            if k is None: k = list(concs[methods[0]].keys())[0]

            algs = {}

            # if total: algs.update({f'Literature': np.array(list(gt_concs.values()))[np.newaxis, :]})
            algs.update({f'Literature': gt_concs[list(gt_concs.keys())[0]][idx].detach().cpu().numpy()})
            algs.update({m: concs[m][k][idx].detach().cpu().numpy() for m in methods})
            algs.update({f'{m} + L': concs_l[m][k][idx].detach().cpu().numpy() for m in methods})
            algs.update({f'{m} + WAND': concs_c[m][k][idx].detach().cpu().numpy() for m in methods})
            plot_algs_bars(algs, names)

            # if total:
            #     plt.errorbar(np.arange(len(names)), list(tot_gt_concs.values()), yerr=list(tot_gt_std.values()),
            #                  fmt='none', ecolor='k', capsize=2,
            #                  elinewidth=1, capthick=1, alpha=0.8, zorder=3)

            if len(idx) == 1:
                if config.save:
                    if not os.path.exists(f'{save_path}/concs/concs_{idx[0]}/'): os.makedirs(f'{save_path}/concs/concs_{idx[0]}/')
                    plt.savefig(f'{save_path}/concs/concs_{idx[0]}/concs_{k}_t={total}.png', dpi=300)
                else: plt.show()

            if config.save: plt.savefig(f'{save_path}concs_{k}_t={total}.png', dpi=300)
            else: plt.show()

        def plot_concs(gt_concs, concs, concs_l, concs_c, names, idx=None):

            if idx is None: idx = range(concs[methods[0]][list(concs[methods[0]].keys())[0]].shape[0])
            elif isinstance(idx, int): idx = [idx]

            # plot for metabolites
            for i, m in enumerate(names):
                # error plot

                if m == 'NAA+NAAG': plt.figure(figsize=(2 * 4.75, 3.5))
                else: plt.figure(figsize=(4.75, 2.5))

                # plt.hlines(os_concs[:, i].mean(), 0, list(concs[list(concs.keys())[0]].keys())[0],
                #            colors='k', linestyles='dashed', label='Osprey')

                for method in methods:

                    if method.lower() == 'newton' or method.lower() == 'mh': m_name = 'FSL-MRS'
                    else: m_name = method

                    data = torch.stack([concs[method][k][idx, i] for k in concs[method].keys()], dim=0).detach().cpu().numpy()
                    data_l = torch.stack([concs_l[method][k][idx, i] for k in concs_l[method].keys()], dim=0).detach().cpu().numpy()
                    data_c = torch.stack([concs_c[method][k][idx, i] for k in concs_c[method].keys()], dim=0).detach().cpu().numpy()

                    plt.errorbar(concs[method].keys(), data.mean(1), linestyle='--', marker='o', yerr=data.std(1),
                                 color=colors[method][0], label=f'{m_name}', alpha=0.8)
                    # plt.errorbar(concs_l[method].keys(), data_l.mean(1), linestyle='--', marker='o', yerr=data_l.std(1),
                    #              color=colors[method][0], label=f'{m_name}', alpha=0.8)
                    plt.errorbar(concs_c[method].keys(), data_c.mean(1), linestyle='-.', marker='^', yerr=data_c.std(1),
                                 color=colors[method][1], label=f'{m_name} + WAND', alpha=0.8)

                # add dashed line for ground truth
                plt.hlines(gt_concs[list(gt_concs.keys())[0]][idx, i].mean(0), 0, list(concs[list(concs.keys())[0]].keys())[0],
                           colors='k', linestyles='dashed', label='Reported')

                # # add shaded region for ground truth std
                # plt.fill_between(concs.keys(), gt_concs[list(gt_concs.keys())[0]][idx, i].mean(0) - gt_concs[list(gt_concs.keys())[0]][idx, i].std(0),
                #                     gt_concs[list(gt_concs.keys())[0]][idx, i].mean(0) + gt_concs[list(gt_concs.keys())[0]][idx, i].std(0),
                #                     color='k', alpha=0.2)

                plt.title(f'{m}')
                plt.xlabel('Averages')
                plt.ylabel('Concentration [/tCr]')
                if m == 'NAA+NAAG':
                    plt.legend(frameon=False, loc='upper right')
                    plt.ylim(np.min(np.concatenate([data, data_l, data_c])) - 0.1,
                             np.max(np.concatenate([data, data_l, data_c])) + 0.55)

                if config.averages < 0: plt.xscale('log', base=2)

                # change xticks and fromat to integers
                plt.xticks(list(concs[list(concs.keys())[0]].keys()), list(concs[list(concs.keys())[0]].keys()))

                if len(idx) == 1:
                    if config.save:
                        if not os.path.exists(f'{save_path}/concs/concs_{idx[0]}/'): os.makedirs(f'{save_path}/concs/concs_{idx[0]}/')
                        plt.savefig(f'{save_path}/concs/concs_{idx[0]}/concs_{m}.png', dpi=1000, bbox_inches='tight', transparent=True)
                    else: plt.show()
                else:
                    if config.save:
                        if not os.path.exists(f'{save_path}/concs/'): os.makedirs(f'{save_path}/concs/concs/')
                        plt.savefig(f'{save_path}/concs/concs_{m}.png', dpi=1000, bbox_inches='tight', transparent=True)
                    else: plt.show()

        # what to plot (metabolites or total metabolites)
        if config.plot_conc_metabs == 'metabs' or config.plot_conc_metabs == 'both':
            plot_bar(gt_concs, concs, concs_l, concs_c, model.basisObj.names[:model.basisObj.n_metabs])
            plot_concs(gt_concs, concs, concs_l, concs_c, model.basisObj.names[:model.basisObj.n_metabs])
        if config.plot_conc_metabs == 'total' or config.plot_conc_metabs == 'both':
            plot_bar(tot_gt_concs, tot_concs, tot_concs_l, tot_concs_c, tot_names, total=True)
            plot_concs(tot_gt_concs, tot_concs, tot_concs_l, tot_concs_c, tot_names)

        for i in range(concs[methods[0]][list(concs[methods[0]].keys())[0]].shape[0]):
            if config.plot_conc_metabs == 'metabs' or config.plot_conc_metabs == 'both':
                plot_bar(gt_concs, concs, concs_l, concs_c, model.basisObj.names[:model.basisObj.n_metabs], idx=i)
                plot_concs(gt_concs, concs, concs_l, concs_c, model.basisObj.names[:model.basisObj.n_metabs], idx=i)
            if config.plot_conc_metabs == 'total' or config.plot_conc_metabs == 'both':
                plot_bar(tot_gt_concs, tot_concs, tot_concs_l, tot_concs_c, tot_names, idx=i, total=True)
                plot_concs(tot_gt_concs, tot_concs, tot_concs_l, tot_concs_c, tot_names, idx=i)


    # plot correlations for all samples and averages
    if config.plot_corrs:

        def compute_metrics(concs, gt_concs, metric='ccc'):
            if concs.shape[1] == 7:   # remove GABA and GSH
                concs = concs[:, :5]
                gt_concs = gt_concs[:, :5]
            # elif concs.shape[1] == 22:
            #     # filterIdx = [4, 7, 8, 12, 15, 16, 17, 18, 20]
            #     filterIdx = [1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21]
            #     concs = concs[:, filterIdx]
            #     gt_concs = gt_concs[:, filterIdx]

            if metric.lower() == 'ccc':
                from torchmetrics.regression import ConcordanceCorrCoef
                ccc =  ConcordanceCorrCoef()
                return np.array([ccc(concs[k], gt_concs[k]).detach().cpu().numpy() for k in range(concs.shape[0])])

            elif metric.lower() == 'mae':
                from torchmetrics.regression import MeanAbsoluteError
                mae = MeanAbsoluteError()
                return np.array([mae(concs[k], gt_concs[k]).detach().cpu().numpy() for k in range(concs.shape[0])])

            elif metric.lower() == 'mape':
                from torchmetrics.regression import MeanAbsolutePercentageError
                mape = MeanAbsolutePercentageError()
                return np.array([mape(concs[k], gt_concs[k]).detach().cpu().numpy() for k in range(concs.shape[0])])

            elif metric.lower() == 'pe':
                return model.concsLoss(gt_concs, concs, 'pe').detach().cpu().numpy()

            else: raise ValueError(f'Unknown metric: {metric}')

        # compute correlations
        corrs, corrs_l, corrs_c = {}, {}, {}
        tot_corrs, tot_corrs_l, tot_corrs_c = {}, {}, {}

        maes, maes_l, maes_c = {}, {}, {}
        tot_maes, tot_maes_l, tot_maes_c = {}, {}, {}

        for method in methods:
            corrs[method] = {k: compute_metrics(concs[method][k], gt_concs[list(gt_concs.keys())[0]])
                             for k in concs[method].keys()}
            corrs_l[method] = {k: compute_metrics(concs_l[method][k], gt_concs[list(gt_concs.keys())[0]])
                               for k in concs_l[method].keys()}
            corrs_c[method] = {k: compute_metrics(concs_c[method][k], gt_concs[list(gt_concs.keys())[0]])
                               for k in concs_c[method].keys()}

            tot_corrs[method] = {k: compute_metrics(tot_concs[method][k], tot_gt_concs[list(tot_gt_concs.keys())[0]])
                                 for k in tot_concs[method].keys()}
            tot_corrs_l[method] = {k: compute_metrics(tot_concs_l[method][k], tot_gt_concs[list(tot_gt_concs.keys())[0]])
                                   for k in tot_concs_l[method].keys()}
            tot_corrs_c[method] = {k: compute_metrics(tot_concs_c[method][k], tot_gt_concs[list(tot_gt_concs.keys())[0]])
                                   for k in tot_concs_c[method].keys()}

            maes[method] = {k: compute_metrics(concs[method][k], gt_concs[list(gt_concs.keys())[0]], metric='mae')
                            for k in concs[method].keys()}
            maes_l[method] = {k: compute_metrics(concs_l[method][k], gt_concs[list(gt_concs.keys())[0]], metric='mae')
                              for k in concs_l[method].keys()}
            maes_c[method] = {k: compute_metrics(concs_c[method][k], gt_concs[list(gt_concs.keys())[0]], metric='mae')
                              for k in concs_c[method].keys()}

            tot_maes[method] = {k: compute_metrics(tot_concs[method][k], tot_gt_concs[list(tot_gt_concs.keys())[0]], metric='mae')
                                for k in tot_concs[method].keys()}
            tot_maes_l[method] = {k: compute_metrics(tot_concs_l[method][k], tot_gt_concs[list(tot_gt_concs.keys())[0]], metric='mae')
                                  for k in tot_concs_l[method].keys()}
            tot_maes_c[method] = {k: compute_metrics(tot_concs_c[method][k], tot_gt_concs[list(tot_gt_concs.keys())[0]], metric='mae')
                                  for k in tot_concs_c[method].keys()}


        def plot_correlations(corrs, title, total=False):

            if 'mae' in title.lower():
                cmap = 'Reds'
                lim1, lim2 = 0.0, 0.3
            else:
                cmap = 'Blues'
                lim1, lim2 = 0.75, 1.0

            corrs_np = np.array(list(corrs.values()))

            plt.figure(figsize=(4, 3))
            plt.imshow(corrs_np, cmap=cmap, aspect='auto')
            plt.xticks(range(corrs_np.shape[1]), range(1, corrs_np.shape[1] + 1))
            plt.yticks(range(len(corrs)), corrs.keys())
            if 'colorbar' in title.lower(): plt.colorbar()
            plt.title(f'{title}')
            plt.xlabel('Dataset')
            plt.ylabel('Averages')

            plt.clim(lim1, lim2)

            if config.save:
                if not os.path.exists(f'{save_path}/corrs/'): os.makedirs(f'{save_path}/corrs/')
                plt.savefig(f'{save_path}/corrs/{title if total else "Total " + title}.png',
                            dpi=1000, bbox_inches='tight', transparent=True)


        # plot correlations
        for method in methods:

            if method.lower() == 'newton' or method.lower() == 'mh': m_name = 'FSL-MRS'
            else: m_name = method

            if config.plot_conc_metabs == 'metabs' or config.plot_conc_metabs == 'both':
                plot_correlations(corrs[method], f'Concordance Correlation ({m_name})')
                # plot_correlations(corrs_l[method], f'Concordance Correlation ({m_name} + L)')
                plot_correlations(corrs_c[method], f'Concordance Correlation ({m_name} + WAND)')

                plot_correlations(maes[method], f'MAE ({m_name})')
                # plot_correlations(maes_l[method], f'MAE ({m_name} + L)')
                plot_correlations(maes_c[method], f'MAE ({m_name} + WAND)')

                # plot one with color bar
                plot_correlations(corrs[method], f'Colorbar')

            if config.plot_conc_metabs == 'total' or config.plot_conc_metabs == 'both':
                plot_correlations(tot_corrs[method], f'Concordance Correlation ({m_name})')
                # plot_correlations(tot_corrs_l[method], f'Concordance Correlation ({m_name} + L)')
                plot_correlations(tot_corrs_c[method], f'Concordance Correlation ({m_name} + WAND)')

                plot_correlations(tot_maes[method], f'MAE ({m_name})')
                # plot_correlations(tot_maes_l[method], f'MAE ({m_name} + L)')
                plot_correlations(tot_maes_c[method], f'MAE ({m_name} + WAND)')

                # plot one with color bar
                plot_correlations(tot_corrs[method], f'Colorbar', total=True)

            if not config.save: plt.show()


    # plot the LCModel fits
    if config.plot_fits and 'LCModel' in methods:

        for avg in concs['LCModel'].keys():
            for idx in range(concs['LCModel'][avg].shape[0]):
                def plot_LCModel_fit(fit, name):
                    if fit:
                        plt.figure()
                        plt.plot(fit['ppm'], fit['data'], 'k', label='Data', linewidth=1)
                        plt.plot(fit['ppm'], fit['completeFit'], 'r', label='Fit', alpha=0.6, linewidth=2)
                        plt.plot(fit['ppm'], fit['data'] - fit['completeFit'] + 1.1 * np.max(fit['data']),
                                 'k', label='Residual', alpha=0.8, linewidth=1)

                        # compute scale factor
                        # scale = xc[0][idx, 0, model.first:model.last].detach().cpu().numpy() / fit['completeFit'][model.first:model.last]
                        #
                        # art = xl[0][idx, 0, model.first:model.last].detach().cpu().numpy() - xc[0][idx, 0, model.first:model.last].detach().cpu().numpy()
                        # plt.plot(np.linspace(fit['ppm'][0], fit['ppm'][-1], art.shape[0]), art, 'b', label='Artifact', alpha=0.5, linewidth=2)

                        # plt.xlabel('Chemical Shift [ppm]')
                        plt.gca().invert_xaxis()

                        # # remove left, top, right spines
                        # plt.gca().spines['right'].set_visible(False)
                        # plt.gca().spines['top'].set_visible(False)
                        # plt.gca().spines['left'].set_visible(False)
                        #
                        # # remove y axis
                        # plt.gca().yaxis.set_visible(False)
                        # plt.gca().yaxis.set_ticks_position('none')

                        plt.axis('off')

                        if config.save:
                            if not os.path.exists(f'{save_path}lcm_fits/{avg}/'): os.makedirs(f'{save_path}lcm_fits/{avg}/')
                            plt.savefig(f'{save_path}lcm_fits/{avg}/{name}', dpi=1000, bbox_inches='tight', transparent=True)
                        else:
                            plt.show()


                fit = read_LCModel_fit(f'{save_path}quant_LCModel/lcm{int(avg)}/temp{idx}.coord')
                plot_LCModel_fit(fit, name=f'lcm_{idx}.png')

                fit = read_LCModel_fit(f'{save_path}quant_LCModel/lcm_l{int(avg)}/temp{idx}.coord')
                plot_LCModel_fit(fit, name=f'lcm_l_{idx}.png')

                fit = read_LCModel_fit(f'{save_path}quant_LCModel/lcm_c{int(avg)}/temp{idx}.coord')
                plot_LCModel_fit(fit, name=f'lcm_c_{idx}.png')


    # check if folder exists and create json file of config
    if os.path.exists(save_path):
        with open(f'{save_path}config.json', 'w') as f:
            f.write(str(config))