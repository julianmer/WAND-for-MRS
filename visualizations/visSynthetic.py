####################################################################################################
#                                          visSynthetic.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/02/24                                                                                #
#                                                                                                  #
# Purpose: Visualizes the synthetic results.                                                       #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutup; shutup.please()   # shut up warnings
import torch

from suspect.processing.denoising import *

from tqdm import tqdm

# own
from frameworks.frameworkFSL import FrameworkFSL
from frameworks.frameworkLCM import FrameworkLCModel
from loading.lcmodel import read_LCModel_fit
from simulation.dataModules import SynthDataModule, ChallengeDataModule
from train import Pipeline
from utils.auxiliary import challenge_scenario_indices, computeTotalMetabolites
from visualizations.plotFunctions import *



#****************#
#   initialize   #
#****************#
config = {
    # path to a trained model
    'path2trained': './lightning_logs/sskt2pvt/checkpoints/epoch=0-step=1195264.ckpt',

    # path to basis set
    'path2basis': '../Data/BasisSets/ISMRM_challenge/basisset_JMRUI/',  # path to basis set

    # for nn models
    'arch': 'unet',   # 'unet', ...
    'specType': 'synth',   # 'auto', 'synth', 'invivo', 'biggaba', 'fMRSinPain'
    'basisFmt': '',  # '', 'cha_philips', 'biggaba', 'fMRSinPain'

    # for lcm model
    'method': ['Newton', 'LCModel'],   # 'Newton', 'MH', 'LCModel'
    'ppmlim': (0.5, 4.2),  # ppm limits for the spectra

    # data settings
    'dataType': 'norm_rw_p',   # 'cha', 'clean', 'norm', 'norm_rw', 'norm_rw_p', 'custom', ...
    'note': 'norm_rw_p',   # additional note

    'test_size': 100,   # number of test samples
    'load_model': True,   # load model from path2trained
    'skip_train': True,   # skip the training procedure

    # visual settings
    'run': True,   # run the inference (will try to load results if False)
    'save': True,   # save the plots
    'denoising': True,   # run denoising benchmark methods
    'relative': False,   # relative concentrations (/tCr)
    'error': 'mae',   # 'mae', 'mse', 'mape', ...
    'save_fit': True,   # save fit results
    'plot_ind': 32,   # plot individual samples
}



if __name__ == '__main__':

    # initialize pipeline
    pipeline = Pipeline()
    pipeline.default_config.update(config)
    pipeline.main({'online': False})

    config, model = pipeline.config, pipeline.model
    save_path = f'../Imgs/{config.dataType}/{config.note}/'

    if not isinstance(config.method, list): methods = [config.method]
    else: methods = config.method

    if config.run:
        for method in tqdm(methods):

            # lcm method
            if method.lower() == 'newton' or method.lower() == 'mh':
                lcm = FrameworkFSL(config['path2basis'], method=method,
                                   specType=config.specType, dataType=config.dataType,
                                   basisFmt=config.basisFmt, device=model.device)
            elif method.lower() == 'lcmodel':
                lcm = FrameworkLCModel(config['path2basis'], specType=config.specType,
                                       dataType=config.dataType, basisFmt=config.basisFmt,
                                       device=model.device)
            else:
                raise ValueError('Method not recognized.')

            # remove mean from basis
            lcm.basisFSL._raw_fids -= lcm.basisFSL._raw_fids.mean(-2, keepdims=True)

            # test data
            if config.dataType[:3] == 'cha':
                dataloader = ChallengeDataModule(basis_dir=config.path2basis,
                                                 nums_cha=config.test_size)
            else:
                dataloader = SynthDataModule(basis_dir=config.path2basis,
                                             nums_test=config.test_size,
                                             sigModel=model.sigModel,
                                             params=model.ps,
                                             concs=model.concs,
                                             basisFmt=config.basisFmt,
                                             specType=config.specType)

            data = next(iter(dataloader.test_dataloader()))
            x, y, t = data
            x, y, t = x.to(model.device), y.to(model.device), t.to(model.device)

            # inference
            xn, yn, t = model.prepareData(x.clone(), y.clone(), t.clone())

            if xn.shape[0] > 32:   # memory constraints (TODO: find leak)
                model = model.to('cpu')
                y_hat_s, (cwtmatr, masks) = model.forward(xn.to('cpu'))
                y_hat_s, (cwtmatr, masks) = y_hat_s.to(x.device), (cwtmatr.to(x.device), masks.to(x.device))
            else:
                y_hat_s, (cwtmatr, masks) = model.forward(xn)

            # pad back to original length
            y_hat = torch.nn.functional.pad(y_hat_s, (0, 0, model.first, model.basisObj.n - model.last),
                                            mode='replicate')

            # cut and pad back for other signals to remove issues with boundary regions
            xs = xn[:, :, model.first:model.last:model.skip]
            xl = torch.nn.functional.pad(xs, (model.first, model.basisObj.n - model.last), mode='replicate')

            if config.dataType[:6] == 'invivo':
                ys = None
            else:
                ys = yn[:, :, model.first:model.last:model.skip]

            # get the normal lcm quantification
            if config.save_fit: lcm.save_path = f'{save_path}quant_{method.lower()}/lcm_l/'
            t_lcm_l, u_lcm_l = lcm.forward(xl)

            # get lcm quantification without artifact signal
            xc = xl - y_hat[..., -1]   # remove artifact signal
            if config.save_fit: lcm.save_path = f'{save_path}quant_{method.lower()}/lcm_c/'
            t_lcm_c, u_lcm_c = lcm.forward(xc)

            if config.denoising:   # get lcm quantification with denoising
                xd = np.array([sliding_gaussian((xn[i, 0] + 1j * xn[i, 1]).detach().cpu().numpy(), window_width=10)
                               for i in range(xn.shape[0])])
                xd1 = torch.from_numpy(np.stack((xd.real, xd.imag), axis=1))
                if config.save_fit: lcm.save_path = f'{save_path}quant_{method.lower()}/lcm_d1/'
                t_lcm_d1, u_lcm_d1 = lcm.forward(xd1)

                xd = np.array([sift((xn[i, 0] + 1j * xn[i, 1]).detach().cpu().numpy(), threshold=1000)
                               for i in range(xn.shape[0])])
                xd2 = torch.from_numpy(np.stack((xd.real, xd.imag), axis=1))
                if config.save_fit: lcm.save_path = f'{save_path}quant_{method.lower()}/lcm_d2/'
                t_lcm_d2, u_lcm_d2 = lcm.forward(xd2)

                xd = np.array([svd((xn[i, 0] + 1j * xn[i, 1]).detach().cpu().numpy(), rank=150)
                               for i in range(xn.shape[0])])
                xd3 = torch.from_numpy(np.stack((xd.real, xd.imag), axis=1))
                if config.save_fit: lcm.save_path = f'{save_path}quant_{method.lower()}/lcm_d3/'
                t_lcm_d3, u_lcm_d3 = lcm.forward(xd3)

                xd = np.array([wavelet((xn[i, 0] + 1j * xn[i, 1]).detach().cpu().numpy(), wavelet_shape='db1', threshold=300)
                               for i in range(xn.shape[0])])
                xd4 = torch.from_numpy(np.stack((xd.real, xd.imag), axis=1))
                if config.save_fit: lcm.save_path = f'{save_path}quant_{method.lower()}/lcm_d4/'
                t_lcm_d4, u_lcm_d4 = lcm.forward(xd4)

            if not os.path.exists(f'{save_path}quant_{method.lower()}/save'):
                os.makedirs(f'{save_path}quant_{method.lower()}/save')

            # save results
            torch.save(t, f'{save_path}quant_{method.lower()}/save/t.pt')
            torch.save(t_lcm_l, f'{save_path}quant_{method.lower()}/save/t_lcm_l.pt')
            torch.save(t_lcm_c, f'{save_path}quant_{method.lower()}/save/t_lcm_c.pt')

            torch.save(u_lcm_l, f'{save_path}quant_{method.lower()}/save/u_lcm_l.pt')
            torch.save(u_lcm_c, f'{save_path}quant_{method.lower()}/save/u_lcm_c.pt')

            torch.save(x, f'{save_path}quant_{method.lower()}/save/x.pt')
            torch.save(xs, f'{save_path}quant_{method.lower()}/save/xs.pt')
            torch.save(xl, f'{save_path}quant_{method.lower()}/save/xl.pt')
            torch.save(xc, f'{save_path}quant_{method.lower()}/save/xc.pt')

            torch.save(y, f'{save_path}quant_{method.lower()}/save/y.pt')
            torch.save(ys, f'{save_path}quant_{method.lower()}/save/ys.pt')
            torch.save(y_hat, f'{save_path}quant_{method.lower()}/save/y_hat.pt')
            torch.save(y_hat_s, f'{save_path}quant_{method.lower()}/save/y_hat_s.pt')

            torch.save(cwtmatr, f'{save_path}quant_{method.lower()}/save/cwtmatr.pt')
            torch.save(masks, f'{save_path}quant_{method.lower()}/save/masks.pt')

            if config.denoising:
                torch.save(t_lcm_d1, f'{save_path}quant_{method.lower()}/save/t_lcm_d1.pt')
                torch.save(t_lcm_d2, f'{save_path}quant_{method.lower()}/save/t_lcm_d2.pt')
                torch.save(t_lcm_d3, f'{save_path}quant_{method.lower()}/save/t_lcm_d3.pt')
                torch.save(t_lcm_d4, f'{save_path}quant_{method.lower()}/save/t_lcm_d4.pt')

                torch.save(u_lcm_d1, f'{save_path}quant_{method.lower()}/save/u_lcm_d1.pt')
                torch.save(u_lcm_d2, f'{save_path}quant_{method.lower()}/save/u_lcm_d2.pt')
                torch.save(u_lcm_d3, f'{save_path}quant_{method.lower()}/save/u_lcm_d3.pt')
                torch.save(u_lcm_d4, f'{save_path}quant_{method.lower()}/save/u_lcm_d4.pt')

                torch.save(xd1, f'{save_path}quant_{method.lower()}/save/xd1.pt')
                torch.save(xd2, f'{save_path}quant_{method.lower()}/save/xd2.pt')
                torch.save(xd3, f'{save_path}quant_{method.lower()}/save/xd3.pt')
                torch.save(xd4, f'{save_path}quant_{method.lower()}/save/xd4.pt')


    # load results
    t, t_lcm_l, t_lcm_c = {}, {}, {}
    u_lcm_l, u_lcm_c = {}, {}
    x, xs, xl, xc = {}, {}, {}, {}
    y, ys, y_hat, y_hat_s = {}, {}, {}, {}
    cwtmatr, masks = {}, {}

    for i, m in enumerate(methods):
        if m.lower() not in ['newton', 'mh', 'lcmodel']:
            raise ValueError(f'Method {m} not recognized.')

        t[i] = torch.load(f'{save_path}quant_{m.lower()}/save/t.pt')[:, :model.basisObj.n_metabs]
        t_lcm_l[i] = torch.load(f'{save_path}quant_{m.lower()}/save/t_lcm_l.pt')[:, :model.basisObj.n_metabs]
        t_lcm_c[i] = torch.load(f'{save_path}quant_{m.lower()}/save/t_lcm_c.pt')[:, :model.basisObj.n_metabs]

        u_lcm_l[i] = torch.load(f'{save_path}quant_{m.lower()}/save/u_lcm_l.pt')[:, :model.basisObj.n_metabs]
        u_lcm_c[i] = torch.load(f'{save_path}quant_{m.lower()}/save/u_lcm_c.pt')[:, :model.basisObj.n_metabs]

        x[i] = torch.load(f'{save_path}quant_{m.lower()}/save/x.pt')
        xs[i] = torch.load(f'{save_path}quant_{m.lower()}/save/xs.pt')
        xl[i] = torch.load(f'{save_path}quant_{m.lower()}/save/xl.pt')
        xc[i] = torch.load(f'{save_path}quant_{m.lower()}/save/xc.pt')

        y[i] = torch.load(f'{save_path}quant_{m.lower()}/save/y.pt')
        ys[i] = torch.load(f'{save_path}quant_{m.lower()}/save/ys.pt')
        y_hat[i] = torch.load(f'{save_path}quant_{m.lower()}/save/y_hat.pt')
        y_hat_s[i] = torch.load(f'{save_path}quant_{m.lower()}/save/y_hat_s.pt')

        cwtmatr[i] = torch.load(f'{save_path}quant_{m.lower()}/save/cwtmatr.pt')
        masks[i] = torch.load(f'{save_path}quant_{m.lower()}/save/masks.pt')

    print(f'Loaded {t[i].shape[0]} samples of {config.dataType} data for method(s): {methods}...')

    t_t, t_lcm_l_t, t_lcm_c_t = {}, {}, {}
    error, error_c, error_t, error_c_t = {}, {}, {}, {}

    for i, m in enumerate(methods):

        # reference metabolite concentrations
        t_lcm_l[i] = model.optimalReference(t[i], t_lcm_l[i]) * t_lcm_l[i]
        t_lcm_c[i] = model.optimalReference(t[i], t_lcm_c[i]) * t_lcm_c[i]

        # compute total metabolites
        t_t[i] = computeTotalMetabolites(t[i], model.basisObj.names[:model.basisObj.n_metabs])
        t_lcm_l_t[i] = computeTotalMetabolites(t_lcm_l[i], model.basisObj.names[:model.basisObj.n_metabs])
        t_lcm_c_t[i] = computeTotalMetabolites(t_lcm_c[i], model.basisObj.names[:model.basisObj.n_metabs])

        if config.relative:  # relative concentrations
            t[i] = t[i] / (t_t[i]['tCr'].unsqueeze(-1) + torch.finfo(t[i].dtype).eps)
            t_lcm_l[i] = t_lcm_l[i] / (t_lcm_l_t[i]['tCr'].unsqueeze(-1) + torch.finfo(t_lcm_l[i].dtype).eps)
            t_lcm_c[i] = t_lcm_c[i] / (t_lcm_c_t[i]['tCr'].unsqueeze(-1) + torch.finfo(t_lcm_c[i].dtype).eps)

        # plot and print results for idx
        error[i] = model.concsLoss(t[i], t_lcm_l[i], type=config.error).detach().cpu().numpy()
        error_c[i] = model.concsLoss(t[i], t_lcm_c[i], type=config.error).detach().cpu().numpy()

        error_t[i] = model.concsLoss(torch.stack(list(t_t[i].values()), dim=1), torch.stack(list(t_lcm_l_t[i].values()), dim=1),
                                     type=config.error).detach().cpu().numpy()
        error_c_t[i] = model.concsLoss(torch.stack(list(t_t[i].values()), dim=1), torch.stack(list(t_lcm_c_t[i].values()), dim=1),
                                       type=config.error).detach().cpu().numpy()

        print(f'{m} error: {error[i].mean()} ({error[i].std() / np.sqrt(error[i].shape[0])})')
        print(f'{m}_c error: {error_c[i].mean()} ({error_c[i].std() / np.sqrt(error_c[i].shape[0])}')


    # choose colors that match plasma cmap
    colors = {'Newton': ['royalblue', 'mediumpurple'],
              'LCModel': ['darkslategray', 'turquoise'],
              'MH': ['brown', 'coral']}
    color_list = []
    for m in methods: color_list.extend(colors[m])


    # plot results
    algs = {}
    for i, m in enumerate(methods):

        if m.lower() == 'newton' or m.lower() == 'mh': m_name = 'FSL-MRS'
        else: m_name = m

        algs[m_name] = error[i]
        algs[f'{m_name} + WAND'] = error_c[i]

    plot_algs_bars(algs, model.basisObj.names[:model.basisObj.n_metabs], yLabel='MAE [mM]', colors=color_list)
    if config.save: plt.savefig(f'{save_path}concs_all.png', dpi=1000)
    else: plt.show()

    # plot bar plot for individual samples
    for idx in range(config.plot_ind):
        algs = {}
        for i, m in enumerate(methods):

            if m.lower() == 'newton' or m.lower() == 'mh': m_name = 'FSL-MRS'
            else: m_name = m

            algs[m_name] = error[i][idx:idx + 1]
            algs[f'{m_name} + WAND'] = error_c[i][idx:idx + 1]

        plot_algs_bars(algs, model.basisObj.names[:model.basisObj.n_metabs], yLabel='Absolute Error [mM]', colors=color_list)
        if config.save:
            if not os.path.exists(save_path + 'inds/'): os.makedirs(save_path + 'inds/')
            plt.savefig(f'{save_path}inds/concs_{idx}.png', dpi=1000)

        algs = {}
        for i, m in enumerate(methods):

            if m.lower() == 'newton' or m.lower() == 'mh': m_name = 'FSL-MRS'
            else: m_name = m

            algs[m_name] = error_t[i][idx:idx + 1]
            algs[f'{m_name} + WAND'] = error_c_t[i][idx:idx + 1]

        plot_algs_bars(algs, list(t_t[list(t_t.keys())[0]].keys()), yLabel='Absolute Error [mM]', colors=color_list)
        if config.save:
            if not os.path.exists(save_path + 'inds/'): os.makedirs(save_path + 'inds/')
            plt.savefig(f'{save_path}inds/concs_t_{idx}.png', dpi=1000)

        else: plt.show()


    # plot uncertainties
    algs = {}
    for i, m in enumerate(methods):
        if m.lower() == 'newton' or m.lower() == 'mh': m_name = 'FSL-MRS'
        else: m_name = m

        algs[m_name] = u_lcm_l[i].detach().cpu().numpy()
        algs[f'{m_name} + WAND'] = u_lcm_c[i].detach().cpu().numpy()

    plot_algs_bars(algs, model.basisObj.names[:model.basisObj.n_metabs], yLabel='CRLB%', colors=color_list)
    if config.save: plt.savefig(f'{save_path}uncertainties_all.png', dpi=1000)


    # plot percentage errors
    for i, m in enumerate(methods):
        if m.lower() == 'newton' or m.lower() == 'mh': m_name = 'FSL-MRS'
        else: m_name = m

        perr = model.concsLoss(t[i], t_lcm_l[i], type='pe').detach().cpu().numpy()
        perr_c = model.concsLoss(t[i], t_lcm_c[i], type='pe').detach().cpu().numpy()

        plot_percentage_error(perr, model.basisObj.names, f'Percentage Error ({m})', f'{save_path}pe_{m}.png')
        plot_percentage_error(perr_c, model.basisObj.names, f'Percentage Error ({m} + WAND)', f'{save_path}pe_{m}_c.png')

        plot_percentage_error(u_lcm_l[i].detach().cpu().numpy(), model.basisObj.names,
                              f'CRLB% ({m_name})', f'{save_path}crlb_pe_{m}.png')
        plot_percentage_error(u_lcm_c[i].detach().cpu().numpy(), model.basisObj.names,
                              f'CRLB% ({m_name} + WAND)', f'{save_path}crlb_pe_{m}_c.png')

        # add one with colorbar
        plot_percentage_error(perr, model.basisObj.names, f'CB', f'{save_path}pe_cb.png')
        plot_percentage_error(perr_c, model.basisObj.names, 'CRLB CB', f'{save_path}crlb_cb.png')



    if config.denoising:
        t_lcm_d1, t_lcm_d2, t_lcm_d3, t_lcm_d4 = {}, {}, {}, {}
        u_lcm_d1, u_lcm_d2, u_lcm_d3, u_lcm_d4 = {}, {}, {}, {}
        error_d1, error_d2, error_d3, error_d4 = {}, {}, {}, {}
        for i, m in enumerate(methods):
            t_lcm_d1[i] = torch.load(f'{save_path}quant_{m.lower()}/save/t_lcm_d1.pt')[:, :model.basisObj.n_metabs]
            t_lcm_d2[i] = torch.load(f'{save_path}quant_{m.lower()}/save/t_lcm_d2.pt')[:, :model.basisObj.n_metabs]
            t_lcm_d3[i] = torch.load(f'{save_path}quant_{m.lower()}/save/t_lcm_d3.pt')[:, :model.basisObj.n_metabs]
            t_lcm_d4[i] = torch.load(f'{save_path}quant_{m.lower()}/save/t_lcm_d4.pt')[:, :model.basisObj.n_metabs]

            u_lcm_d1[i] = torch.load(f'{save_path}quant_{m.lower()}/save/u_lcm_d1.pt')[:, :model.basisObj.n_metabs].detach().cpu().numpy()
            u_lcm_d2[i] = torch.load(f'{save_path}quant_{m.lower()}/save/u_lcm_d2.pt')[:, :model.basisObj.n_metabs].detach().cpu().numpy()
            u_lcm_d3[i] = torch.load(f'{save_path}quant_{m.lower()}/save/u_lcm_d3.pt')[:, :model.basisObj.n_metabs].detach().cpu().numpy()
            u_lcm_d4[i] = torch.load(f'{save_path}quant_{m.lower()}/save/u_lcm_d4.pt')[:, :model.basisObj.n_metabs].detach().cpu().numpy()

            # reference metabolite concentrations
            t_lcm_d1[i] = model.optimalReference(t[i], t_lcm_d1[i]) * t_lcm_d1[i]
            t_lcm_d2[i] = model.optimalReference(t[i], t_lcm_d2[i]) * t_lcm_d2[i]
            t_lcm_d3[i] = model.optimalReference(t[i], t_lcm_d3[i]) * t_lcm_d3[i]
            t_lcm_d4[i] = model.optimalReference(t[i], t_lcm_d4[i]) * t_lcm_d4[i]

            # compute errors
            error_d1[i] = model.concsLoss(t[i], t_lcm_d1[i], type=config.error).detach().cpu().numpy()
            error_d2[i] = model.concsLoss(t[i], t_lcm_d2[i], type=config.error).detach().cpu().numpy()
            error_d3[i] = model.concsLoss(t[i], t_lcm_d3[i], type=config.error).detach().cpu().numpy()
            error_d4[i] = model.concsLoss(t[i], t_lcm_d4[i], type=config.error).detach().cpu().numpy()

            print(f'{m} + D1 error: {error_d1[i].mean()} ({error_d1[i].std() / np.sqrt(error_d1[i].shape[0])})')
            print(f'{m} + D2 error: {error_d2[i].mean()} ({error_d2[i].std() / np.sqrt(error_d2[i].shape[0])})')
            print(f'{m} + D3 error: {error_d3[i].mean()} ({error_d3[i].std() / np.sqrt(error_d3[i].shape[0])})')
            print(f'{m} + D4 error: {error_d4[i].mean()} ({error_d4[i].std() / np.sqrt(error_d4[i].shape[0])})')

            # plot results
            algs = {}
            algs[m] = error[i]
            algs[f'{m} + WAND'] = error_c[i]
            algs[f'{m} + Gauss. Win.'] = error_d1[i]
            algs[f'{m} + SIFT'] = error_d2[i]
            algs[f'{m} + HSVD'] = error_d3[i]
            algs[f'{m} + Wavelet Thresh.'] = error_d4[i]

            plot_algs_bars(algs, model.basisObj.names[:model.basisObj.n_metabs], yLabel='MAE [mM]')
            if config.save: plt.savefig(f'{save_path}concs_all_{m}_denoising.png', dpi=1000)

            # plot for individual samples
            for idx in range(config.plot_ind):
                algs = {}
                algs[m] = error[i][idx:idx + 1]
                algs[f'{m} + WAND'] = error_c[i][idx:idx + 1]
                algs[f'{m} + Gauss. Win.'] = error_d1[i][idx:idx + 1]
                algs[f'{m} + SIFT'] = error_d2[i][idx:idx + 1]
                algs[f'{m} + HSVD'] = error_d3[i][idx:idx + 1]
                algs[f'{m} + Wavelet Thresh.'] = error_d4[i][idx:idx + 1]

                plot_algs_bars(algs, model.basisObj.names[:model.basisObj.n_metabs], yLabel='Absolute Error [mM]')
                if config.save:
                    if not os.path.exists(save_path + 'inds/'): os.makedirs(save_path + 'inds/')
                    plt.savefig(f'{save_path}inds/concs_{idx}_{m}_denoising.png', dpi=1000)
                else: plt.show()

            # plot uncertainties
            algs = {}
            algs[m] = u_lcm_l[i].detach().cpu().numpy()
            algs[f'{m} + WAND'] = u_lcm_c[i].detach().cpu().numpy()
            algs[f'{m} + Gauss. Win.'] = torch.load(f'{save_path}quant_{m.lower()}/save/u_lcm_d1.pt').detach().cpu().numpy()
            algs[f'{m} + SIFT'] = torch.load(f'{save_path}quant_{m.lower()}/save/u_lcm_d2.pt').detach().cpu().numpy()
            algs[f'{m} + HSVD'] = torch.load(f'{save_path}quant_{m.lower()}/save/u_lcm_d3.pt').detach().cpu().numpy()
            algs[f'{m} + Wavelet Thresh.'] = torch.load(f'{save_path}quant_{m.lower()}/save/u_lcm_d4.pt').detach().cpu().numpy()

            plot_algs_bars(algs, model.basisObj.names[:model.basisObj.n_metabs], yLabel='CRLB%')
            if config.save: plt.savefig(f'{save_path}uncertainties_all_{m}_denoising.png', dpi=1000)


        def violinplot(data, labels, ylabel, save=None, palette=None):
            import seaborn as sns

            plt.figure(figsize=(len(labels) + 2, 4))
            if palette is None:
                palette = [colors[m][0], colors[m][1], 'mediumseagreen', 'skyblue', 'khaki', 'tan']
            vio = sns.violinplot(data=data, palette=palette, saturation=1.0)

            # set specific colors with alpha
            for patch in vio.collections:
                r, g, b, _ = patch.get_facecolor().flatten()
                patch.set_facecolor((r, g, b, 0.6))  # Adjust the alpha value here

            plt.xticks(ticks=range(len(labels)), labels=labels, rotation=0)
            plt.ylabel(ylabel)

            if len(labels) == 8 and ylabel == 'MAE [mM]': plt.ylim(-0.25, 2.1)
            elif len(labels) == 8 and ylabel == 'Mean CRLB%': plt.ylim(-50, 450)

            if save: plt.savefig(save, dpi=1000, bbox_inches='tight', transparent=True)
            else: plt.show()


        violinplot([error[0].mean(-1), error_c[0].mean(-1), error[1].mean(-1), error_c[1].mean(-1),
                        error_d1[1].mean(-1), error_d2[1].mean(-1), error_d3[1].mean(-1), error_d4[1].mean(-1)],
                          ['FSL-MRS', 'FSL-MRS +\nWAND', 'LCModel', 'LCModel +\nWAND', 'LCModel +\nGauss. Win.',
                           'LCModel +\nSIFT', 'LCModel +\nHSVD', 'LCModel +\nWavelet Thresh.'],
                            'MAE [mM]', f'{save_path}violin_mae_all_denoising.png',
                            palette=[colors['Newton'][0], colors['Newton'][1], colors['LCModel'][0], colors['LCModel'][1],
                                     'mediumseagreen', 'skyblue', 'khaki', 'tan'])

        violinplot([u_lcm_l[0].mean(-1), u_lcm_c[0].mean(-1), u_lcm_l[1].mean(-1), u_lcm_c[1].mean(-1),
                    u_lcm_d1[1].mean(-1), u_lcm_d2[1].mean(-1), u_lcm_d3[1].mean(-1), u_lcm_d4[1].mean(-1)],
                   ['FSL-MRS', 'FSL-MRS +\nWAND', 'LCModel', 'LCModel +\nWAND', 'LCModel +\nGauss. Win.',
                    'LCModel +\nSIFT', 'LCModel +\nHSVD', 'LCModel +\nWavelet Thresh.'],
                   'Mean CRLB%', f'{save_path}violin_mcrlb_all_denoising.png',
                     palette=[colors['Newton'][0], colors['Newton'][1], colors['LCModel'][0], colors['LCModel'][1],
                             'mediumseagreen', 'skyblue', 'khaki', 'tan'])


    # plot the LCModel fits
    if 'LCModel' in methods:
        for idx in range(config.plot_ind):
            def plot_LCModel_fit(fit, name):
                plt.figure()
                plt.plot(fit['ppm'], fit['data'], 'k', label='Data', linewidth=1)
                plt.plot(fit['ppm'], fit['completeFit'], 'r', label='Fit', alpha=0.6, linewidth=2)
                plt.plot(fit['ppm'], fit['data'] - fit['completeFit'] + 1.1 * np.max(fit['data']), 'k', label='Residual', alpha=0.8, linewidth=1)

                # compute scale factor
                # scale = xc[0][idx, 0, model.first:model.last].detach().cpu().numpy() / fit['completeFit'][model.first:model.last]
                #
                # art = xl[0][idx, 0, model.first:model.last].detach().cpu().numpy() - xc[0][idx, 0, model.first:model.last].detach().cpu().numpy()
                # plt.plot(np.linspace(fit['ppm'][0], fit['ppm'][-1], art.shape[0]), art, 'b', label='Artifact', alpha=0.5, linewidth=2)

                plt.xlabel('Chemical Shift [ppm]')
                plt.gca().invert_xaxis()

                # remove left, top, right spines
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['left'].set_visible(False)

                # remove y axis
                plt.gca().yaxis.set_visible(False)
                plt.gca().yaxis.set_ticks_position('none')

                if config.save:
                    if not os.path.exists(save_path + 'lcm_fits/'): os.makedirs(save_path + 'lcm_fits/')
                    plt.savefig(f'{save_path}lcm_fits/{name}', dpi=1000, bbox_inches='tight', transparent=True)
                else:
                    plt.show()


            fit = read_LCModel_fit(f'{save_path}quant_lcmodel/lcm_l/temp{idx}.coord')
            plot_LCModel_fit(fit, name=f'lcm_l_{idx}.png')

            fit = read_LCModel_fit(f'{save_path}quant_lcmodel/lcm_c/temp{idx}.coord')
            plot_LCModel_fit(fit, name=f'lcm_c_{idx}.png')

            fit = read_LCModel_fit(f'{save_path}quant_lcmodel/lcm_d1/temp{idx}.coord')
            plot_LCModel_fit(fit, name=f'lcm_d1_{idx}.png')

            fit = read_LCModel_fit(f'{save_path}quant_lcmodel/lcm_d2/temp{idx}.coord')
            plot_LCModel_fit(fit, name=f'lcm_d2_{idx}.png')

            fit = read_LCModel_fit(f'{save_path}quant_lcmodel/lcm_d3/temp{idx}.coord')
            plot_LCModel_fit(fit, name=f'lcm_d3_{idx}.png')

            fit = read_LCModel_fit(f'{save_path}quant_lcmodel/lcm_d4/temp{idx}.coord')
            plot_LCModel_fit(fit, name=f'lcm_d4_{idx}.png')


    # check if folder exists and create json file of config
    if os.path.exists(save_path):
        with open(f'{save_path}config.json', 'w') as f:
            f.write(str(config))
