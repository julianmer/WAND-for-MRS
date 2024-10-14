####################################################################################################
#                                         visChallenge.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/02/24                                                                                #
#                                                                                                  #
# Purpose: Visualizes results connected to the ISMRM 2016 fitting challenge data.                  #
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
import shutup; shutup.please()   # shut up warnings
import torch

from tqdm import tqdm

# own
from frameworks.frameworkFSL import FrameworkFSL
from frameworks.frameworkLCM import FrameworkLCModel
from simulation.dataModules import SynthDataModule, ChallengeDataModule
from train import Pipeline
from utils.auxiliary import challenge_scenario_indices, computeTotalMetabolites
from visualizations.plotFunctions import *



#****************#
#   initialize   #
#****************#
config = {
    # path to a trained model
    'path2trained': './lightning_logs/sskt2pvt/checkpoints/epoch=0-step=1195264.ckpt',   # norm rw p

    # path to basis set
    'path2basis': '../Data/BasisSets/ISMRM_challenge/basisset_JMRUI/',

    # for nn models
    'arch': 'unet',   # 'unet', ...
    'specType': 'synth',  # 'synth', 'invivo', 'biggaba', 'fMRSinPain'
    'basisFmt': '',  # '', 'cha_philips', 'biggaba', 'fMRSinPain'

    # for lcm model
    'method': ['Newton', 'LCModel'],   # 'Newton', 'MH', 'LCModel'
    'ppmlim': (0.5, 4.2),  # ppm limits for the spectra

    # data settings
    'dataType': 'cha',   # 'cha', 'clean', 'std', 'norm', 'norm_rw', 'norm_rw_p', 'custom', ...
    'note': 'norm_rw_p',   # additional note

    'test_size': 28,   # number of test samples
    'load_model': True,   # load model from path2trained
    'skip_train': True,   # skip the training procedure

    # visual settings
    'run': True,   # run the inference (will try to load results if False)
    'save': True,   # save the plots
    'relative': False,   # relative concentrations (/tCr)
    'error': 'mae',   # 'mae', 'mse', 'mape', ...
    'linewidth': 5,  # linewidth for plots
    'idx': list(range(28)),   # batch index (all if empty)
    'metIdx': [],   # metabolite index (all if empty)
    'imag': False,   # plot imaginary part of spectra
    'ind_limit': 28,   # limit plots to this number of samples
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
            lcm.save_path = f'{save_path}quant_{method.lower()}/lcm_l/'
            t_lcm_l, u_lcm_l = lcm.forward(xl)

            # get lcm quantification without artifact signal
            xc = xl - y_hat[..., -1]   # remove artifact signal
            lcm.save_path = f'{save_path}quant_{method.lower()}/lcm_c/'
            t_lcm_c, u_lcm_c = lcm.forward(xc)

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
    error, error_c = {}, {}

    for i, m in enumerate(methods):

        # reference metabolite concentrations
        t_lcm_l[i] = model.optimalReference(t[i], t_lcm_l[i]) * t_lcm_l[i]
        t_lcm_c[i] = model.optimalReference(t[i], t_lcm_c[i]) * t_lcm_c[i]

        # compute total metabolites
        t_t[i] = computeTotalMetabolites(t[i], model.basisObj.names[:model.basisObj.n_metabs])
        t_lcm_l_t[i] = computeTotalMetabolites(t_lcm_l[i], model.basisObj.names[:model.basisObj.n_metabs])
        t_lcm_c_t[i] = computeTotalMetabolites(t_lcm_c[i], model.basisObj.names[:model.basisObj.n_metabs])

        if config.relative:   # relative concentrations
            t[i] = t[i] / (t_t[i]['tCr'].unsqueeze(-1) + torch.finfo(t[i].dtype).eps)
            t_lcm_l[i] = t_lcm_l[i] / (t_lcm_l_t[i]['tCr'].unsqueeze(-1) + torch.finfo(t_lcm_l[i].dtype).eps)
            t_lcm_c[i] = t_lcm_c[i] / (t_lcm_c_t[i]['tCr'].unsqueeze(-1) + torch.finfo(t_lcm_c[i].dtype).eps)

        # plot and print results for idx
        error[i] = model.concsLoss(t[i], t_lcm_l[i], type=config.error).detach().cpu().numpy()
        error_c[i] = model.concsLoss(t[i], t_lcm_c[i], type=config.error).detach().cpu().numpy()

        print(f'{m} error:', error[i].mean())
        print(f'{m}_c error:', error_c[i].mean())

        # # give based on scenario
        # if config.dataType[:3] == 'cha':
        #     scenarios = challenge_scenario_indices()
        #     for key, value in scenarios.items():
        #         print(f'{key}: {error[m][value].mean()}')
        #         print(f'{key}_c: {error_c[m][value].mean()}')

    # sort
    mapping = {1:0, 2:11, 3:21, 4:22, 5:23, 6:24, 7:25, 8:26, 9:27, 10:1, 11:2, 12:3,
               13:4, 14:5, 15:6, 16:7, 17:8, 18:9, 19:10, 20:12, 21:13, 22:14, 23:15,
               24:16, 25:17, 26:18, 27:19, 28:20}
    idx = list(mapping.values())

    labels = {2:'Lorentzian', 5:'Gaussian', 8.5:'GABA/GSH', 11:'No MM', 13:'SNR=20', 16:'SNR=30',
              19:'SNR=40', 21:'SNR\n=160', 22:'EC', 23.5:'Residual\nWater', 26.5:'Tumor-Like'}
    borders = [3.5, 6.5, 10.5, 11.5, 14.5, 17.5, 20.5, 21.5, 22.5, 24.5]

    # choose colors that match plasma cmap
    colors = {'Newton': ['royalblue', 'mediumpurple'],
              'LCModel': ['darkslategray', 'turquoise'],
              'MH': ['brown', 'coral']}

    # plot total metabolites
    for key in t_t[i].keys():
        if key == 'NAA+NAAG': plt.figure(figsize=(17, 4))   # 17, 4
        else: plt.figure(figsize=(7.5, 2.5))   # 8, 3

        plt.plot(range(1, len(idx) + 1), t_t[i][key][idx].detach().cpu().numpy(),
                 'k', label='True', marker='s')

        for i, m in enumerate(methods):

            if m.lower() == 'newton' or m.lower() == 'mh': m_name = 'FSL-MRS'
            else: m_name = m

            plt.plot(range(1, len(idx) + 1), t_lcm_l_t[i][key][idx].detach().cpu().numpy(),
                     '--', label=m_name, marker='o', color=colors[m][0], alpha=0.8)
            plt.plot(range(1, len(idx) + 1), t_lcm_c_t[i][key][idx].detach().cpu().numpy(),
                     '-.', label=m_name + ' + WAND', marker='^', color=colors[m][1], alpha=0.8)

        # labels
        plt.ylabel('Concentration [mM]')

        # ticks per dataset
        plt.xticks(range(1, len(idx) + 1), range(1, len(idx) + 1))

        # have y ticks be integers
        from matplotlib.ticker import MaxNLocator
        plt.gca().get_yaxis().set_major_locator(MaxNLocator(integer=True))

        # add dashed lines between scenarios
        for border in borders: plt.axvline(x=border, color='grey', linestyle='dotted')

        # add specific labels
        if key == 'NAA+NAAG':
            plt.xlabel('Dataset')

            for loc in labels.keys():
                plt.text(loc, -3, labels[loc], horizontalalignment='center',
                         verticalalignment='center', rotation=0, fontsize=10)

            plt.legend(frameon=False)

            all = [[t_t[i], t_lcm_l_t[i], t_lcm_c_t[i]] for i in range(len(methods))]
            all = [item for sublist in all for item in sublist]

            # add whitespace above x-axis to fit labels
            plt.ylim(min([a[key][idx].min().item() for a in all]) - 4.9,
                     max([a[key][idx].max().item() for a in all]) + 1.0)

        plt.title(f'{key}')

        if config.save:
            plt.savefig(f'{save_path}{key}.png', dpi=1000, bbox_inches='tight')
        else:
            plt.show()


    # plot percentage error
    for i, m in enumerate(methods):
        perr = model.concsLoss(t[i], t_lcm_l[i], type='pe').detach().cpu().numpy()
        perr_c = model.concsLoss(t[i], t_lcm_c[i], type='pe').detach().cpu().numpy()

        if m.lower() == 'newton' or m.lower() == 'mh': m_name = 'FSL-MRS'
        else: m_name = m

        # index mapping
        perr = perr[idx]
        perr_c = perr_c[idx]
        u_lcm_l[i] = u_lcm_l[i][idx]
        u_lcm_c[i] = u_lcm_c[i][idx]

        plot_percentage_error(perr, model.basisObj.names, f'Percentage Error ({m_name})',
                              f'{save_path}pe_{m}.png', 'equal')
        plot_percentage_error(perr_c, model.basisObj.names, f'Percentage Error ({m_name} + WAND)',
                              f'{save_path}pe_{m}_c.png', 'equal')

        plot_percentage_error(u_lcm_l[i].detach().cpu().numpy(), model.basisObj.names,
                              f'CRLB% ({m_name})', f'{save_path}crlb_pe_{m}.png', 'equal')
        plot_percentage_error(u_lcm_c[i].detach().cpu().numpy(), model.basisObj.names,
                              f'CRLB% ({m_name} + WAND)', f'{save_path}crlb_pe_{m}_c.png', 'equal')

        # add one with colorbar
        plot_percentage_error(perr, model.basisObj.names, f'CB', f'{save_path}pe_cb.png', 'equal')
        plot_percentage_error(perr_c, model.basisObj.names, f'CRLB CB', f'{save_path}crlb_cb.png', 'equal')

        # print average percentage error
        metabs = ['Ala', 'Asc', 'Asp', 'Cr', 'GABA', 'GPC', 'GSH', 'Glc', 'Gln', 'Glu',
                  'Gly', 'Ins', 'Lac', 'NAA', 'NAAG', 'PCho', 'PCr', 'PE', 'Tau', 'sIns']
        metab_idx = [model.basisObj.names.index(m) for m in metabs]

        print(f'{m} average percentage error:', perr[:, metab_idx].mean())
        print(f'{m}_c average percentage error:', perr_c[:, metab_idx].mean())


    # check if folder exists and create json file of config
    if os.path.exists(save_path):
        with open(f'{save_path}config.json', 'w') as f:
            f.write(str(config))