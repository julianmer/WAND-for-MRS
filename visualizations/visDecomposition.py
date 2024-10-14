####################################################################################################
#                                       visDecomposition.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/02/24                                                                                #
#                                                                                                  #
# Purpose: Visualizes and quantifies the decomposition performance.                                #
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
import pandas as pd
import shutup; shutup.please()   # shut up warnings
import torch

# own
from simulation.dataModules import SynthDataModule, ChallengeDataModule, InVivoDataModule
from train import Pipeline
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
    'specType': 'synth',  # 'synth', 'invivo', 'biggaba', 'fMRSinPain'
    'basisFmt': '',  # '', 'cha_philips', 'biggaba', 'fMRSinPain'
    'ppmlim': (0.5, 4.2),  # ppm limits for the spectra

    # data settings
    'dataType': 'norm_rw_p',   # 'cha', 'clean', 'std', 'norm', 'norm_rw', 'norm_rw_p', 'custom', ...
    'note': 'norm_rw_p',   # additional note

    'test_size': 32,   # number of test samples
    'load_model': True,   # load model from path2trained
    'skip_train': True,   # skip the training procedure

    # visual settings
    'table': True,   # save table with decomposition error
    'plot_over': True,   # plot overview (scalograms, masks, metabolite spectra)
    'plot_decomp': True,  # plot decomposition (predicted and ground truth)
    'linewidth': 2.25,   # linewidth of the plots
    'idx': [],   # batch index (all if empty)
    'metIdx': [],   # metabolite index (all if empty)
    'imag': False,   # plot imaginary part of the spectra
}



if __name__ == '__main__':

    # initialize pipeline
    pipeline = Pipeline()
    pipeline.default_config.update(config)
    pipeline.main({'online': False})

    config, model = pipeline.config, pipeline.model
    save_path = f'../Imgs/{config.dataType}/{config.note}/'

    # test data
    if config.dataType[:6] == 'invivo':
        dataloader = InVivoDataModule(# data_dir='../Data/DataSets/InVivo/',
                                      data_dir='../Data/DataSets/fMRSinPain/SUBJECTS/BASELINE/',
                                      basis_dir=config.path2basis,
                                      nums_test=config.test_size,
                                      pre_pro=False,)
        x, = next(iter(dataloader.test_dataloader()))
        y, t = None, None
        model = model.to(x.device)

    else:
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
        model = model.to(x.device)


    # inference
    xn, yn, t = model.prepareData(x, y, t)

    if xn.shape[0] > 32:  # memory constraints (TODO: find leak)
        model = model.to('cpu')
        y_hat, (cwtmatr, masks) = model.forward(xn.to('cpu'))
        y_hat, (cwtmatr, masks) = y_hat.to(x.device), (cwtmatr.to(x.device), masks.to(x.device))
    else:
        y_hat, (cwtmatr, masks) = model.forward(xn)

    xs = xn[:, :, model.first:model.last:model.skip]
    if config.dataType[:6] == 'invivo': ys = None
    else: ys = yn[:, :, model.first:model.last:model.skip]

    # # take artifact channel and push trough network again
    # ar = torch.zeros_like(xn)
    # ar[:, :, model.first:model.last] = y_hat[..., -1]
    # y_hat, _ = model.forward(ar)
    # xs = ar[:, :, model.first:model.last:model.skip].detach()

    # visualize specs
    linewidth = config.linewidth
    idx, metIdx, imag = config.idx, config.metIdx, config.imag


    if config.table:   # capture mean deviation of predicted and actual decomposition
        mse = torch.mean((ys - y_hat) ** 2, dim=-2).detach().cpu().numpy()
        cos = torch.nn.functional.cosine_similarity(ys, y_hat, dim=-2).detach().cpu().numpy()

        act = (ys - xs.min(-1)[0][..., None, None]) / (
                xs.max(-1)[0][..., None, None] - xs.min(-1)[0][..., None, None])
        pre = (y_hat - xs.min(-1)[0][..., None, None]) / (
                xs.max(-1)[0][..., None, None] - xs.min(-1)[0][..., None, None])
        mse_sc = torch.mean((act - pre) ** 2, dim=-2).detach().cpu().numpy()

        act = (ys - ys.min(-2, keepdim=True)[0]) / (ys.max(-2, keepdim=True)[0] - ys.min(-2, keepdim=True)[0])
        pre = (y_hat - y_hat.min(-2, keepdim=True)[0]) / (
                y_hat.max(-2, keepdim=True)[0] - y_hat.min(-2, keepdim=True)[0])
        mse_si = torch.mean((act - pre) ** 2, dim=-2).detach().cpu().numpy()

        df = pd.DataFrame({}, index=model.basisObj.names)
        for im, com in enumerate(['Real', 'Imag']):
            df[f'MSE {com}'] = ["%.2e ± %.1e" % (float(np.mean(mse[:, im, m], 0)),
                                                 float(np.std(mse[:, im, m], 0) / np.sqrt(mse.shape[0])))
                                for m in range(mse.shape[-1])]
            df[f'MSE {com} (Norm)'] = ["%.2e ± %.1e" % (float(np.mean(mse_sc[:, im, m], 0)),
                                                         float(np.std(mse_sc[:, im, m], 0) / np.sqrt(mse_sc.shape[0])))
                                       for m in range(mse_sc.shape[-1])]
            df[f'MSE {com} (ind. Norm)'] = ["%.2e ± %.1e" % (float(np.mean(mse_si[:, im, m], 0)),
                                                             float(np.std(mse_si[:, im, m], 0) / np.sqrt(mse_si.shape[0])))
                                            for m in range(mse_si.shape[-1])]
            df[f'Cos Sim. {com}'] = ["%.2f ± %.1e" % (float(np.mean(cos[:, im, m], 0)),
                                                      float(np.std(cos[:, im, m], 0) / np.sqrt(cos.shape[0])))
                                     for m in range(cos.shape[-1])]

        if not os.path.exists(f'{save_path}'): os.makedirs(f'{save_path}')
        df.to_csv(f'{save_path}/dec_error.csv')


    # quick plots of original, decomposition, and reconstructed spectra
    def quick_analysis(xn, y_hat, xs, ys, imag, idx, metIdx, linewidth=3):
        plt.figure()
        plt.plot(xn[idx[0], int(imag), model.first:model.last].cpu().numpy(),
                 linewidth=linewidth, label='input spectrum')
        plt.legend(frameon=False)

        plt.figure()
        for i in range(y_hat.shape[-1]):
            if i == y_hat.shape[-1] - 1:
                plt.plot(y_hat[idx[0], int(imag), :, i].detach().cpu().numpy(),
                         linewidth=linewidth, label='artifact signal')
            else:
                plt.plot(y_hat[idx[0], int(imag), :, i].detach().cpu().numpy(),
                         linewidth=linewidth)
        plt.legend(frameon=False)

        plt.figure()
        plt.plot(xn[idx[0], int(imag), model.first:model.last].cpu().numpy(), color='grey',
                 linewidth=linewidth, label='input spectrum')
        plt.plot((xs - y_hat[..., -1])[idx[0], int(imag)].detach().cpu().numpy(),
                 linewidth=linewidth, label='reconstructed spectrum')
        plt.plot(ys[..., :-1].sum(-1)[idx[0], int(imag)].detach().cpu().numpy(), 'k--',
                 linewidth=linewidth, label='gt spectrum')
        plt.plot(((xs - y_hat[..., -1])[idx[0], int(imag)] -
                  ys[..., :-1].sum(-1)[idx[0], int(imag)]).detach().cpu().numpy(),
                 'r--', linewidth=linewidth, label='residual')
        plt.legend(frameon=False)

    # quick_analysis(xn, y_hat, xs, ys, imag, idx, metIdx, linewidth=linewidth)


    if config.plot_over:   # plot/save thoroughly the scalograms, masks, and metabolite spectra
        visualize_pred(xs, cwtmatr, masks, y_hat, model.names, imag=imag, batch_idx=idx, metab_idx=metIdx, linewidth=linewidth,
                       save_path=f'{save_path}/x{"Real" if not imag else "Imag"}/')

        if not config.dataType[:6] == 'invivo':
            visualize_truth(ys, model.names, imag=imag, batch_idx=idx, metab_idx=metIdx, linewidth=linewidth,
                            save_path=f'{save_path}/y{"Real" if not imag else "Imag"}/')


    if config.plot_decomp:   # plot decomposition in one figure
        plot_decomposition(xs, y_hat, model.names, imag=imag, batch_idx=idx, metab_idx=metIdx, linewidth=linewidth,
                           yLabel='Predicted Decomposition',
                           save_path=f'{save_path}/d{"Real" if not imag else "Imag"}/')

        if not config.dataType[:6] == 'invivo':
            plot_decomposition(xs, ys, model.names, imag=imag, batch_idx=idx, metab_idx=metIdx, linewidth=linewidth,
                               yLabel='Ground Truth Decomposition', truth=True,
                           save_path=f'{save_path}/d{"Real_gt" if not imag else "Imag_gt"}/')


    # check if folder exists and create json file of config
    if os.path.exists(f'{save_path}'):
        with open(f'{save_path}/config.json', 'w') as f:
            f.write(str(config))
