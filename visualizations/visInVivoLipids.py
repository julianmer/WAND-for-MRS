####################################################################################################
#                                        visInVivoLipids.py                                        #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/02/24                                                                                #
#                                                                                                  #
# Purpose: Visualizes the in-vivo lipid contamination data results.                                #
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
    'path2trained': './lightning_logs/gtc0zplt/checkpoints/epoch=0-step=67584.ckpt',   # norm rw p do (phillips)

    # path to basis set
    'path2basis': '../Data/BasisSets/ISMRM_challenge/press3T_30ms_Philips.BASIS',

    # for nn models
    'arch': 'unet',   # 'mlp', 'cnn', 'unet', 'aspp'
    'specType': 'synth',  # 'synth', 'invivo', 'biggaba', 'fMRSinPain'
    'basisFmt': 'cha_philips',  # '', 'cha_philips', 'biggaba', 'fMRSinPain'

    # for lcm model
    'method': ['Newton', 'LCModel'],  # 'Newton', 'MH', 'LCModel'
    'ppmlim': (0.5, 4.2),  # ppm limits for the spectra
    'lcmodel_ignore': 'default',  # 'default', 'none',
    'mm': False,  # include model internal macromolecules
    'lipids': True,  # include lipids in the basis set (only LCModel)
    'shift_ref': 'cr',  # 'cr', 'naa'
    'constrain_baseline': False,  # constrain baseline (for WAND)

    # data settings
    'dataType': 'invivo',  # 'cha', 'invivo', 'invivo_ua', 'synth'
    'note': 'norm_rw_p',  # additional note

    'test_size': 10,   # number of test samples
    'load_model': True,  # load model from path2trained
    'skip_train': True,  # skip the training procedure

    # visual settings
    'run': True,   # run the inference (will try to load results if False)
    'save': True,   # save the plots
    'error': 'mae',   # 'mae', 'mse', 'mape', ...
    'scale': 'tcr',   # 'gts', 'tcr', 'max'
    'linewidth': 5,   # linewidth for the plots
    'idx': [],    # batch index (all if empty)
    'metIdx': [],   # metabolite index (all if empty)
    'imag': False,  # plot imaginary part
    'plot_dec': True,  # plot the decomposition
    'plot_conc': True,  # plot the concentrations
    'plot_comp': True,   # plot the comparison
    'plot_conc_metabs': 'total',   # 'metabs', 'total', 'both'
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
                                   basisFmt=config.basisFmt, device=model.device)

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
            elif config.dataType == 'invivo' or config.dataType == 'invivo_up':
                dataloader = InVivoDataModule(data_dir='../Data/DataSets/InVivo/volunteer1/SPARSDAT/',
                                              # data_dir='../Data/DataSets/InVivoSelection/SPARSDAT/',
                                              # data_dir='../Data/DataSets/fMRSinPain/SUBJECTS/BASELINE/',
                                              basis_dir=config.path2basis,
                                              nums_test=config.test_size)
            elif config.dataType == 'invivo_ua':
                dataloader = InVivoNSAModule(data_dir='../Data/DataSets/InVivo/volunteer2/listdata/NSA128/',
                                             # data_dir='../Data/DataSets/InVivoSelection/LISTDATA/',
                                             # data_dir='../Data/DataSets/fMRSinPain/SUBJECTS/RAW/',
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

            # reference signal
            if config.dataType == 'invivo' or config.dataType == 'invivo_ua':

                # to nifti for processing.py
                x_proc = (x[:, 0] + 1j * x[:, 1]).detach().cpu().numpy()
                x_proc = np.fft.ifft(x_proc, axis=-1)
                x_proc = gen_nifti_mrs(data=x_proc.reshape(x_proc.shape[0:1] + (1, 1,) + x_proc.shape[1:]),
                                       dwelltime=lcm.basisFSL.original_dwell,
                                       spec_freq=lcm.basisFSL.cf,
                                       nucleus='1H',
                                       dim_tags=[None, None, None],
                                       no_conj=False)

                x_ref = next(iter(dataloader.ref_dataloader()))[:config.test_size]
                x_ref = (x_ref[:, 0] + 1j * x_ref[:, 1]).detach().cpu().numpy()
                x_ref = np.fft.ifft(x_ref, axis=-1)
                x_ref = gen_nifti_mrs(data=x_ref.reshape(x_ref.shape[0:1] + (1, 1,) + x_ref.shape[1:]),
                                      dwelltime=lcm.basisFSL.original_dwell,
                                      spec_freq=lcm.basisFSL.cf,
                                      nucleus='1H',
                                      dim_tags=[None, None, None],
                                      no_conj=False)

                # process (ATTENTION: SDAT is usually already processed)
                x_proc = proc.ecc(x_proc, x_ref)  # eddy current correction
                # x_proc = own_nifti_ecc(x_proc, x_ref)  # eddy current correction
                x_proc = proc.remove_peaks(x_proc, [-0.15, 0.15], limit_units='ppm')  # remove residual water
                if config.shift_ref.lower() == 'cr':
                    x_proc = proc.shift_to_reference(x_proc, 3.027, (2.9, 3.1))  # shift to reference
                    x_proc = proc.phase_correct(x_proc, (2.9, 3.1))  # phase correction
                elif config.shift_ref.lower() == 'naa':
                    x_proc = proc.shift_to_reference(x_proc, 2.02, (1.9, 2.1))  # shift to reference
                    x_proc = proc.phase_correct(x_proc, (1.9, 2.1))  # phase correction

                print(x_proc.hdr_ext)

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
            lcm.save_path = f'{save_path}quant_{method}/lcm/'
            t_lcm, u_lcm = lcm.forward(x)
            if method.lower() == 'lcmodel': lcm.control[1] = 'nunfil=1536'

            lcm.save_path = f'{save_path}quant_{method}/lcm_l/'
            t_lcm_l, u_lcm_l = lcm.forward(xl)

            # get lcm quantification without artifact signal
            xc = xl - y_hat[..., -1]  # remove artifact signal
            lcm.save_path = f'{save_path}quant_{method}/lcm_c/'
            if config.constrain_baseline:
                if method.lower() == 'lcmodel': lcm.control.insert(-2, 'dkntmn=1.5')
                elif method.lower() == 'newton' or method.lower() == 'mh': lcm.baseline_order = -1
            t_lcm_c, u_lcm_c = lcm.forward(xc)
            if config.constrain_baseline:
                if method.lower() == 'lcmodel': lcm.control.pop(-2)
                elif method.lower() == 'newton' or method.lower() == 'mh': lcm.baseline_order = 2

            # visualize specs
            linewidth = config.linewidth
            idx, metIdx, imag = config.idx, config.metIdx, config.imag

            if config.plot_dec:   # plot/save the scalograms, masks, and metabolite spectra
                visualize_pred(xs, cwtmatr, masks, y_hat_s, model.names, imag=imag, batch_idx=idx, metab_idx=metIdx,
                               save_path=f'{save_path}x{"Real" if not imag else "Imag"}/')

                # plot decomposition in one figure
                plot_decomposition(xs, y_hat_s, model.names, imag=imag, batch_idx=idx, metab_idx=metIdx, linewidth=2,
                                   save_path=f'{save_path}d{"Real" if not imag else "Imag"}/', yLim=False)

            # save estimated concentrations
            torch.save(t_lcm, f'{save_path}{method.lower()}_concs.pth')
            torch.save(t_lcm_l, f'{save_path}{method.lower()}_concs_l.pth')
            torch.save(t_lcm_c, f'{save_path}{method.lower()}_concs_c.pth')

            # save estimated uncertainties
            torch.save(u_lcm, f'{save_path}{method.lower()}_uncs.pth')
            torch.save(u_lcm_l, f'{save_path}{method.lower()}_uncs_l.pth')
            torch.save(u_lcm_c, f'{save_path}{method.lower()}_uncs_c.pth')

        # check if folder exists and create json file of config
        if os.path.exists(save_path):
            with open(f'{save_path}config.json', 'w') as f:
                f.write(str(config))


    # testing (load and visualize)
    if not isinstance(config.method, list): methods = [config.method]
    else: methods = config.method

    # get metab names
    tot_names = list(computeTotalMetabolites(torch.zeros(1, model.basisObj.n_metabs), model.basisObj.names).keys())

    colors = {'Newton': ['royalblue', 'mediumpurple', 'cornflowerblue', 'rebeccapurple'],
              'LCModel': ['darkslategray', 'turquoise', 'lightslategray', 'teal'],
              'MH': ['brown', 'coral', 'lightsalmon', 'indianred']}
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
        tot_concs[method] = torch.stack(list(computeTotalMetabolites(concs[method], model.basisObj.names, sIns=True).values()), dim=1)
        tot_concs_l[method] = torch.stack(list(computeTotalMetabolites(concs_l[method], model.basisObj.names, sIns=True).values()), dim=1)
        tot_concs_c[method] = torch.stack(list(computeTotalMetabolites(concs_c[method], model.basisObj.names, sIns=True).values()), dim=1)

    # ground truth concentrations
    gt_concs = None
    tot_gt_concs = {'NAA+NAAG': 1.3685, 'Cr+PCr': 1.0, 'Gln+Glu': 1.4357, 'Ins+Gly': 0.9883,
                    'GPC+PCho': 0.27961, 'GABA': 0.24077341, 'GSH': 0.423337856}
    tot_gt_std = {'NAA+NAAG': 0.14956, 'Cr+PCr': 0.0, 'Gln+Glu': 0.295167, 'Ins+Gly': 0.1375,
                  'GPC+PCho': 0.032389, 'GABA': 0.04409769, 'GSH': 0.07717096}

    def scaleTo(concs, type='gts', scale=None):
        if type.lower() == 'gts':
            return concs * model.optimalReference(scale, concs)
        elif type.lower() == 'tcr':
            return concs / scale[:, tot_names.index('Cr+PCr')].unsqueeze(1)
        elif type.lower() == 'max':
            return concs / concs.max(1, keepdim=True).values

    if config.scale == 'gts':   # scale to ground truth concentrations
        for method in methods:
            concs[method] = scaleTo(concs[method], 'gts', gt_concs)
            concs_l[method] = scaleTo(concs_l[method], 'gts', gt_concs)
            concs_c[method] = scaleTo(concs_c[method], 'gts', gt_concs)

            tot_concs[method] = scaleTo(tot_concs[method], 'gts', tot_gt_concs)
            tot_concs_l[method] = scaleTo(tot_concs_l[method], 'gts', tot_gt_concs)
            tot_concs_c[method] = scaleTo(tot_concs_c[method], 'gts', tot_gt_concs)

    elif config.scale == 'tcr':   # scale to total creatine
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
        def plot_bar(gt_concs, concs, concs_l, concs_c, names, idx=None, total=False):

            if idx is None: idx = list(range(concs[methods[0]].shape[0]))
            elif isinstance(idx, int): idx = [idx]

            algs = {}
            if total: algs.update({f'Literature': np.array(list(tot_gt_concs.values()))[np.newaxis, :]})
            algs.update({m: concs[m][idx].detach().cpu().numpy() for m in methods})
            # algs.update({f'{m} + L': concs_l[m][idx].detach().cpu().numpy() for m in methods})
            algs.update({f'{m} + WAND': concs_c[m][idx].detach().cpu().numpy() for m in methods})

            cs = [colors[m][0] for m in methods] + [colors[m][1] for m in methods]
            if total: cs = ['k'] + cs

            plot_algs_bars(algs, names, colors=cs, hlines=False)

            if total:
                plt.errorbar(np.arange(len(names)), list(tot_gt_concs.values()), yerr=list(tot_gt_std.values()),
                             fmt='none', ecolor='k', capsize=2,
                             elinewidth=1, capthick=1, alpha=0.8, zorder=3)

            if len(idx) == 1:
                if config.save:
                    if not os.path.exists(f'{save_path}concs_{idx[0]}/'): os.makedirs(f'{save_path}concs_{idx[0]}/')
                    plt.savefig(f'{save_path}concs_{idx[0]}/concs{"_t" if total else ""}.png', dpi=1000)
                else: plt.show()
            else:
                if config.save: plt.savefig(f'{save_path}concs{"_t" if total else ""}.png', dpi=1000)
                else: plt.show()

            for m in methods:
                algs = {}
                if total: algs.update({f'Literature': np.array(list(tot_gt_concs.values()))[np.newaxis, :]})
                algs.update({m: concs[m][idx].detach().cpu().numpy()})
                # algs.update({f'{m} + L': concs_l[m][idx].detach().cpu().numpy()})
                algs.update({f'{m} + WAND': concs_c[m][idx].detach().cpu().numpy()})
                cs = [colors[m][0]] + [colors[m][1]]
                if total: cs = ['k'] + cs

                plot_algs_bars(algs, names, colors=cs, hlines=False)

                if total:
                    plt.errorbar(np.arange(len(names)), list(tot_gt_concs.values()), yerr=list(tot_gt_std.values()),
                                 fmt='none', ecolor='k', capsize=2,
                                 elinewidth=1, capthick=1, alpha=0.8, zorder=3)

                if len(idx) == 1:
                    if config.save:
                        if not os.path.exists(f'{save_path}concs_{idx[0]}/'): os.makedirs(f'{save_path}concs_{idx[0]}/')
                        plt.savefig(f'{save_path}concs_{idx[0]}/concs_{m}{"_t" if total else ""}.png', dpi=1000)
                    else: plt.show()
                else:
                    if config.save: plt.savefig(f'{save_path}concs_{m}{"_t" if total else ""}.png', dpi=1000)
                    else: plt.show()


        # what to plot (metabolites or total metabolites)
        if config.plot_conc_metabs == 'metabs' or config.plot_conc_metabs == 'both':
            plot_bar(gt_concs, concs, concs_l, concs_c, model.basisObj.names[:model.basisObj.n_metabs])
        if config.plot_conc_metabs == 'total' or config.plot_conc_metabs == 'both':
            plot_bar(tot_gt_concs, tot_concs, tot_concs_l, tot_concs_c, tot_names, total=True)

        for i in range(concs[methods[0]].shape[0]):
            if config.plot_conc_metabs == 'metabs' or config.plot_conc_metabs == 'both':
                plot_bar(gt_concs, concs, concs_l, concs_c, model.basisObj.names[:model.basisObj.n_metabs], idx=i)
            if config.plot_conc_metabs == 'total' or config.plot_conc_metabs == 'both':
                plot_bar(tot_gt_concs, tot_concs, tot_concs_l, tot_concs_c, tot_names, idx=i, total=True)


    # compare concentrations
    if config.plot_comp:

        def plot_comparsion(concs, names, idx=None, total=False, add=''):

            for m in methods:
                algs = {}
                if total: algs.update({f'Literature': np.array(list(tot_gt_concs.values()))[np.newaxis, :]})
                algs.update({f'{m + add} (Voxel 1)': concs[m][0:1].detach().cpu().numpy()})
                algs.update({f'{m + add} (Voxel {i+1})': concs[m][idx:idx+1].detach().cpu().numpy()})

                cs = [colors[m][int(add == ' + WAND')]] + [colors[m][int(add == ' + WAND') + 2]]
                if total: cs = ['k'] + cs

                plot_algs_bars(algs, names, colors=cs, hlines=False)

                if total:
                    plt.errorbar(np.arange(len(names)), list(tot_gt_concs.values()), yerr=list(tot_gt_std.values()),
                                 fmt='none', ecolor='k', capsize=2,
                                 elinewidth=1, capthick=1, alpha=0.8, zorder=3)

                if config.save:
                    if not os.path.exists(f'{save_path}comps/{m + add}/'): os.makedirs(f'{save_path}comps/{m + add}/')
                    plt.savefig(f'{save_path}comps/{m + add}/comps_0vs{idx}{"_t" if total else ""}_comp.png', dpi=1000)
                else: plt.show()

        for i in range(1, concs[methods[0]].shape[0]):
            if config.plot_conc_metabs == 'metabs' or config.plot_conc_metabs == 'both':
                plot_comparsion(concs, model.basisObj.names[:model.basisObj.n_metabs], idx=i)
                plot_comparsion(concs_l, model.basisObj.names[:model.basisObj.n_metabs], idx=i, add=' + L')
                plot_comparsion(concs_c, model.basisObj.names[:model.basisObj.n_metabs], idx=i, add=' + WAND')

            if config.plot_conc_metabs == 'total' or config.plot_conc_metabs == 'both':
                plot_comparsion(tot_concs, tot_names, idx=i, total=True)
                plot_comparsion(tot_concs_l, tot_names, idx=i, total=True, add=' + L')
                plot_comparsion(tot_concs_c, tot_names, idx=i, total=True, add=' + WAND')


    # plot the LCModel fits
    if 'LCModel' in methods:
        for idx in range(concs['LCModel'].shape[0]):
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


            fit = read_LCModel_fit(f'{save_path}quant_LCModel/lcm/temp{idx}.coord')
            plot_LCModel_fit(fit, name=f'lcm_{idx}.png')

            fit = read_LCModel_fit(f'{save_path}quant_LCModel/lcm_l/temp{idx}.coord')
            plot_LCModel_fit(fit, name=f'lcm_l_{idx}.png')

            fit = read_LCModel_fit(f'{save_path}quant_LCModel/lcm_c/temp{idx}.coord')
            plot_LCModel_fit(fit, name=f'lcm_c_{idx}.png')


    # check if folder exists and create json file of config
    if os.path.exists(save_path):
        with open(f'{save_path}config.json', 'w') as f:
            f.write(str(config))

