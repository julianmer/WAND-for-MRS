####################################################################################################
#                                         plotFunctions.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 09/05/22                                                                                #
#                                                                                                  #
# Purpose: Defines functions to visualize results.                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import  matplotlib.colors as clt
import matplotlib.pyplot as plt
import numpy as np
import os


#*****************************#
#   plot spec and scalogram   #
#*****************************#
def plot_signal_and_scalogram(cwtmatr, signal, scales=None, time=None, cmap='plasma',
                              title=None, savePath=None, color='blue', linewidth=2):
    """
        Plots a signal and its Continuous Wavelet Transform (CWT) scalogram
        with precisely aligned x-axes without titles.

        @param cwtmatr -- The CWT scalogram.
        @param signal -- The signal.
        @params scales -- The scales (default None).
        @params time -- The time (default None).
        @params cmap -- The colormap (default 'plasma').
        @params title -- The title of the plot (default None).
        @params savePath -- The path to save the plot (default None).
        @params color -- The color of the signal (default 'blue').
        @params linewidth -- The linewidth of the signal (default 2).
    """
    if scales is None: scales = np.arange(0, cwtmatr.shape[0])
    if time is None: time = np.arange(0, cwtmatr.shape[1])

    fig, axs = plt.subplots(2, 1, figsize=(6, 10))

    # plot the original complex signal
    axs[0].plot(time, signal, color=color, linewidth=linewidth)
    # axs[0].set_ylabel('Real ', fontsize=14)
    # axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    # plot the CWT scalogram
    axs[1].imshow(np.abs(cwtmatr), aspect='auto', cmap=cmap)
    axs[1].set_xlabel('Chemical Shift')
    axs[1].set_ylabel('Scale')

    axs[0].invert_xaxis()
    axs[1].invert_xaxis()

    if title is not None: fig.suptitle(title)

    # plt.tight_layout()
    if savePath is not None:
        plt.savefig(savePath, dpi=300, bbox_inches='tight', transparent=True)


#******************#
#   plot spectra   #
#******************#
def plot_spectra(spectra, imag=False, linewidth=3, savePath=None):
    """
        Plots the given spectra.

        @param spectra -- The spectra to plot.
        @params imag -- If True, the imaginary part is visualized (default False).
        @params linewidth -- The linewidth of the plots (default 3).
        @params save_path -- The path to save the plots (default None).
    """
    if imag: color = 'g'
    else: color = 'b'

    imag = int(imag)

    plt.figure()
    plt.plot(spectra[imag].cpu().detach().numpy(), color, linewidth=linewidth)
    plt.axis('off')
    plt.gca().invert_xaxis()

    # check if folder exists and create it if not
    if savePath is not None:
        if not os.path.exists(savePath): os.makedirs(savePath)
        plt.savefig(savePath, dpi=300, bbox_inches='tight', transparent=True)
    else: plt.show()


#********************#
#   plot scalogram   #
#********************#
def plot_scalogram(cwtmatr, scales=None, time=None, cmap='plasma', title=None, savePath=None):
    """
        Plots a scalogram.

        @param cwtmatr -- The scalogram.
        @params scales -- The scales (default None).
        @params time -- The time (default None).
        @params cmap -- The colormap (default 'plasma').
        @params title -- The title of the plot (default None).
        @params savePath -- The path to save the plot (default None).
    """
    if scales is None: scales = np.arange(0, cwtmatr.shape[0])
    if time is None: time = np.arange(0, cwtmatr.shape[1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.abs(cwtmatr), aspect='auto', cmap=cmap)
    ax.set_xlabel('Chemical Shift')
    ax.set_ylabel('Scale')
    ax.invert_xaxis()

    ax.axis('off')
    # plt.tight_layout()

    # if title is not None: fig.suptitle(title)

    if savePath is not None:
        plt.savefig(savePath, dpi=300, bbox_inches='tight', transparent=True)


#****************************************#
#   visualize input, mask, and outputs   #
#****************************************#
def visualize_pred(xs, cwtmatr, masks, specs, names, imag=False, linewidth=3,
                   batch_idx=[0], metab_idx=[15], save_path=None):
    """
        Visualizes the input, masks, and outputs of the model.

        @param xs -- The input data.
        @param cwtmatr -- The CWT scalogram.
        @param masks -- The masks.
        @param specs -- The output data.
        @param names -- The names of the metabolites.
        @params imag -- If True, the imaginary part is visualized (default False).
        @params linewidth -- The linewidth of the plots (default 3).
        @params batch_idx -- The batch indices (default [0], can pass an empty list [] to visualize all).
        @params metab_idx -- The metabolite indices (default [15], can pass an empty list [] to visualize all).
        @params save_path -- The path to save the plots (default None).
    """
    specs = specs.permute(0, 3, 1, 2)
    masks = masks.permute(0, 4, 1, 2, 3)

    if imag:
        color = 'g'
        cmap = 'viridis'
    else:
        color = 'b'
        cmap = 'plasma'

    imag = int(imag)
    if len(batch_idx) == 0: batch_idx = list(range(masks.shape[0]))
    if len(metab_idx) == 0: metab_idx = list(range(masks.shape[1]))

    # metab_idx, 4 <-> Cr, 15 <-> NAA
    print([names[i] for i in metab_idx])

    def plot_spec(spec, title=None, linewidth=linewidth):
        plt.figure()
        plt.plot(spec, color, linewidth=linewidth)
        plt.axis('off')
        plt.gca().invert_xaxis()
        # if title is not None: plt.title(title)

    # check if folder exists and create it if not
    if save_path is not None:
        if not os.path.exists(save_path): os.makedirs(save_path)

    for i in batch_idx:
        plot_spec(xs[i, imag].cpu().detach().numpy(), title='Input Spectrum')
        if save_path is not None:
            plt.savefig(save_path + f'input_{i}.png', dpi=300, bbox_inches='tight', transparent=True)
        plot_scalogram(cwtmatr[i, imag].cpu().detach().numpy(), title='Input Scalogram', cmap=cmap)
        if save_path is not None:
            plt.savefig(save_path + f'input_scalogram_{i}.png', dpi=300, bbox_inches='tight', transparent=True)

        for j in metab_idx:
            plot_scalogram(masks[i, j, imag].cpu().detach().numpy(), cmap='Greys_r', title=f'Mask {names[j]}')
            if save_path is not None:
                plt.savefig(save_path + f'mask_{i}_{names[j]}.png', dpi=300, bbox_inches='tight', transparent=True)
            plot_scalogram((masks * cwtmatr.unsqueeze(1))[i, j, imag].cpu().detach().numpy(), title=f'Scalogram {names[j]}', cmap=cmap)
            if save_path is not None:
                plt.savefig(save_path + f'scalogram_{i}_{names[j]}.png', dpi=300, bbox_inches='tight', transparent=True)
            plot_spec(specs[i, j, imag].cpu().detach().numpy(), title=f'Output Spectrum {names[j]}')
            if save_path is not None:
                plt.savefig(save_path + f'output_{i}_{names[j]}.png', dpi=300, bbox_inches='tight', transparent=True)

        plot_spec((xs[i, imag] - specs[i, -1, imag]).cpu().detach().numpy(), title='Cleaned Spectrum')
        if save_path is not None:
            plt.savefig(save_path + f'cleaned_{i}.png', dpi=300, bbox_inches='tight', transparent=True)

    if save_path is None: plt.show()


#****************************************#
#   visualize input, mask, and outputs   #
#****************************************#
def visualize_artifact_removal(xs, ys, specs, names, imag=False,
                               batch_idx=[0], metab_idx=[15], save_path=None):
    """
        Visualizes the cleaned and ground truth spectra.

        @param xs -- The input data.
        @param ys -- The ground truth data.
        @param specs -- The output data.
        @param names -- The names of the metabolites.
        @params imag -- If True, the imaginary part is visualized (default False).
        @params batch_idx -- The batch indices (default [0], can pass an empty list [] to visualize all).
        @params metab_idx -- The metabolite indices (default [15], can pass an empty list [] to visualize all).
        @params save_path -- The path to save the plots (default None).
    """
    specs = specs.permute(0, 3, 1, 2)

    if imag:
        color = 'g'
        cmap = 'viridis'
    else:
        color = 'b'
        cmap = 'plasma'

    imag = int(imag)

    # metab_idx, 4 <-> Cr, 15 <-> NAA
    print([names[i] for i in metab_idx])

    # check if folder exists and create it if not
    if save_path is not None:
        if not os.path.exists(save_path): os.makedirs(save_path)

    for i in batch_idx:
        plt.figure()
        if ys is not None:
            plt.plot(ys[i, imag, :, :-1].sum(-1).cpu().detach().numpy(), 'k--', linewidth=3)
        plt.plot((xs[i, imag] - specs[i, -1, imag]).cpu().detach().numpy(), color, linewidth=3)
        plt.axis('off')
        plt.gca().invert_xaxis()

        if save_path is not None:
            plt.savefig(save_path + f'artifact_removal_{i}.png', dpi=300,
                        bbox_inches='tight', transparent=True)

    if save_path is None: plt.show()


#**********************#
#   visualize truths   #
#**********************#
def visualize_truth(y, names, imag=False, batch_idx=[0], metab_idx=[15], save_path=None, linewidth=3):
    """
        Visualizes the true spectra.

        @param y -- The true spectra.
        @param names -- The names of the metabolites.
        @params imag -- If True, the imaginary part is visualized (default False).
        @params batch_idx -- The batch indices (default [0], can pass an empty list [] to visualize all).
        @params metab_idx -- The metabolite indices (default [15], can pass an empty list [] to visualize all).
        @params save_path -- The path to save the plots (default None).
        @params linewidth -- The linewidth of the plots (default 3).
    """
    if imag: color = 'grey'
    else: color = 'k'

    imag = int(imag)
    if len(batch_idx) == 0: batch_idx = list(range(y.shape[0]))
    if len(metab_idx) == 0: metab_idx = list(range(y.shape[-1]))

    if y.shape[-1] == len(names) + 1: names.append('artifacts')

    # metab_idx, 4 <-> Cr, 15 <-> NAA
    print([names[i] for i in metab_idx])

    def plot_spec(spec, title=None):
        plt.figure()
        plt.plot(spec, color, linewidth=linewidth)
        plt.axis('off')
        plt.gca().invert_xaxis()
        # if title is not None: plt.title(title)

    # check if folder exists and create it if not
    if save_path is not None:
        if not os.path.exists(save_path): os.makedirs(save_path)

    for i in batch_idx:
        for j in metab_idx:
            plot_spec(y[i, imag, :, j].cpu().detach().numpy(), title='Target Spectrum')
            plt.axis('off')
            if save_path is not None:
                plt.savefig(save_path + f'target_{i}_{names[j]}.png', dpi=300,
                            bbox_inches='tight', transparent=True)
            else: plt.show()


#********************************#
#   plot spectra and residuals   #
#********************************#
def plot_residuals(specPred, specTrue, bw, cf, ppmAxis=None, ppmRange=(0, 6), shift=2.02, name='', figsize=(7, 4)):
    """
        Plots the given spectra and their residuals in ppm scale.

        @param specPred -- The spectrum prediction to plot.
        @param specTrue -- The true spectrum to plot.
        @param bw -- The bandwidth of the
        @param cf -- The central frequency.
        @param ppmAxis -- The ppm axis to plot (default None).
        @param ppmRange -- The ppm range to plot. If tuple, the ppm axis is computed,
                           if list, the indices are used (as [beg, end]).
        @param shift -- The reference is shift (default: 2 (NAA)).
        @params name -- Title of the plot (default '').
        @params figsize-- The size of the plot (default (7, 4)).
    """
    plt.figure(figsize=figsize)
    plt.xticks(range(ppmRange[0], ppmRange[1], 1))

    if ppmAxis is not None:
        ppmAxis = ppmAxis + 4.68   # shift by water

        beg = min(range(len(ppmAxis)), key=lambda i: abs(ppmAxis[i] - ppmRange[0]))
        end = min(range(len(ppmAxis)), key=lambda i: abs(ppmAxis[i] - ppmRange[1]))

        # frequency shift
        specTrue = np.fft.fftshift(specTrue)
        specPred = np.fft.fftshift(specPred)

        plt.plot(ppmAxis[beg:end], specTrue[beg:end], 'k', linewidth=1.0)
        plt.plot(ppmAxis[beg:end], specPred[beg:end], 'r--', linewidth=1.5)
        plt.plot(ppmAxis[beg:end], specTrue[beg:end] - specPred[beg:end] + 1.4,
                 'b--', linewidth=1.0)

        plt.xlabel('Chemical Shift [ppm]')

    else:
        if type(ppmRange) is tuple:
            l = specTrue.shape[0] // 2
            specTrue = np.concatenate((specTrue[l:], specTrue[:l]))
            specPred = np.concatenate((specPred[l:], specPred[:l]))

            numSamples = specTrue.shape[0]
            reference = np.argmax(specTrue) / numSamples * bw

            # compute ppm axis
            cs = np.array([delta(freq / numSamples * bw, reference, cf * 1e6)
                           for freq in range(numSamples)]) + shift

            beg = min(range(len(cs)), key=lambda i: abs(cs[i] - ppmRange[0]))
            end = min(range(len(cs)), key=lambda i: abs(cs[i] - ppmRange[1]))

            plt.plot(cs[beg:end], specTrue[beg:end], 'k', linewidth=1.0)
            plt.plot(cs[beg:end], specPred[beg:end], 'r--', linewidth=1.5)
            plt.plot(cs[beg:end], specTrue[beg:end] - specPred[beg:end] + 1.4,
                     'b--', linewidth=1.0)

            plt.xlabel('Chemical Shift [ppm]')

        elif type(ppmRange) is list:
            beg, end = ppmRange[0], ppmRange[1]
            plt.plot(specTrue[beg:end], 'k', linewidth=1.0)
            plt.plot(specPred[beg:end], 'r--', linewidth=1.5)
            plt.plot(specTrue[beg:end] - specPred[beg:end] + 1.4,
                     'b--', linewidth=1.0)

            plt.xlabel('Frequency [hz]')

    plt.legend(['Spectrum', 'Prediction', 'Residuals'], loc=2)
    plt.title(name)
    plt.gca().invert_xaxis()


#********************************#
#   plot decomposition signals   #
#********************************#
def plot_decomposition(x, y, names, imag=False, truth=False, batch_idx=[0], metab_idx=[],
                       yLabel=None, ppmRange=(0.5, 4.2), linewidth=3, save_path=None, yLim=True,
                       no_x_axis=True, no_y_axis=True):
    """
    Plots the decomposition of the given signals.

    @param x -- The input data.
    @param y -- The decomposed data.
    @param names -- The names of the metabolites.
    @params imag -- If True, the imaginary part is visualized (default False).
    @params truth -- If True, the ground truth is visualized (default False).
    @params batch_idx -- The batch indices (default [0], can pass an empty list [] to visualize all).
    @params metab_idx -- The metabolite indices (default [], can pass an empty list [] to visualize all).
    @params yLabel -- The label of the y axis (default None).
    @params ppmRange -- The ppm range to plot (default (0.5, 4.2)).
    @params linewidth -- The linewidth of the plot (default 3).
    @params save_path -- The path to save the plots (default None).
    @params yLim -- If True, the y axis limits are set globally (default True).
    @params no_x_axis -- If True, the x axis is not shown (default True).
    @params no_y_axis -- If True, the y axis is not shown (default True).
    """

    if truth:
        if imag: color = 'grey'
        else: color = 'k'
    else:
        if imag: color = 'g'
        else: color = 'b'

    if len(batch_idx) == 0: batch_idx = range(x.shape[0])
    if len(metab_idx) == 0: metab_idx = range(y.shape[-1])

    if 'baseline' in names: names[names.index('baseline')] = 'Baseline'
    if 'artifacts' in names: names[names.index('artifacts')] = 'Artifacts'

    res = x[:, int(imag), :] - y[:, int(imag), :].sum(-1)

    if yLim:
        shift = 1 / 2
        x_max = np.mean(np.max(x[:, int(imag)].detach().cpu().numpy(), axis=1))
        y_max = np.mean(np.max(y[:, int(imag)].detach().cpu().numpy(), axis=1))
        res_max = np.mean(np.max(res.detach().cpu().numpy(), axis=1))

    for i in batch_idx:

        if not yLim:
            shift = 1 / 8
            x_max = np.max(x[i, int(imag)].detach().cpu().numpy())
            y_max = np.max(y[i, int(imag)].detach().cpu().numpy())
            res_max = np.max(res[i].detach().cpu().numpy())
            plt.figure(figsize=(10, 10))
        else:
            if imag: plt.figure(figsize=(10, 60))
            else: plt.figure(figsize=(10, 25))
        plt.plot(x[i, int(imag), :].cpu().numpy(), 'k--', linewidth=linewidth)
        plt.plot(y[i, int(imag), :].sum(-1).detach().cpu().numpy(), color, linewidth=linewidth)

        plt.plot(res[i].detach().cpu().numpy() + x_max + res_max + y_max * shift, color, linewidth=linewidth)
        plt.text(-2, x_max + res_max, 'Reconstr.\nError', fontsize=14, color=color, weight='bold')
        for j in range(y.shape[-1]):
            if j in metab_idx:
                plt.plot(y[i, int(imag), :, j].detach().cpu().numpy() - (j + 3) * y_max * shift,
                         color, linewidth=linewidth)
                plt.text(-2, -(j + 3) * y_max * shift, names[j], fontsize=14, color=color, weight='bold')

        if yLim:
            plt.ylim(np.min(y[:, int(imag), :, -1].detach().cpu().numpy()) -(y.shape[-1] + 3) * y_max * shift,
                     np.max(x[:, int(imag)].detach().cpu().numpy()) + np.max(res.detach().cpu().numpy()))

        plt.legend(frameon=False)

        # invert x axis and add ppm scale
        plt.gca().invert_xaxis()

        if not no_x_axis:
            # # convert the points to ppm
            # ppm = np.round(np.linspace(ppmRange[0], ppmRange[1], x.shape[-1]), 2)
            # plt.xticks(range(0, len(ppm), 50), ppm[::50])
            ppm = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
            points = (np.array(ppm) - 0.5) / (4.2 - 0.5) * x.shape[-1]
            plt.xticks(points, ppm)

            # increase thikness of the bottom spine and of ticks
            plt.gca().spines['bottom'].set_linewidth(linewidth)
            plt.gca().tick_params(axis='x', width=linewidth)

            # also increase the font size of the ticks and x label
            plt.gca().tick_params(axis='x', labelsize=19)
            plt.xlabel('Chemical Shift [ppm]', fontsize=19)
        else:
            plt.xticks([])
            plt.xlabel('')
            plt.gca().spines['bottom'].set_visible(False)

        if not no_y_axis:
            if yLabel is not None: plt.ylabel(yLabel, fontsize=24)
        else:
            plt.yticks([])
            plt.ylabel('')
            plt.gca().spines['left'].set_visible(False)

        # remove top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        # check if folder exists and create it if not
        if save_path is not None:
            if not os.path.exists(save_path): os.makedirs(save_path)
            plt.savefig(f'{save_path}/decomposition_{i}.png', bbox_inches='tight', dpi=1000,
                        transparent=True)
        else:
            plt.show()


#***************************#
#   bar plot of 2 metrics   #
#***************************#
def plot_bars_2metrics(mapeM1, mapeM2, xLabels, yLabel='', label1='', label2='', name='',
                       color1='tab:blue', color2='tab:green', alpha=1.0):
    """
        Plots a bar pot with the mean and std MAPE for each metabolite for 2 models.

        @param mapeM1 -- MAPE of model 1.
        @param mapeM2 -- MAPE of model 2.
        @param xLabels -- The metabolite names.
        @params yLabel -- label of the y axis (default 'Error').
        @params label1 -- Name of the first model (default '').
        @params label2 -- Name of the second model (default '').
        @params name -- Title of the plot (default '').
        @params color1 -- Color of the first model (default 'tab:blue').
        @params color2 -- Color of the second model (default 'tab:green').
        @params alpha -- Alpha value of the bars (default 1.0).
    """
    fig, ax = plt.subplots(figsize=(len(xLabels) * 2 // 3, 4))
    xPos = np.arange(mapeM1.shape[1])

    shift = 0.4
    ax.bar(xPos - shift / 2, np.mean(mapeM1, axis=0), shift, yerr=np.std(mapeM1, axis=0),
           color=color1, align='center', ecolor='black', capsize=4, label=label1, alpha=alpha)
    ax.bar(xPos + shift / 2, np.mean(mapeM2, axis=0), shift, yerr=np.std(mapeM2, axis=0),
           color=color2, align='center', ecolor='black', capsize=4, label=label2, alpha=alpha)

    xPosL = xPos / mapeM1.shape[1] * (mapeM1.shape[1] + 2 * shift) - shift
    ax.plot(xPosL, np.full(xPos.shape, np.mean(mapeM2)), color=color2, linestyle='--', alpha=alpha)
    ax.plot(xPosL, np.full(xPos.shape, np.mean(mapeM1)), color=color1, linestyle='--', alpha=alpha)

    if np.mean(mapeM1) == np.mean(mapeM2):
        ax.plot(xPosL, np.full(xPos.shape, np.mean(mapeM1)), color='black', linestyle='--', alpha=alpha)

    ax.set_ylabel(yLabel)
    ax.set_xticks(xPos)
    ax.set_xticklabels([elem.split()[0] for elem in xLabels])
    # ax.yaxis.grid(True)

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # make text bold
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    leg = ax.legend()
    for text in leg.get_texts():
        text.set_fontweight('bold')
    ax.set_ylabel(yLabel, fontweight='bold')


    plt.legend()
    plt.title(name)
    # plt.tight_layout()


#*********************#
#   plot algorithms   #
#*********************#
def plot_algs_bars(algs, names, xLabel='', yLabel='', colors=None, savePath=None, hlines=True):
    """
        Plots a bar plot of the mean and std of the given algorithms.

        @param algs -- The algorithms to plot.
        @param names -- The names of the metabolites.
        @params xLabel -- The label of the x axis (default '').
        @params yLabel -- The label of the y axis (default '').
        @params colors -- The colors of the algorithms (default None).
        @params savePath -- The path to save the plot (default None).
    """
    if colors is None: colors = plt.cm.tab10(np.linspace(0, 1, len(algs.keys())))
    space = (1 - 0.2) / len(algs.keys())
    if len(names) >= 12: plt.figure(figsize=(len(names) // 2.125, 4))
    else: plt.figure(figsize=(len(names), 4))
    for alg in algs.keys():
        allC = algs[alg]

        # plot mean line
        if hlines:
            plt.hlines(np.mean(allC), -space, len(names)-space, colors=colors[list(algs.keys()).index(alg)],
                       linestyles='--', alpha=0.8, zorder=1)

        plt.bar(np.arange(len(names)) + space * list(algs.keys()).index(alg), np.mean(allC, axis=0),
                width=space, label=alg, color=colors[list(algs.keys()).index(alg)], alpha=0.8, zorder=2)
        plt.errorbar(np.arange(len(names)) + space * list(algs.keys()).index(alg), np.mean(allC, axis=0),
                     yerr=np.std(allC, axis=0), fmt='none', ecolor='k', capsize=2,
                     elinewidth=1, capthick=1, alpha=0.8, zorder=3)

    plt.xticks(np.arange(len(names)) + space * (len(algs.keys()) - 1) / 2, names)

    # set limits
    plt.xlim(- 2 * space, len(names))

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend(frameon=False)
    plt.tight_layout(w_pad=1.0)

    # # place the y-axis on the right
    # plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
    # plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
    # plt.gca().get_yaxis().set_label_position('right')

    # remove top and left spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)

    if savePath is not None: plt.savefig(savePath, dpi=1000, bbox_inches='tight', transparent=True)


#***************************#
#   plot percentage error   #
#***************************#
def plot_percentage_error(perr, names, title='', save=None, aspect='auto'):
    """
        Plots the percentage error of the given metabolites.

        @param perr -- The percentage error.
        @param names -- The names of the metabolites.
        @params title -- The title of the plot (default '').
        @params save -- The path to save the plot (default None).
        @params aspect -- The aspect ratio of the plot (default 'auto').
    """
    # filter
    metabs = ['Ala', 'Asc', 'Asp', 'Cr', 'GABA', 'GPC', 'GSH', 'Glc', 'Gln', 'Glu',
              'Gly', 'Ins', 'Lac', 'NAA', 'NAAG', 'PCho', 'PCr', 'PE', 'Tau', 'sIns']
    metab_idx = [names.index(m) for m in metabs]
    perr = perr[:, metab_idx].T

    if title[:4] == 'CRLB': cmap = 'magma_r'
    else: cmap = 'RdBu'

    # plt.figure(figsize=(5, 4))
    plt.figure(figsize=(4, 3))
    plt.imshow(perr, cmap=cmap, aspect=aspect)
    if 'CB' in title: plt.colorbar()
    plt.title(title)

    plt.xlabel('Dataset')
    plt.gca().set_xticks(range(perr.shape[1]), labels=range(1, perr.shape[1] + 1))
    plt.gca().set_yticks(range(len(metabs)), labels=metabs)

    # only every second tick
    for i, label in enumerate(plt.gca().get_xticklabels()):
        if i % 2: label.set_visible(False)

    # clip colorbar
    if title[:4] == 'CRLB': plt.clim(0, 100)
    else: plt.clim(-90, 90)

    if save: plt.savefig(save, dpi=1000, bbox_inches='tight', transparent=True)
    else: plt.show()