####################################################################################################
#                                           auxiliary.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 21/06/22                                                                                #
#                                                                                                  #
# Purpose: Some helpful functions are defined here.                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import torch


#**********************************#
#   delta scale (chemical shift)   #
#**********************************#
def delta(freq, refFreq, spectFreq):
    """
        Implements the delta scale (chemical shift scale).

        @param freq -- The resonance frequency of the nucleus of interest.
        @param refFreq -- The frequency of the reference (peak).
        @param spectFreq -- The central/spectrometer frequency.

        @returns -- The chemical shift, i.e. the position of the signal to
                    the signal of reference (in parts per million [ppm]).
    """
    return 1e6 * (freq - refFreq) / spectFreq


#**********************************#
#   get the chemical shift scale   #
#**********************************#
def getCS(spectrum, bw, cf, reference=None, shift=2):
    """
        Calculates the chemical shift scale for given spectrum.

        @param spectrum -- The spectrum.
        @param bw -- The bandwidth of the
        @param cf -- The central frequency.
        @param reference -- The reference frequency.
        @param shift -- The reference is shift (default: 2 (NAA)).

        @returns -- The chemical shift, i.e. the positions of the signal to
                    the signal of reference (in parts per million [ppm]).
    """
    numSamples = spectrum.shape[0]

    # if not reference: reference = np.argmax(spectrum) / numSamples * bw
    if not reference: reference = ((numSamples // 2)  / numSamples) * bw

    # compute ppm axis
    cs = np.array([delta((freq / numSamples) * bw, reference, cf * 1e6)
                   for freq in range(numSamples)]) + shift
    return cs


#*********************#
#   absolute errors   #
#*********************#
def aes(concsMB, concsGT):
    """
        Implements the absolute errors (AEs).

        @param concsMB -- The estimated concentrations.
        @param concsGT -- The ground truth concentrations.

        @returns -- The list of AEs [mM].
    """
    return np.abs(np.array(concsMB) - np.array(concsGT))


#********************************#
#   absolute percentage errors   #
#********************************#
def apes(concsMB, concsGT):
    """
        Implements the absolute percentage errors (APEs).

        @param concsMB -- The estimated concentrations.
        @param concsGT -- The ground truth concentrations.

        @returns -- The list of APEs [%].
    """
    return 100 * (np.abs(np.array(concsMB) - np.array(concsGT)) /
                  (np.array(concsGT) + np.finfo(float).eps))


#*******************************#
#   compute total metabolites   #
#*******************************#
def computeTotalMetabolites(thetas, names, sIns=False, GABA=True, GSH=True):
    """
        Computes the total metabolites.

        @param thetas -- The estimated concentrations.
        @param names -- The names of the metabolites.
        @param sIns -- Whether to include sIns in Ins + Gly.
        @param GABA -- Whether to include GABA.
        @param GSH -- Whether to include GSH.

        @returns -- The total metabolites.
    """
    idxNAA = names.index('NAA')
    idxNAAG = names.index('NAAG')
    idxGABA = names.index('GABA')
    idxGln = names.index('Gln')
    idxGlu = names.index('Glu')
    idxGly = names.index('Gly')
    idxGSH = names.index('GSH')
    idxCr = names.index('Cr')
    idxPCr = names.index('PCr')
    idxGPC = names.index('GPC')
    idxPCho = names.index('PCho') if 'PCho' in names else names.index('PCh')
    idxIns = names.index('Ins') if 'Ins' in names else names.index('mI') \
        if 'mI' in names else names.index('mIns')
    idxsIns = names.index('sIns') if 'sIns' in names else names.index('sI') \
        if 'sI' in names else names.index('Scyllo')

    totals = {
        'NAA+NAAG': thetas[..., idxNAA] + thetas[..., idxNAAG],
        'Cr+PCr': thetas[..., idxCr] + thetas[..., idxPCr],
        'Gln+Glu': thetas[..., idxGln] + thetas[..., idxGlu],
        'Ins+Gly': thetas[..., idxIns] + thetas[..., idxGly] + thetas[..., idxsIns],
        'GPC+PCho': thetas[..., idxGPC] + thetas[..., idxPCho],
    }
    if GABA: totals['GABA'] = thetas[..., idxGABA]
    if GSH: totals['GSH'] = thetas[..., idxGSH]
    return totals


#************************#
#   rename metabolites   #
#************************#
def renameMetabolites(concs):
    """
        Renames the metabolites.

        @param concs -- The metabolite concentrations.

        @returns -- The renamed metabolites.
    """
    if 'PCh' in concs: concs['PCho'] = concs.pop('PCh')
    if 'mI' in concs: concs['Ins'] = concs.pop('mI')
    if 'sI' in concs: concs['sIns'] = concs.pop('sI')

    # get all macromolecules and form a total
    macromolecules = [key for key in concs.keys() if 'MM' in key]
    concs['Mac'] = sum([concs[mm] for mm in macromolecules])

    return concs


#***************************#
#   get total metabolites   #
#***************************#
def getTotalMetabolites(thetas, names):
    """
        Gets the total metabolites.

        @param thetas -- The estimated concentrations.
        @param names -- The names of the metabolites.

        @returns -- The total metabolites.
    """
    return {
        'tNAA': thetas[..., names.index('TNAA')],
        'GABA': thetas[..., names.index('GABA')],
        'Glx': thetas[..., names.index('Glx')],
        'tCr': thetas[..., names.index('TCr')],
        'tCho': thetas[..., names.index('TCho')],
        'Ins': thetas[..., names.index('Ins')],
    }


#**********************#
#   scenario indices   #
#**********************#
def challenge_scenario_indices():
    """
        ISMRM 2016 Challenge scenario indices.

        @returns -- A dictionary with the indices of the scenarios.
    """
    return {
        'Lorentzian': [0, 11, 21],  # 1-3
        'Gaussian': [22, 23, 24],  # 4-6
        'No MM': [2],  # 11
        'SNR 1': [3],  # 12
        'SNR 2': [6],  # 15
        'SNR 3': [9],  # 18
        'SNR 4': [0],  # 1
        'SNR 5': [13],  # 21
        'Eddy Currents': [14],  # 22
        'Residual Water': [15, 16],  # 23, 24
        'Tumor-Like': [17, 18, 19, 20],  # 25-28
        'General': [25, 26, 27, 1, 4, 5, 7, 8, 10, 12],  # 7-10, 13, 14, 16, 17, 19, 20
        'Normal': list(range(14)) + list(range(21, 28))   # 0-13, 21-27
    }


#*******************************#
#   minimize function with GD   #
#*******************************#
def general_minimize(func, init, iter=1e4):
    """
        Minimize a function using torch optimizers...

        @param func -- The function to minimize.
        @param init -- The initial guess.
        @param iter -- The number of iterations.

        @returns -- The optimized parameters.
    """
    params = torch.nn.Parameter(init)
    opt = torch.optim.Adam([params], lr=1e-1)
    for _ in range(int(iter)):
        opt.zero_grad()
        loss = func(params)
        loss.backward()
        opt.step()
    return params.detach()


#********************************#
#   minimize function with LSQ   #
#********************************#
def general_minimize_lsq(func, init, iter=1e4):
    """
        Minimize a function using torch optimizers...

        @param func -- The function to minimize.
        @param init -- The initial guess.
        @param iter -- The number of iterations.

        @returns -- The optimized parameters.
    """
    params = torch.nn.Parameter(init)
    opt = torch.optim.LBFGS([params], lr=1e-1)
    for _ in range(int(iter)):
        def closure():
            opt.zero_grad()
            loss = func(params)
            loss.backward()
            return loss
        opt.step(closure)
    return params.detach()


#**********************************#
#   minimize function with scipy   #
#**********************************#
def general_minimize_scipy(func, init, method='TNC', bounds=None, **kwargs):
    """
        Minimize a function using scipy...

        @param func -- The function to minimize.
        @param init -- The initial guess.
        @param method -- The optimization method.
        @param bounds -- The bounds of the optimization.
        @param kwargs -- Additional arguments.

        @returns -- The optimized parameters.
    """
    from scipy.optimize import minimize
    res = minimize(func, init, method=method, bounds=bounds, **kwargs)
    return res.x