####################################################################################################
#                                          simulation.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 15/12/22                                                                                #
#                                                                                                  #
# Purpose: Simulate a corpus of MRS data.                                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

# own
from simulation.simulationDefs import params, stdConcs


#*********************#
#   draw parameters   #
#*********************#
def simulateParam(basis, batch, params=params, concs=stdConcs):
    """
    Function to simulate SVS parameters.

    @param basis -- The basis set of metabolites to simulate as (FSL) MRS object.
    @param batch -- The number of samples.
    @param params -- The simulation parameters in form of a dictionary,
                     if not given internal parameter configuration will be used.
    @param concs -- The concentration ranges of the metabolites in form of a dictionary,
                    if not given standard concentrations will be used.

    @returns -- The parameters.
    """
    if params['dist'] == 'unif':
        dist = np.random.uniform
    elif params['dist'] == 'normal':
        dist = np.random.normal

    # get metabolite concentrations
    randomConc = {}
    for name in basis.names:
        cName = name.split('.')[0]   # remove format ending (e.g. 'Ace.raw' -> 'Ace')

        #  draw randomly from range
        randomConc[name] = dist(concs[cName]['low_limit'], concs[cName]['up_limit'], batch)

    gamma = dist(params['broadening'][0][0], params['broadening'][1][0], batch)
    sigma = dist(params['broadening'][0][1], params['broadening'][1][1], batch)
    shifting = dist(params['shifting'][0], params['shifting'][1], batch)
    phi0 = dist(params['phi0'][0], params['phi0'][1], batch)
    phi1 = dist(params['phi1'][0], params['phi1'][1], batch)

    theta = np.array(list(randomConc.values()))
    theta = np.concatenate((theta, gamma[np.newaxis, :]))
    theta = np.concatenate((theta, sigma[np.newaxis, :]))
    theta = np.concatenate((theta, shifting[np.newaxis, :]))
    theta = np.concatenate((theta, phi0[np.newaxis, :]))
    theta = np.concatenate((theta, phi1[np.newaxis, :]))

    for i in range(len(params['baseline'][0])):
        theta = np.concatenate((theta, dist(params['baseline'][0][i],
                                            params['baseline'][1][i], batch)[np.newaxis, :]))

    noise = np.random.normal(params['noise'][0], params['noise'][1], (batch, basis.fids.shape[0])) + \
            1j * np.random.normal(params['noise'][0], params['noise'][1], (batch, basis.fids.shape[0]))

    return np.swapaxes(theta, 0, 1), noise


#*************************************#
#   draw parameters for random walk   #
#*************************************#
def simulateRW(batch, params=params):
    """
    Function to simulate random walk parameters.

    @param batch -- The number of samples.
    @param params -- The simulation parameters in form of a dictionary,
                     if not given internal parameter configuration will be used.

    @returns -- The parameters.
    """
    if params['dist'] == 'unif':
        dist = np.random.uniform
    elif params['dist'] == 'normal':
        dist = np.random.normal

    scale = dist(params['scale'][0], params['scale'][1], batch)
    smooth = dist(params['smooth'][0], params['smooth'][1], batch)
    lowLimit = dist(params['limits'][0][0], params['limits'][0][1], batch)
    highLimit = dist(params['limits'][1][0], params['limits'][1][1], batch)
    return scale, smooth, lowLimit, highLimit


#**************************************#
#   draw parameters for random peaks   #
#**************************************#
def simulatePeaks(basis, batch, params=params):
    """
    Function to simulate peaks parameters.

    @param basis -- The basis set of metabolites to simulate as (FSL) MRS object.
    @param batch -- The number of samples.
    @param params -- The simulation parameters in form of a dictionary,
                     if not given internal parameter configuration will be used.

    @returns -- The parameters.
    """
    if params['dist'] == 'unif':
        dist = np.random.uniform
    elif params['dist'] == 'normal':
        dist = np.random.normal

    amps = dist(params['peakAmp'][0], params['peakAmp'][1], batch)[:, np.newaxis]
    widths = dist(params['peakWidth'][0], params['peakWidth'][1], batch)[:, np.newaxis]
    phases = dist(params['peakPhase'][0], params['peakPhase'][1], batch)[:, np.newaxis]
    return amps, widths, phases

