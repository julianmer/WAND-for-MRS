####################################################################################################
#                                           components.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 21/06/22                                                                                #
#                                                                                                  #
# Purpose: Some helpful functions for generating signal components are defined here.               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np

from scipy.signal import hilbert


#*********************************#
#   create random walk artifact   #
#*********************************#
def randomWalk(waveLength=1024, scale=1, smooth=10, ylim=[-10, 10]):
    """
        Produces a spectrum with a random walk. Inspired by:
        https://stackoverflow.com/questions/29050164/produce-random-wavefunction.

        @param waveLength -- The number of points of the spectrum.
        @param scale -- The y scale of the steps of the walk.
        @param smooth -- The smoothness.
        @param ylim -- The y limits, format [min, max].

        @returns -- The random wave shape from the random walk.
    """
    y = np.random.uniform(ylim[0], ylim[1])  # init randomly between limits
    wave = []
    for _ in range(waveLength):
        step = np.random.normal(scale=scale)
        if y + step >= ylim[0] and y + step <= ylim[1]: y += step
        else: y -= step
        wave.append(y)

    # smoothing
    wave = np.convolve(wave, np.ones((int(smooth),)) / smooth)[(int(smooth) - 1):]
    return wave


#*********************************#
#   create random peak artifact   #
#*********************************#
def randomPeak(waveLength=1024, batch=1, amp=None, pos=None, width=None, phase=None, td=False):
    """
        Produces a spectrum of a random peak.

        @param waveLength -- The number of points of the spectrum.
        @param batch -- The number of peaks to produce.
        @param amp -- The amplitude of the peak.
        @param pos -- The position of the peak.
        @param width -- The width of the peak.
        @param phase -- The phase of the peak.
        @param td -- If True the spectrum is returned in the time domain.

        @returns -- The random wave shape from the random peak (complex).
    """
    if amp is None: amp = np.ones((batch, 1))
    if pos is None: pos = np.ones((batch, 1)) * waveLength // 2
    if width is None: width = np.ones((batch, 1)) * 10
    if phase is None: phase = np.zeros((batch, 1)) + 0.5
    t = np.arange(waveLength)[None, :]

    x = amp * np.exp(- (t - pos) ** 2 / (2 * width ** 2))
    x = hilbert(x.real) * np.exp(- 1j * phase)

    if td: x = np.fft.ifft(x, axis=-1)
    return x

