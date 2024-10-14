####################################################################################################
#                                           wavelets.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/10/23                                                                                #
#                                                                                                  #
# Purpose: Implements (synchrosqueezed) forward and inverse continuous wavelet transforms.         #
#          The implementations are inspired by:                                                    #
#          https://github.com/OverLordGoldDragon/ssqueezepy/blob/master/ssqueezepy                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import torch

from ssqueezepy.utils import (process_scales, logscale_transition_idx, infer_scaletype,
                              p2up, adm_ssq)
from ssqueezepy.wavelets import Wavelet
from ssqueezepy._cwt import _process_gmw_wavelet



#**************************************************************************************************#
#                                           Class CWT                                              #
#**************************************************************************************************#
#                                                                                                  #
# Implements the class for the forward and inverse continuous wavelet transforms (cwt and icwt).   #
# Options for different wavelets and scales are available.                                         #
#                                                                                                  #
#**************************************************************************************************#
class CWT(torch.nn.Module):

    #*************************#
    #   initialize instance   #
    #*************************#
    def __init__(self, signal_length, wavelet='gmw', scales='log-piecewise', nv=32,
                 padtype='reflect', rpadded=False):
        """
            Continuous Wavelet Transform. Uses FFT convolution via frequency-domain
            wavelets matching (padded) input's length.

            @param signal_length -- The length of the signal.
            @param wavelet -- The wavelet to use.
            @param scales -- The scales to use.
            @param nv -- The number of voices (wavelets per octave). Suggested >= 16.
            @param padtype -- Pad scheme to apply on input.
                              `None` -> no padding.
            @param rpadded -- Whether to return padded Wx.
        """
        super().__init__()

        self.nv = nv
        self.padtype = padtype
        self.rpadded = rpadded

        if padtype is not None:
            up, self.n1, self.n2, = p2up(signal_length)
        else:
            up = signal_length

        # process `wavelet`, get its `dtype`
        wavelet = _process_gmw_wavelet(wavelet, l1_norm=True)
        self.wavelet = Wavelet._init_if_not_isinstance(wavelet)
        self.wavelet.N = up

        # process `scales`
        self.scales = process_scales(scales, signal_length, self.wavelet, nv=nv)

        self.Psih = torch.Tensor(self.wavelet.Psih(scale=self.scales, nohalf=False))
        self.Cpsi = adm_ssq(self.wavelet)   # admissibility coefficient


    #*****************#
    #   forward cwt   #
    #*****************#
    def forward(self, x):
        # pad, ensure correct data type
        if self.padtype is not None:
            xp = torch.nn.functional.pad(x, (self.n1, self.n2), mode=self.padtype)
        else:
            xp = x

        # take to freq-domain
        xh = torch.fft.fft(xp, dim=-1).unsqueeze(-2)

        # take CWT
        Psih_xh = self.Psih.to(xh.device) * xh
        Wx = torch.fft.ifft(Psih_xh, dim=-1)

        # handle unpadding, normalization
        if not self.rpadded and self.padtype is not None:
            Wx = Wx[..., self.n1:self.n1 + x.shape[-1]]
        return Wx


    #*****************#
    #   inverse cwt   #
    #*****************#
    def inverse(self, Wx, scales='log-piecewise', nv=None, x_len=None, x_mean=0):
        # prepare for inversion
        na, n = Wx.shape[-2:]
        x_len = x_len or n

        if not isinstance(scales, np.ndarray):
            nv = self.nv  # must match forward's; default to `cwt`'s
            scales = self.scales
            scaletype = 'log-piecewise'
        else:
            scaletype, nv = infer_scaletype(scales)

        # handle piecewise scales case
        # `nv` must be left unspecified, so it's inferred automatically from `scales`
        if scaletype == 'log-piecewise':
            idx = logscale_transition_idx(scales)
            x = self.inverse(Wx[..., :idx, :], scales=scales[:idx], x_len=x_len, x_mean=x_mean)
            x += self.inverse(Wx[..., idx:, :], scales=scales[idx:], x_len=x_len, x_mean=x_mean)
            return x

        # one-integral iCWT (assumes analytic wavelet)
        if scaletype == 'log': x = (2 / self.Cpsi) * np.log(2 ** (1 / nv)) * Wx.real.sum(axis=-2)
        else: x = (2 / self.Cpsi) * (Wx.real / scales).sum(axis=-2)

        x += x_mean
        return x