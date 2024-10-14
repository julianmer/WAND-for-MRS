####################################################################################################
#                                        frameworkWAND.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/12/22                                                                                #
#                                                                                                  #
# Purpose: Optimization framework consisting of neural models as well as training and              #
#          testing loops.                                                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from tqdm import tqdm


# own
from frameworks.framework import Framework
from frameworks.nnModels import *
from simulation.dataModules import SynthDataModule, ChallengeDataModule
from simulation.sigModels import VoigtModel
from utils.processing import processBasis, processSpectra
from utils.wavelets import CWT



#**************************************************************************************************#
#                                          Class Framework                                         #
#**************************************************************************************************#
#                                                                                                  #
# The framework allowing to define, train, and test data-driven models.                            #
#                                                                                                  #
#**************************************************************************************************#
class FrameworkWAND(Framework, pl.LightningModule):
    def __init__(self, basis_dir, val_size=1000, batch=16, lr=1e-3, reg_l1=0.0, reg_l2=0.0,
                 dropout=0.0, model='mlp', specType='synth', basisFmt='', loss='mse',
                 dataType='none', ppmlim=None, domain='freq', logPlot=False):
        Framework.__init__(self, basis_dir,
                           basisFmt=basisFmt, specType=specType, dataType=dataType, ppmlim=ppmlim)
        pl.LightningModule.__init__(self)
        # self.save_hyperparameters(ignore=['net'])

        self.valSize = val_size   # validation set size
        self.batchSize = batch
        self.lr = lr
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.dropout = dropout

        self.basis = processBasis(self.basisObj.fids)
        self.sigModel = VoigtModel(basis=self.basis, first=self.first, last=self.last,
                                   t=self.basisObj.t, f=self.basisObj.f)

        self.dataSynth = SynthDataModule(basis_dir=basis_dir,
                                         nums_test=-1,
                                         sigModel=self.sigModel,
                                         params=self.ps,
                                         concs=self.concs,
                                         basisFmt=basisFmt,
                                         specType=specType,
                                         )

        self.lossType = loss
        self.domain = domain
        self.logPlot = logPlot

        self.channels = 2   # real and imaginary part
        self.classes = self.basis.shape[1] + 1   # number of metabolites + additional as below
        self.names = self.basisObj.names
        if self.classes >= self.basis.shape[1] + 1: self.names.append('baseline')
        if self.classes >= self.basis.shape[1] + 2: self.names.append('noise')
        if self.classes >= self.basis.shape[1] + 3: self.names.append('artifacts')

        # the decomposition model
        self.net = self.setModel(model=model)

        # the mask method
        self.maskMethod = 'softmax'   # 'softmax', 'sigmoid', 'max'

        # an artifact mask is added to the output (inverse from all other masks)
        self.artifactMask = True
        if self.artifactMask:
            self.classes += 1
            self.names.append('artifacts')

        # the continuous wavelet transform and default parameters
        self.wavelet = ('morlet', {'mu': 14})
        self.nv = 32

        self.cwt = CWT(self.last - self.first, wavelet=self.wavelet,
                       scales='log-piecewise', nv=self.nv, padtype=None)


    #***********************#
    #   configure network   #
    #***********************#
    def setModel(self, model):
        if model.lower() == 'unet':
            net = UNet(n_channels=self.channels, n_classes=self.channels * self.classes,
                       dropout=self.dropout)
        elif model.lower() == 'reunet':
            net = ReUNet(n_channels=self.channels, n_classes=self.channels * self.classes,
                         dropout=self.dropout)
        else: raise ValueError(f'Unknown model: {model}')
        return net.to(self.device)


    #*************************#
    #   normalize the masks   #
    #*************************#
    def normalizeMasks(self, masks, method='softmax'):
        if method == 'softmax':
            masks = torch.softmax(masks, dim=1)  # constrain to sum to 1
        elif method == 'sigmoid':
            masks = torch.sigmoid(masks)
        elif method == 'max':
            norm = masks.sum((1, 3, 4), keepdim=True) + torch.finfo(torch.float32).eps
            masks = masks / norm
        else: raise ValueError(f'Unknown method: {method}')
        return masks


    #**********************#
    #   prepare the data   #
    #**********************#
    def prepareData(self, x=None, y=None, t=None, return_mean=False):
        if x is not None:
            mean_x = x[:, :, self.first:self.last:self.skip].mean(dim=-1, keepdim=True)
            x -= mean_x
        if y is not None:
            mean_y = y[:, :, self.first:self.last:self.skip].mean(dim=-2, keepdim=True)
            y -= mean_y
        if return_mean: return x, y, t, mean_x, mean_y, None
        else: return x, y, t


    #********************#
    #   domain changes   #
    #********************#
    def domainChange(self, x, forward=True):
        if forward: x = torch.fft.fft(x[..., 0, :] + 1j * x[..., 1, :], dim=-1)
        else: x = torch.fft.ifft(x[..., 0, :] + 1j * x[..., 1, :], dim=-1)
        x = torch.stack((x.real, x.imag), dim=-2)
        return x


    #*********************#
    #   input to output   #
    #*********************#
    def forward(self, x):
        if self.domain[:4] == 'time': x = self.domainChange(x, forward=False)

        xs = x[:, :, self.first:self.last:self.skip]
        cwtmatr = self.cwt.forward(xs)
        masks = self.net(cwtmatr.abs().float())
        masks = masks.view(masks.shape[0], self.classes - int(self.artifactMask),
                           self.channels, masks.shape[-2], masks.shape[-1])

        if self.artifactMask:
            art_mask_re = 1 - masks[:, :, 0].sum(dim=1, keepdim=True)
            art_mask_im = 1 - masks[:, :, 1].sum(dim=1, keepdim=True)
            art_mask = torch.stack((art_mask_re, art_mask_im), dim=2)
            masks = torch.cat((masks, art_mask), dim=1)

        masks = self.normalizeMasks(masks, method=self.maskMethod)
        specs = self.cwt.inverse(masks * cwtmatr.unsqueeze(1)).float()

        if self.domain[:4] == 'time' and not self.domain == 'time-time':
            specs = self.domainChange(specs, forward=True)

        return specs.permute(0, 2, 3, 1), (cwtmatr, masks.permute(0, 2, 3, 4, 1))


    #**********************#
    #   loss calculation   #
    #**********************#
    def loss(self, x, y, y_hat, type='mae', norm=False):
        if type == 'mse':

            if self.domain == 'time-time':
                y = self.domainChange(y.permute(0, 3, 1, 2), forward=False)
                y = y.permute(0, 2, 3, 1)

            y = y[:, :, self.first:self.last:self.skip].float()

            # add zero artifact channel(s)
            if y.shape[-1] < y_hat.shape[-1]:
                zeros = torch.zeros(y.shape[0], y.shape[1], y.shape[2],
                                    y.shape[-1] - y_hat.shape[-1], device=y.device)
                y = torch.cat((y, zeros), dim=-1)

            # remove artifact channel
            elif y.shape[-1] > y_hat.shape[-1]:
                y = y[..., :y_hat.shape[-1]]

            # normalize
            if norm:
                y = y / torch.abs(y).max(dim=-1)[0].unsqueeze(-1)
                y_hat = y_hat / torch.abs(y_hat).max(dim=-1)[0].unsqueeze(-1)

            err = torch.nn.MSELoss()(y_hat, y)
            return err
        else:
            print('SELECT A LOSS')
            return None


    #*******************************#
    #   defines the training loop   #
    #*******************************#
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        x, y, t = batch
        x, y, t = self.prepareData(x, y, t)
        y_hat, ms = self.forward(x)

        loss = self.loss(x, y, y_hat, type=self.lossType)
        self.log("train_loss", loss, prog_bar=True)

        # regularization
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        loss = loss + self.reg_l1 * l1_norm + self.reg_l2 * l2_norm

         # log plot to wandb
        if self.logPlot and batch_idx % 1000 == 0:
            plt.figure()
            plt.plot(y_hat[0, 0, :, 4].cpu().detach().numpy(),
                     'b--', linewidth=3, label='Predicted')
            plt.plot(y[0, 0, self.first:self.last:self.skip, 4].cpu().detach().numpy(),
                     'k', linewidth=3, label='True')
            plt.legend()
            plt.title(self.names[4] + 'Validation Spectrum' + f' {batch_idx}')
            wandb.log({'figure': plt})

            plt.figure()
            plt.plot(y_hat[0, 0, :, -1].cpu().detach().numpy(),
                     'b--', linewidth=3, label='Predicted')
            plt.plot(y[0, 0, self.first:self.last:self.skip, -1].cpu().detach().numpy(),
                     'k', linewidth=3, label='True')
            plt.legend()
            plt.title(self.names[-1] + 'Validation Spectrum' + f' {batch_idx}')
            wandb.log({'figure ar': plt})
        return loss


    #*********************************#
    #   defines the validation loop   #
    #*********************************#
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y, t = batch
        x, y, t = self.prepareData(x, y, t)
        y_hat, ms = self.forward(x)
        loss = self.loss(x, y, y_hat, type=self.lossType)
        self.log("val_loss", loss, prog_bar=True)


    #***************************#
    #   defines the test loop   #
    #***************************#
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y, t = batch
        x, y, t = self.prepareData(x, y, t)
        y_hat, ms = self.forward(x)
        loss = self.loss(x, y, y_hat, type=self.lossType)
        self.log("test_loss", loss)


    #**************************#
    #   training data loader   #
    #**************************#
    def train_dataloader(self):
        while True:  # ad-hoc simulation
            yield self.dataSynth.get_batch(self.batchSize)


    #****************************#
    #   validation data loader   #
    #****************************#
    def val_dataloader(self):
        data = []
        for _ in range(self.valSize):
            x, y, t = self.dataSynth.get_batch(1)
            data.append([x[0], y[0], t[0]])
        return DataLoader(data, batch_size=self.batchSize)


    #*****************************#
    #   configure the optimizer   #
    #*****************************#
    def configure_optimizers(self):        
        params = [param for param in self.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        return optimizer