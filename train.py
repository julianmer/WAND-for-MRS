####################################################################################################
#                                             train.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/12/22                                                                                #
#                                                                                                  #
# Purpose: Train and evaluate neural models implemented in Pytorch Lightning.                      #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import numpy as np
import pytorch_lightning as pl
import shutup; shutup.please()   # shut up warnings
import torch
import wandb

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

# own
from utils.gpu_config import set_gpu_usage



#**************************************************************************************************#
#                                             Pipeline                                             #
#**************************************************************************************************#
#                                                                                                  #
# The pipeline allowing to load, augment, train, and test methods.                                 #
#                                                                                                  #
#**************************************************************************************************#
class Pipeline():

    #***************#
    #   main init   #
    #***************#
    def __init__(self):
        pl.seed_everything(42)

        self.default_config = {
            'path2trained': '',   # path to a trained model
            'path2basis': '../Data/BasisSets/ISMRM_challenge/basisset_JMRUI/',  # path to basis set

            'model': 'wand',   # 'wand', ...
            'loss': 'mse',  # 'mse', ...

            # for nn models
            'arch': 'unet',   # 'unet', ...
            'specType': 'synth',  # 'auto', 'synth', 'invivo', 'fMRSinPain', 'all'
            'ppmlim': (0.5, 4.2),   # ppm limits for the spectra (used if specType='auto')
            'basisFmt': '',  # '', 'cha_philips', 'fMRSinPain'

            'dataType': 'std',   # 'clean', 'std', 'std_rw', 'std_rw_p', 'custom', ...

            'val_size': 1024,   # number of validation samples
                                # (simulated new validation set each val_check_interval)

            'max_epochs': -1,  # limit number of epochs (not possible with generator)
            'max_steps': -1,  # limit number of steps/batches (useful with generator)

            'val_check_interval': 256,   # None, if fixed training set
            'check_val_every_n_epoch': None,   # None, if trained with generator
            'trueBatch':16,   # accumulates the gradients over trueBatch/batch
            'batch': 16,
            'learning': 1e-3,
            'dropout': 0.0,
            'l1_reg': 0.0,
            'l2_reg': 0.0,

            'num_workers': 4,   # adjust to roughly 4 times the number GPUs
            'shuffle': True,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'gpu_selection': set_gpu_usage() if torch.cuda.is_available() else 0,

            'load_model': False,   # load model from path2trained
            'skip_train': False,   # skip the training procedure
        }

        # limit number of threads if run is CPU heavy
        torch.set_num_threads(self.default_config['num_workers'])

        # set device
        if str(self.default_config['device']) == 'cuda':
            self.default_config['device'] = torch.device(self.default_config['gpu_selection'])


    #**********************************#
    #   switch model based on config   #
    #**********************************#
    def getModel(self, config):
        if config.model.lower() == 'wand':
            from frameworks.frameworkWAND import FrameworkWAND
            model = FrameworkWAND(basis_dir=config.path2basis,
                                  val_size=config.val_size,
                                  batch=config.batch,
                                  lr=config.learning,
                                  reg_l1=config.l1_reg,
                                  reg_l2=config.l2_reg,
                                  dropout=config.dropout,
                                  model=config.arch,
                                  specType=config.specType,
                                  basisFmt=config.basisFmt,
                                  loss=config.loss,
                                  dataType=config.dataType,
                                  ppmlim=config.ppmlim,
                                  )
            model.to(config.device)

        else:
            raise ValueError('model %s is not recognized' % config.model)
        return model


    #********************************#
    #   main pipeline for training   #
    #********************************#
    def main(self, config=None):
        # wandb init
        if (config is None) or config['online']:
            wandb.init(config=config)
            wandb_logger = WandbLogger()
        else:
            wandb.init(mode='disabled', config=config)
            wandb_logger = WandbLogger(save_dir='./lightning_logs/', offline=True)
        config = wandb.config

        # combine default configs and wandb config
        parser = argparse.ArgumentParser()
        for keys in self.default_config:
            parser.add_argument('--' + keys, default=self.default_config[keys],
                                type=type(self.default_config[keys]))
        args = parser.parse_known_args()[0]
        config.update(args, allow_val_change=False)

        # model inits
        self.model = self.getModel(config)

        # callbacks, etc.
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True)
        early_stop_callback = EarlyStopping(monitor='val_loss', mode='min',
                                            min_delta=0.0, patience=10)
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # loading model...
        if config.load_model and config.model == 'wand':
            self.model = self.model.load_from_checkpoint(config.path2trained,
                                                         basis_dir=config.path2basis,
                                                         val_size=config.val_size,
                                                         batch=config.batch,
                                                         lr=config.learning,
                                                         reg_l1=config.l1_reg,
                                                         reg_l2=config.l2_reg,
                                                         dropout=config.dropout,
                                                         model=config.arch,
                                                         specType=config.specType,
                                                         basisFmt=config.basisFmt,
                                                         loss=config.loss,
                                                         dataType=config.dataType,
                                                         ppmlim=config.ppmlim,
                                                         map_location=config.device,
                                                         strict=False)
        # ...train model
        if not config.skip_train:
            if torch.cuda.is_available():  # gpu acceleration
                try: torch.set_float32_matmul_precision('medium')    # matrix multiplications
                except: print('bfloat16 for matmul not supported')   # use the bfloat16

                trainer = pl.Trainer(max_epochs=config.max_epochs,
                                     max_steps=config.max_steps,
                                     accelerator='gpu',
                                     devices=[config.gpu_selection],  # select gpu by idx
                                     logger=wandb_logger,
                                     callbacks=[checkpoint_callback, lr_monitor],  # , early_stop_callback],
                                     log_every_n_steps=10,
                                     accumulate_grad_batches=config.trueBatch // config.batch,
                                     val_check_interval=config.val_check_interval,
                                     )
            else:
                trainer = pl.Trainer(max_epochs=config.max_epochs,
                                     max_steps=config.max_steps,
                                     logger=wandb_logger,
                                     callbacks=[checkpoint_callback, lr_monitor],  # , early_stop_callback],
                                     log_every_n_steps=10,
                                     accumulate_grad_batches=config.trueBatch // config.batch,
                                     val_check_interval=config.val_check_interval,
                                     )

            trainer.fit(self.model)

            # loading best model
            self.model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                            basis_dir=config.path2basis)

        wandb.finish()
        self.config = config


#**********#
#   main   #
#**********#
if __name__ == '__main__':
    Pipeline().main({'online': False})

