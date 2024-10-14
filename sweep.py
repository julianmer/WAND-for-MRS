####################################################################################################
#                                            sweep.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 23/01/23                                                                                #
#                                                                                                  #
# Purpose: Sweep parameter definition for Weights & Biases.                                        #
#                                                                                                  #
####################################################################################################

if __name__ == '__main__':

    #*************#
    #   imports   #
    #*************#
    import pytorch_lightning as pl
    import wandb

    # own
    from train import Pipeline


    #**************************#
    #   eliminate randomness   #
    #**************************#
    pl.seed_everything(42)


    #************************************#
    #   configure sweep and parameters   #
    #************************************#
    sweep_config = {'method': 'grid'}

    metric = {
        'name': 'val_loss',
        'goal': 'minimize',

        'additional_metrics': {
            'name': 'test_loss',
            'goal': 'minimize'
        }
    }

    sweep_parameters = {
        'load_model': {'values': [False]},

        'path2trained': {'values': ['./lightning_logs/sskt2pvt/checkpoints/epoch=0-step=1195264.ckpt']},   # norm rw p

        # 'path2basis': {'values': ['../Data/BasisSets/ISMRM_challenge/press3T_30ms_Philips.BASIS']},  # path to basis set
        'path2basis': {'values': ['../Data/BasisSets/basis_fMRSinPain/']},

        'arch': {'values': ['unet']},
        'specType': {'values': ['fMRSinPain']},
        'basisFmt': {'values': ['fMRSinPain']},  # '', 'cha_philips', 'fMRSinPain'
        'dataType': {'values': ['norm']},
        'learning': {'values': [0.0001]},
        'dropout': {'values': [0.1]},
        'l1_reg': {'values': [0.0]},
        'l2_reg': {'values': [0.0]},
    }

    sweep_config['name'] = 'model_sweep'   # sweep name
    sweep_config['parameters'] = sweep_parameters   # add parameters to sweep
    sweep_config['metric']= metric    # add metric to sweep

    # create sweep ID and name project
    wandb.login(key='')   # add your own key here
    sweep_id = wandb.sweep(sweep_config, project='', entity='')   # add project name...
                                                                  # and entity here
    # training the model
    pipeline = Pipeline()
    wandb.agent(sweep_id, pipeline.main)
