####################################################################################################
#                                          gpu_config.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 14/12/22                                                                                #
#                                                                                                  #
# Purpose: GPU configuration file.                                                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import os
import pandas as pd
import subprocess as sp


#*************************************#
#   get list of gpu based on memory   #
#*************************************#
def get_gpu_memory(verbose=True):
    """
        Retrieve memory allocation information of all gpus.

        @param verbose -- Set True to print output.

        @returns -- The list of available memory for each gpu in MiB.
    """
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

    if verbose:
        df = df = pd.DataFrame({'memory': memory_free_values})
        df.index.name = 'GPU'
        print(df)
    return memory_free_values


#***********************************#
#   set gpu usage based on memory   #
#***********************************#
def set_gpu_usage():
    """
        Select gpu based on memory.

        @returns -- The selected gpu id (with maximum memory available).
    """
    mem = get_gpu_memory()
    sorted_gpu_ids = np.argsort(mem)[::-1]
    gpu_id = sorted_gpu_ids[0]
    print(f'Selected GPU {gpu_id} with {mem[gpu_id]} MiB of memory available')
    return gpu_id
