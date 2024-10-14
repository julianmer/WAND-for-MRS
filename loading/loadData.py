####################################################################################################
#                                            loadData.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 20/06/22                                                                                #
#                                                                                                  #
# Purpose: Load various formats and data types of MRS and MRSI data.                               #
#                                                                                                  #
#          Inspired by [1]: https://github.com/wtclarke/mrs_nifti_standard                         #
#                                                                                                  #
#          [1] Clarke, W.T., Mikkelsen, M., Oeltzschner, G., Bell, T.K., Shamaei, A.,              #
#              Soher, B.J., Emir, U.E., & Wilson, M. (2021). NIfTI-MRS: A standard format          #
#              for magnetic resonance spectroscopic data. bioRxiv.                                 #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import os
import pandas as pd

from fsl_mrs.core import mrs
from fsl_mrs.utils import mrs_io

from spec2nii.Philips.philips import read_sdat, read_spar
from spec2nii.Philips.philips_data_list import _read_data, _read_list, read_data_list_pair

# own
from loading.dicom import loadDICOM, loadDICOMDIR
from loading.lcmodel import read_LCModel_raw
from loading.philips import read_Philips_data


#************************#
#   loading MyMRS data   #
#************************#
def load_MyMRS_data(path2data):
    """
    Load a data set in the MyMRS format.

    @param path2data -- The path to the data set.

    @returns -- The MyMRS objects.
    """
    return pd.read_pickle(path2data)


#****************************#
#   loading MyMRS data set   #
#****************************#
def load_MyMRS_dataSet(path2data, num=None):
    """
    Load a corpus of num data sets in the MyMRS format.

    @param path2data -- The path to the data sets.
    @param num -- The number of sets to loading (if possible), defaults to all.

    @returns -- The list of MyMRS objects.
    """
    dataNames = sorted(os.listdir(path2data))
    data = []
    for name in dataNames[:num]:
        myMRS = pd.read_pickle(path2data + '/' + name)
        data.append(myMRS)

    return data


#*****************************#
#   loading data as FSL-MRS   #
#*****************************#
def loadDataAsFSL(path2data, params=None, bw=None, cf=None, fmt=None):
    """
    Main function to loading various types data formats.
    The bw, cf need to be specified for certain data formats.

    @param path2data -- The path to the data set.
    @param params -- The parameters describing the basis (might be easier
                     than separately passing bw and cf).
    @param bw -- The bandwidth of the basis used.
    @param cf -- The central frequency of the basis used.
    @param fmt -- The file format (if not given it is inferred).
                  Attention: text .txt files are treated as JMRUI .txt file
                             if the format fmt='text' is not specified.

    @returns -- The FSL-MRS MRS object.
    """
    if params: bw, cf = params['bandwidth'], params['centralFrequency']

    if fmt:
        # test for given format
        fmts = ['mymrs', 'jmrui', 'lcmodel', 'text', 'nifti', 'philips', 'ge',
                'siemens', 'dicom', 'dicomdir']
        if fmt.lower() not in fmts:
            print(f'File format should be in {fmts}, but {fmt} was given.')
    else:
        # ... or find file format ...
        fmt = path2data.split(os.extsep, -1)[-1].lower()

    # ... and match
    if fmt == 'mymrs' or fmt == 'pickle':
        return load_MyMRS_data(path2data).toFSLMRS()

    elif fmt == 'jmrui' or fmt == 'txt':
        return load_JMRUI_data(path2data)

    elif fmt == 'lcmodel' or fmt == 'raw' or fmt == 'h2o':
        return load_LCModel_data(path2data, bw, cf)

    elif fmt == 'text' or fmt == 'txt':
        return load_Text_data(path2data, bw, cf)

    elif fmt == 'nifti' or fmt == 'nii' or fmt == 'nii.gz':
        return load_NIfTI_data(path2data)

    elif fmt == 'philips' or fmt == 'sdat' or fmt == 'spar':
        # only loading when SDAT file is given (ignore for SPAR)
        if path2data.split(os.extsep)[-1].lower() == 'sdat':
            return load_SDATSPAR_data(path2data)

    elif fmt == 'datalist' or fmt == 'list' or fmt == 'data':
        # only loading when DATA file is given (ignore for LIST)
        if path2data.split(os.extsep)[-1].lower() == 'data':
            return load_DATALIST_data(path2data, bw, cf)

    elif fmt == 'ge' or fmt == '7':
        return load_GE7_data(path2data)

    elif fmt == 'siemens' or fmt == 'dat':
        return load_SIEMENS_data(path2data)

    elif fmt == 'dicom' or (fmt == path2data.lower() and fmt[-3:] != 'dir'):
        return load_DICOM_data(path2data)

    # !!! DICOMDIR returns list
    elif fmt == 'dicomdir' or fmt == path2data.lower():
        return load_DICOMDIR_dataSets(path2data)

    else:
        print('-------------- Loading failed! --------------\n'
              ' Invalid data type or not implemented yet... \n'
              'File: ' + path2data)


#**********************************#
#   loading data sets as FSL-MRS   #
#**********************************#
def loadDataSetsAsFSL(path2data, num=None, params=None, bw=None, cf=None, fmt=None, skip=True):
    """
    Function wrapper of loadDataAsFSL to loading batches of data.

    @param path2data -- The path to the data sets dir.
    @param num -- The number of sets to loading (if possible), defaults to all.
    @param params -- The parameters describing the basis (might be easier
                     than separately passing bw and cf).
    @param bw -- The bandwidth of the basis used.
    @param cf -- The central frequency of the basis used.
    @param fmt -- The file format (if not given it is inferred).
                  Attention: text .txt files are treated as JMRUI .txt file
                             if the format fmt='text' is not specified.
    @param skip -- If True will skip over invalid/unloadable elements within the directory.

    @returns -- The list of FSL-MRS MRS objects.
    """
    # DICOMDIR
    if path2data[-3:].lower() == 'dir':
        return load_DICOMDIR_dataSets(path2data)

    dataNames = sorted(os.listdir(path2data))

    data = []
    for name in dataNames[:num]:

        # loading file with name
        d = loadDataAsFSL(path2data + '/' + name, params, bw, cf, fmt)

        # if skip, then skip over None entries
        if (skip and not isinstance(d, type(None))) or not skip:
            data.append(d)

    return data


#************************#
#   loading NIfTI data   #
#************************#
def load_NIfTI_data(path2data):
    """
    Load a data set in the NIfTI format.

    @param path2data -- The path to the data set.

    @returns -- The FSL-MRS object.
    """
    return mrs_io.read_FID(path2data).mrs()


#****************************#
#   loading SDAT SPAR data   #
#****************************#
def load_SDATSPAR_data(path2data):
    """
    Load a data set in the SDAT SPAR format.

    @param path2data -- The path to the sdat file.

    @returns -- The FSL-MRS object.
    """
    try:
        params = read_spar(path2data[:-4] + 'SPAR')
        data = read_sdat(path2data[:-4] + 'SDAT',
                         params['samples'],
                         params['rows'])
    except:
        print(f'Make sure SDAT and SPAR are both located at {path2data}.')

    return mrs.MRS(FID=data,
                   cf=params['synthesizer_frequency'] / 10**6,
                   bw=params['sample_frequency'])


#****************************#
#   loading DATA LIST data   #
#****************************#
def load_DATALIST_data(path2data, bw, cf):
    """
    Load a data set in the DATA LIST format.
    # TODO: processing.py (channel and average combining) is not implemented yet!

    @param path2data -- The path to the data set.
    @param bw -- The bandwidth of the data.
    @param cf -- The central frequency of the data.

    @returns -- The FSL-MRS object.
    """
    df, num_dict, coord_dict, os_dict = _read_list(path2data[:-4] + 'list')
    sorted_data_dict = read_Philips_data(path2data[:-4] + 'data', df)

    fids = sorted_data_dict['STD_0'].squeeze()
    h2o = sorted_data_dict['STD_1'].squeeze()

    if len(fids.shape) > 4: raise ValueError('More than 4 dims -> MRSI data not supported yet!')

    # combine channels
    fids = fids.sum(axis=1)
    h2o = h2o.sum(axis=1)

    # combine averages
    fids = fids.sum(axis=1)
    h2o = h2o.sum(axis=1)
    return mrs.MRS(FID=fids, cf=cf, bw=bw, H2O=h2o)


#************************#
#   loading GE .7 data   #
#************************#
def load_GE7_data(path2data):
    """
    Load a data set in the GE .7 format.
    TODO: implement

    @param path2data -- The path to the data set.

    @returns -- The FSL-MRS object.
    """
    print('NOT IMPLEMENTED!')
    return None


#**************************#
#   loading SIEMENS data   #
#**************************#
def load_SIEMENS_data(path2data):
    """
    Load a data set in the SIEMENS .dat format.
    TODO: implement

    @param path2data -- The path to the data set.

    @returns -- The FSL-MRS object.
    """
    print('NOT IMPLEMENTED!')
    return None


#************************#
#   loading JMRUI data   #
#************************#
def load_JMRUI_data(path2data):
    """
    Load a data set in the JMRUI format.

    @param path2data -- The path to the data set.

    @returns -- The FSL-MRS object.
    """
    return mrs_io.read_FID(path2data).mrs()


#**************************#
#   loading LCModel data   #
#**************************#
def load_LCModel_data(path2data, bw=None, cf=None):
    """
    Load a data set in the LCModel format (.RAW, .raw, and .H2O).

    @param path2data -- The path to the data set.
    @param bw -- The bandwidth of the data.
    @param cf -- The central frequency of the data.

    @returns -- The FSL-MRS object.
    """
    data, header = read_LCModel_raw(path2data)

    try: bw, cf = header['bandwidth'], header['centralFrequency']
    except:
        if not (bw and cf):
            print('Could not loading header, '
                  'please specify both bandwidth and central frequency!')

    return mrs.MRS(FID=data, cf=cf, bw=bw)


#***********************#
#   loading text data   #
#***********************#
def load_Text_data(path2data, bw, cf):
    """
    Loads plain text file data (4 channels -> WS real, WS imag, nWS real, nWS imag).

    @param path2data -- The path to the data set.
    @param bw -- The bandwidth of the data.
    @param cf -- The central frequency of the data.

    @returns -- The FSL-MRS object.
    """
    data = np.loadtxt(path2data)

    return mrs.MRS(FID=data[:, 0] + 1j * data[:, 1], cf=cf, bw=bw,
                   H2O=data[:, 2] + 1j * data[:, 3])


#*************************#
#   loading DICOM files   #
#*************************#
def load_DICOM_data(path2data):
    """
    Load data set in the DICOM format (of Philips).

    @param path2data -- The path to the directory.

    @returns -- The FSL-MRS object(s).
    """
    data, dataRef, header = loadDICOM(path2data)

    if isinstance(data, type(None)):
        return None

    return mrs.MRS(FID=data, cf=header['centralFrequency'],
                   bw=header['bandwidth'], H2O=dataRef)


#*****************************#
#   loading DICOM directory   #
#*****************************#
def load_DICOMDIR_dataSets(path2data):
    """
    Load data sets in the DICOMDIR format (of Philips).

    @param path2data -- The path to the directory.

    @returns -- The FSL-MRS object(s).
    """
    data, dataRef, headers = loadDICOMDIR(path2data)

    sets = []
    for i, d in enumerate(data):
        sets.append(mrs.MRS(FID=d, cf=headers[i]['centralFrequency'],
                            bw=headers[i]['bandwidth'], H2O=dataRef[i]))
    return sets
