####################################################################################################
#                                            loadConc.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/07/22                                                                                #
#                                                                                                  #
# Purpose: Load various formats and data types of metabolite concentration lists,                  #
#          e.g. ground truths.                                                                     #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import csv
import os
import pandas as pd

# own
from loading.lcmodel import read_LCModel_coord


#**************************#
#   loading dir of concs   #
#**************************#
def loadConcsDir(path2data, num=None, fmt=None, skip=True):
    """
    Function wrapper of loadDataAsFSL to loading batches of data.

    @param path2data -- The path to the dir.
    @param num -- The number of sets to loading (if possible), defaults to all.
    @param fmt -- The file format (if not given it is inferred).
    @param skip -- If True will skip over invalid/unloadable elements within the directory.

    @returns -- The list of metabolite concentration dictionaries.
    """
    dataNames = sorted(os.listdir(path2data))
    data = []
    crlb = []
    for name in dataNames[:num]:
        # loading file with name
        d, c = loadConcs(path2data + '/' + name, fmt)

        # if skip, then skip over None entries
        if (skip and not isinstance(d, type(None))) or not skip:
            data.append(d)
            crlb.append(c)
    return data, crlb


#****************************************#
#   loading concs in different formats   #
#****************************************#
def loadConcs(path2data, fmt=None):
    """
    Main function to loading various types data formats.

    @param path2data -- The path to the data set.
    @param fmt -- The file format (if not given it is inferred).

    @returns -- The metabolite concentration dictionary.
    """
    if fmt:
        # test for given format
        fmts = ['excel', 'xlsx', 'tarquin', 'csv', 'osprey', 'tsv', 'lcmodel', 'coord']
        if fmt.lower() not in fmts:
            print(f'File format should be in {fmts}, but {fmt} was given.')
    else:
        # ... or find file format ...
        fmt = path2data.split(os.extsep, -1)[-1].lower()

    # ... and match
    if fmt == 'excel' or fmt == 'xlsx':
        return load_EXCEL_conc(path2data)

    elif fmt == 'tarquin' or fmt == 'csv':
        return load_CSV_conc(path2data)

    elif fmt == 'osprey' or fmt == 'tsv':
        return load_TSV_conc(path2data)

    elif fmt == 'lcmodel' or fmt == 'coord' or fmt == 'table':
        if path2data.endswith('.coord'):
            metabs, concs, crlbs, tcr = read_LCModel_coord(path2data, coord=True, meta=False)
            return dict(zip(metabs, concs)), dict(zip(metabs, crlbs))
        elif path2data.endswith('.table'):
            metabs, concs, crlbs, tcr = read_LCModel_coord(path2data, coord=True, meta=False)
            return dict(zip(metabs, concs)), dict(zip(metabs, crlbs))

    else:
        print('-------------- Loading failed! --------------\n'
              ' Invalid data type or not implemented yet... \n'
              'File: ' + path2data)


#************************#
#   loading EXCEL list   #
#************************#
def load_EXCEL_conc(path2conc):
    """
    Load a list of concentrations from an EXCEL file, specifically the ground
    truth concentrations of the ISMRM 2016 challenge.

    @param path2conc -- The path to the concs dir.

    @returns -- The list of metab. conc. dictionaries.
    """
    truth = {'Ace': 0.0}   # initialize, Ace is only partially present

    file = pd.read_excel(path2conc, header=17)
    for i, met in enumerate(file['Metabolites']):
        if not isinstance(met, str): break
        truth[met] = file['concentration'][i]

    truth['Mac'] = truth.pop('MMBL')   # rename key MMBL to Mac

    return dict(sorted(truth.items())), None


#**********************#
#   loading CSV list   #
#**********************#
def load_CSV_conc(path2conc):

    """
    Load a list of concentrations from an csv file exported from TARQUIN.

    @param path2conc -- The path to the concs dir.

    @returns -- The list of metab. conc. dictionaries.
    """
    file = list(csv.reader(open(path2conc)))
    namesC = file[1][3:]   # row, cols
    concs = list(map(float, file[2][3:]))   # map to float
    concs = dict(zip(namesC, concs))

    namesU = file[4][3:]   # row, cols
    crlb = list(map(float, file[5][3:]))   # map to float
    crlb = dict(zip(namesU, crlb))

    if not 'Ace' in concs: concs['Ace'] = 0   # initialize
    if not 'Ace' in crlb: crlb['Ace'] = 0   # initialize

    concs = {key: concs[key] for key in sorted(concs.keys())}  # sort
    crlb = {key: crlb[key] for key in sorted(crlb.keys())}  # sort
    return concs, crlb


#*******************#
#   load TSV list   #
#*******************#
def load_TSV_conc(path2conc):
    """
    Load a list of concentrations from an tsv file exported from Osprey.

    @param path2conc -- The path to the concs dir.

    @returns -- The list of metab. conc. dictionaries.
    """
    fileName = path2conc.split('/')[-1]
    crlbPath = path2conc[:-len(fileName)] + fileName[:2] + 'CRLB' + fileName[-20:]

    file = list(csv.reader(open(path2conc), delimiter='\t'))
    try:  # try to load crlbs
        fileCRLB = list(csv.reader(open(crlbPath), delimiter='\t'))
    except:  # if not possible, set to None
        fileCRLB = None

    allConcs = []
    allCRLBs = []
    for i in range(1, len(file)):
        concs = list(map(float, file[i]))   # map to float
        concs = dict(zip(file[0], concs))
        concs = {key: concs[key] for key in sorted(concs.keys())}  # sort
        allConcs.append(concs)

        if fileCRLB:
            crlbs = list(map(float, fileCRLB[i]))
            crlbs = dict(zip(fileCRLB[0], crlbs))
            crlbs = {key: crlbs[key] for key in sorted(crlbs.keys())}  # sort
            allCRLBs.append(crlbs)

    return allConcs, allCRLBs