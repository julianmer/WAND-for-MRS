####################################################################################################
#                                            lcmodel.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 30/06/22                                                                                #
#                                                                                                  #
# Purpose: Load the raw and h2o file fomats of LCModel.                                            #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import os
import re

from scipy.io import loadmat


#*************************#
#   loading LCModel raw   #
#*************************#
def read_LCModel_raw(filename, conjugate=True):
    """
    Read LCModel (.RAW, .raw, and .H2O) file format. Adapted from [1].

    [1] Clarke, W.T., Stagg, C.J., and Jbabdi, S. (2020). FSL-MRS: An end-to-end
        spectroscopy analysis package. Magnetic Resonance in Medicine, 85, 2950 - 2964.

    @param filename -- Path to .RAW/.H2O file.
    @param bool conjugate -- Apply conjugation upon read.

    @returns -- The basis set data/FID and header if possible.
    """
    header = []
    data   = []
    in_header = False
    after_header = False
    with open(filename, 'r') as f:
        for line in f:
            if (line.find('$') > 0):
                in_header = True

            if in_header:
                header.append(line)
            elif after_header:
                data.append(list(map(float, line.split())))

            if line.find('$END') > 0:
                in_header = False
                after_header = True

    # reshape data
    data = np.concatenate([np.array(i) for i in data])
    data = (data[0::2] + 1j * data[1::2]).astype(complex)

    # LCModel-specific conjugation
    if conjugate:
        data = np.conj(data)

    return data, header


#*******************************************#
#   loading LCModel raw fixed header size   #
#*******************************************#
def read_LCModel_raw_hs(filename, header_size=11, conjugate=True):
    """
    Read LCModel raw format with user-specified header size.

    @param filename -- Path to file.
    @param header_size -- Number of header lines.
    @param bool conjugate -- Apply conjugation upon read.

    @returns -- The basis set data/FID and header if possible.
    """
    header = []
    data = []
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i >= header_size: data.append(list(map(float, line.split())))
            else: header.append(line)

    # reshape data
    data = np.concatenate([np.array(i) for i in data])
    data = (data[0::2] + 1j * data[1::2]).astype(complex)

    # LCModel-specific conjugation
    if conjugate:
        data = np.conj(data)

    return data, header


#*****************************#
#   load LCModel coord data   #
#*****************************#
def read_LCModel_coord(path, coord=True, meta=True):
    """
    Load data based on LCModel coord files (or table files).

    @param path -- The path to the files.
    @param coord -- Load concentration estimates.
    @param meta -- Load meta data.

    @returns -- The data (metabs, concs, crlbs, tcr) or/and meta data (fwhm, snr, shift, phase).
    """
    metabs, concs, crlbs, tcr = [], [], [], []
    fwhm, snr, shift, phase = None, None, None, None

    # go through file and extract all info
    with open(path, 'r') as file:
        concReader = 0
        miscReader = 0

        for line in file:
            if 'lines in following concentration table' in line:
                concReader = int(line.replace('$$CONC ', '').split(' lines')[0])
            elif concReader > 0:  # read concentration table
                concReader -= 1
                values = line.split()

                # check if in header of table
                if values[0] == 'Conc.':
                    continue
                else:
                    try:  # sometimes the fields are fused together with '+'
                        m = values[3]
                        c = float(values[2])
                    except:
                        if 'E+' in values[2]:  # catch scientific notation
                            c = values[2].split('E+')
                            m = str(c[1].split('+')[1:])
                            c = float(c[0] + 'e+' + c[1].split('+')[0])
                        else:
                            if len(values[2].split('+')) > 1:
                                m = str(values[2].split('+')[1:])
                                c = float(values[2].split('+')[0])
                            elif len(values[2].split('-')) > 1:
                                m = str(values[2].split('-')[1:])
                                c = float(values[2].split('-')[0])
                            else:
                                raise ValueError(f'Could not parse {values}')

                    # append to data
                    metabs.append(m)
                    concs.append(float(values[0]))
                    crlbs.append(int(values[1][:-1]))
                    tcr.append(c)
                    continue

            if 'lines in following misc. output table' in line:
                miscReader = int(line.replace('$$MISC ', '').split(' lines')[0])
            elif miscReader > 0:  # read misc. output table
                miscReader -= 1
                values = line.split()

                # extract info
                if 'FWHM' in values:
                    fwhm = float(values[2])
                    snr = float(values[-1].split('=')[-1])
                elif 'shift' in values:
                    if values[3] == 'ppm':
                        shift = float(values[2][1:])  # negative fuses with '='
                    else:
                        shift = float(values[3])
                elif 'Ph' in values:
                    phase = float(values[1])

    if coord and meta: return metabs, concs, crlbs, tcr, fwhm, snr, shift, phase
    elif coord: return metabs, concs, crlbs, tcr
    elif meta: return fwhm, snr, shift, phase


#**************************************#
#   load LCModel fit from coord data   #
#**************************************#
def read_LCModel_fit(path):
    """
        Basic Python function to read LCModel COORD data. Note that this does not extract other
        parameters from the COORD file, and does not attempt to read individual metabolite fits
        (where NEACH is specified).

        Source: https://gist.github.com/alexcraven/3db2c09f14ec489a31df81dc7b5a0f9c

        @param path -- The path to the file to be read.

        @returns -- A dict containing:
            ppm : points on the ppm scale
            data : data to be fit, on the ppm scale
            completeFit : the model fit, including baseline
            baseline : the modelled baseline (not present if NOBASE is set, which is the
                       default for mega-press and certain other modes)
    """
    series_type = None

    series_data = {}

    with open(path) as f:
        vals = []

        for line in f:
            prev_series_type = series_type
            if re.match(".*[0-9]+ points on ppm-axis = NY.*", line):
                series_type = "ppm"
            elif re.match(".*NY phased data points follow.*", line):
                series_type = "data"
            elif re.match(".*NY points of the fit to the data follow.*", line):
                series_type = "completeFit"
                # completeFit implies baseline+fit
            elif re.match(".*NY background values follow.*", line):
                series_type = "baseline"
            elif re.match(".*lines in following.*", line):
                series_type = None
            elif re.match("[ ]+[a-zA-Z0-9]+[ ]+Conc. = [-+.E0-9]+$", line):
                series_type = None

            if prev_series_type != series_type:  # start/end of chunk...
                if len(vals) > 0:
                    series_data[prev_series_type] = np.array(vals)
                    vals = []
            else:
                if series_type:
                    for x in re.finditer(r"([-+.E0-9]+)[ \t]*", line):
                        v = x.group(1)
                        try:
                            v = float(v)
                            vals.append(v)
                        except ValueError:
                            print("Error parsing line: %s" % (line,))
                            print(v)
    return series_data


#*********************#
#   read raw header   #
#*********************#
def read_LCModel_raw_header(path):
    """
    Read the header of an LCModel raw file.

    @param path -- The path to the raw file.

    @returns -- The extracted header information.
    """
    data, header = read_LCModel_raw(path)
    cf, bw, points, fmt, echot, seq = None, None, None, None, None, None
    for line in header:
        if 'HZPPPM' in line and cf is None:
            cf = float(line.split('=')[-1].replace(',', ''))
        if 'BADELT' in line and bw is None:
            bw = 1 / float(line.split('=')[-1].replace(',', ''))
        if 'NDATAB' in line and points is None:
            points = int(line.split('=')[-1])
        if 'FMTDAT' in line and fmt is None:
            fmt = (line.split('=')[-1].replace('\'', '').
                   replace(',', '').strip())
        if 'ECHOT' in line and echot is None:
            echot = float(line.split('=')[-1].replace(',', ''))
        if 'SEQ' in line and seq is None:
            seq = (line.split('=')[-1].replace('\'', '').
                   replace(',', '').strip())
    if points is None: points = len(data)
    return cf, bw, points, fmt, echot, seq


#*******************#
#   read raw data   #
#*******************#
def read_LCModel_raw_data(path):
    """
    Read the data of an LCModel raw file.

    @param path -- The path to the raw file.

    @returns -- The extracted data.
    """
    data, header = read_LCModel_raw(path)
    name = os.path.splitext(os.path.split(path)[-1])[-2]
    conc, volume, tramp, ishift = None, None, None, None

    for line in header:
        if 'CONC' in line: conc = line.split('=')[-1].split(' ')[0]
        else: conc = '1.'
        if 'VOLUME' in line: volume = line.split('=')[-1].split(' ')[0]
        else: volume = '1.'
        if 'TRAMP' in line: tramp = line.split('=')[-1].split(' ')[0]
        else: tramp = '1.'
        if 'ISHIFT' in line: ishift = line.split('=')[-1].split(' ')[0]
        else: ishift = '0'

    # ifft data
    data = np.conjugate(np.fft.ifft(data))
    return data, name, conc, volume, tramp, ishift


#***************************#
#   read FID-A mat header   #
#***************************#
def read_FID_A_mat_header(path):
    """
    Read the header of an LCModel FID-A style MATLAB .mat file.

    @param path -- The path to the .mat file.

    @returns -- The extracted header information.
    """
    data = loadmat(path)['outA']
    return (float(data['txfrq']), int(data['spectralwidth']), int(data['n']),
            None, None, None)


#*************************#
#   read FID-A mat data   #
#*************************#
def read_FID_A_mat_data(path):
    """
    Read the data of an LCModel FID-A style MATLAB .mat file.

    @param path -- The path to the .mat file.

    @returns -- The extracted data.
    """
    # ATTENTION: issues with data basis
    data = loadmat(path)['outA']
    specs = np.conjugate(data['specs'][0][0].squeeze(1))

    # shift water at 4.68 ppm to zero
    specs = np.roll(specs, -(np.abs(data['ppm'][0][0][0] - 4.68)).argmin())
    return specs, data['name'][0][0][0], '1.', '1.', '1.', '0'


#****************************#
#   read FSL-MRS json data   #
#****************************#
def read_FSL_MRS_json(path, header=True):
    """
    Read the header of an FSL-MRS JSON file.

    @param path -- The path to the folder of JSON files.
    @param header -- Return header information, or if False, return the data.

    @returns -- The extracted header information, or the data.
    """
    from fsl_mrs.utils import mrs_io
    mrs = mrs_io.read_basis(path)
    if header:
        return mrs.cf, mrs.original_bw, mrs.original_points, None, None, None
    else:
        specs = np.fft.fft(mrs._raw_fids, axis=0)
        return specs, mrs.names


#***************************#
#   make basis from files   #
#***************************#
def make_basis(path2files, path2basis=None, header=None, filetype=None,
               cf=None, bw=None, points=None, fmt=None, echot=None, seq=None):
    """
    Create a LCModel basis file from individual basis files in a directory.

    @param path2files -- The path to the directory containing the basis files.
    @param path2basis -- The path to the basis set to be saved (optional).
    @param header -- The header to be used (optional).
    @param filetype -- The type of the file to be read (optional, unless ambiguous such as .mat).
    @param cf -- The central frequency of the basis set.
    @param bw -- The bandwidth of the basis set.
    @param points -- The number of points of the basis set.
    @param fmt -- The format of the basis set.
                  (ATTENTION: the format is fixed to '(2E15.6)' for now)
    @param echot -- The echo time of the basis set.
    @param seq -- The sequence of the basis set.
    """
    if path2basis is None: path2basis = os.path.join(path2files, 'basis.basis')
    with open(path2basis, 'w') as output:

        # write header
        if header is not None:
            for line in header: output.write(line)
        else:
            if cf is None or bw is None or points is None or fmt is None:
                # read header from first file
                path = os.path.join(path2files, os.listdir(path2files)[0])

                if path.endswith('.RAW') or path.endswith('.raw'):
                    _cf, _bw, _points, _fmt, _echot, _seq = read_LCModel_raw_header(path)
                elif path.endswith('.mat'):
                    if filetype == 'FID-A':
                        _cf, _bw, _points, _fmt, _echot, _seq = read_FID_A_mat_header(path)
                    elif filetype == 'MARSS':
                        raise NotImplementedError('MARSS filetype not implemented yet.')
                    else:
                        raise ValueError('Invalid filetype. Please provide the filetype '
                                         'of the .mat file, e.g. FID-A, MARSS.')
                elif path.endswith('.json'):
                    # TODO: switch json to be read separately
                    _cf, _bw, _points, _fmt, _echot, _seq = read_FSL_MRS_json(path2files)
                else:
                    _cf, _bw, _points, _fmt, _echot, _seq = None, None, None, None, None, None

                if cf is None: cf = _cf
                if bw is None: bw = _bw
                if points is None: points = _points
                if fmt is None: fmt = _fmt
                if echot is None: echot = _echot
                if seq is None: seq = _seq

                if cf is None or bw is None or points is None:
                    provide = ['central frequency [MHz]' if cf is None else '',
                               'bandwidth [Hz]' if bw is None else '',
                               'number of points' if points is None else '']
                    raise ValueError('Could not extract header information from file. '
                                     'Please provide: ' + ', '.join(provide))
                if fmt is None: fmt = '(2E15.6)'

            output.write(f' $SEQPAR\n')
            output.write(f' FWHMBA = -1.,\n')
            output.write(f' HZPPPM = {cf},\n')
            output.write(f' ECHOT = {echot if echot is not None else "-1."},\n')
            output.write(f' SEQ = \'{seq if seq is not None else " "}\'\n')
            output.write(f' $END\n')
            output.write(f' $BASIS1\n')
            output.write(f' IDBASI = \'JPM\',\n')
            output.write(f' FMTBAS = \'(2E15.6)\',\n')
            # output.write(f' FMTBAS = \'{fmt}\',\n')   # ATTENTION: format is fixed to '(2E15.6)'
            output.write(f' BADELT =  {1/bw},\n')
            output.write(f' NDATAB = {points}\n')
            output.write(f' $END\n')

        for i, f in enumerate(os.listdir(path2files)):
            path = os.path.join(path2files, f)
            if f.endswith('.RAW') or f.endswith('.raw'):
                data, name, conc, volume, tramp, ishift = read_LCModel_raw_data(path)
            elif f.endswith('.mat'):
                if filetype == 'FID-A':
                    data, name, conc, volume, tramp, ishift = read_FID_A_mat_data(path)
                elif filetype == 'MARSS':
                    raise NotImplementedError('MARSS filetype not implemented yet.')
                else:
                    raise ValueError('Invalid filetype. Please provide the filetype '
                                     'of the .mat file, e.g. FID-A, MARSS.')
            elif f.endswith('.json'):
                datas, names = read_FSL_MRS_json(path2files, header=False)
                data, name = datas[:, i], names[i]
                conc, volume, tramp, ishift = '1.', '1.', '1.', '0'
            else:
                raise ValueError(f'Cannot read basis with format .{f.split(".")[-1]}. '
                                 f'Might not be implemented yet...')

            output.write(f' $NMUSED\n')
            output.write(f' XTRASH = 0.00\n')
            output.write(f' $END\n')
            output.write(f' $BASIS\n')
            output.write(f' ID = \'{name}\',\n')
            output.write(f' METABO = \'{name}\',\n')
            output.write(f' CONC = {conc if conc[-1] == "," else conc + ","}\n')
            output.write(f' TRAMP = {tramp if tramp[-1] == "," else tramp + ","}\n')
            output.write(f' VOLUME = {volume if volume[-1] == "," else volume + ","}\n')
            output.write(f' ISHIFT = {ishift[:-1] if ishift[-1] == "," else ishift}\n')
            output.write(f' $END\n')

            # write data
            for i in range(data.shape[0]):
                output.write(f' {data[i].real:.6e}  {data[i].imag:.6e}\n')

    print(f'Basis set saved to {path2basis}.')



#*************#
#   running   #
#*************#
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='LCModel file reader.')
    subparsers = parser.add_subparsers(dest='command')

    # subparser for make basis from raw files
    parser_make_basis = subparsers.add_parser('make_basis', help='Create a basis set from any sort of basis files.')
    parser_make_basis.add_argument('--path2files', type=str, help='Path to the basis files.')
    parser_make_basis.add_argument('--path2basis', type=str, help='Path to the basis set.', default=None)
    parser_make_basis.add_argument('--header', type=str, help='Header to be used.', default=None)
    parser_make_basis.add_argument('--filetype', type=str, help='Type of the file to be read.', default=None)
    parser_make_basis.add_argument('--cf', type=float, help='Central frequency of the basis set [MHz].', default=None)
    parser_make_basis.add_argument('--bw', type=float, help='Bandwidth of the basis set.', default=None)
    parser_make_basis.add_argument('--points', type=int, help='Number of points of the basis set.', default=None)
    parser_make_basis.add_argument('--fmt', type=str, help='Format of the basis set.', default=None)
    parser_make_basis.add_argument('--echot', type=float, help='Echo time of the basis set.', default=None)
    parser_make_basis.add_argument('--seq', type=str, help='Sequence of the basis set.', default=None)


    args = parser.parse_args()
    if args.command == 'make_basis':
        make_basis(args.path2files, args.path2basis, args.header, args.filetype,
                   args.cf, args.bw, args.points, args.fmt, args.echot, args.seq)
    else:
        print('Invalid command.')

