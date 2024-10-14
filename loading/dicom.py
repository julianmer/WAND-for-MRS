####################################################################################################
#                                              dicom.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 18/05/22                                                                                #
#                                                                                                  #
# Purpose: Load dicom data formats for MRS and MRSI data sets.                                     #
#                                                                                                  #
#          Parts of the code are taken form:                                                       #
#          https://pydicom.github.io/pydicom/dev/auto_examples/input_output/                       #
#          plot_read_dicom_directory.html                                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import matplotlib.pyplot as plt
import numpy as np
import os

from pathlib import Path

from pydicom import dcmread


#******************#
#   loading data   #
#******************#
def loadDICOM(path2data, verbose=0):
    """
    Extract data set from the series.

    @param series -- The series to get the records from.
    @param typ -- The type of the series.
    @param root_dir -- The root directory.
    @param verbose -- Default 0, 1 - give dir structure, 2 - visualize all

    @returns -- The spectra and metadata.
    """
    instance = dcmread(path2data)

    # init
    spec_data_main, spec_data_ref, header = None, None, None

    try:
        header = {
            'protocolName': instance.ProtocolName,
            'csShiftReference': instance.ChemicalShiftReference,
            'centralFrequency': instance.TransmitterFrequency,
            'bandwidth': instance.SpectralWidth,
            'resonantNucleus': instance.ResonantNucleus
        }

        specData = np.frombuffer(instance.SpectroscopyData, dtype=np.single)

        specDataCmplx = specData[0::2] + 1j * specData[1::2]

        spec_points = instance.SpectroscopyAcquisitionDataColumns

        spec_data_main = specDataCmplx[:spec_points]
        spec_data_ref = specDataCmplx[spec_points:]

    except:
        if verbose == 1 or verbose == 2:
            print(f"{'  ' * 3}  Instance {os.fspath(p)} does "
                  'not contain required information...')

    return spec_data_main, spec_data_ref, header


#*************************************#
#   get image records in the series   #
#*************************************#
def getRecords(series, typ, root_dir, verbose=0):
    """
    Extract data set from the series.

    @param series -- The series to get the records from.
    @param typ -- The type of the series.
    @param root_dir -- The root directory.
    @param verbose -- Default 0, 1 - give dir structure, 2 - visualize all

    @returns -- The spectra and metadata.
    """

    # get records in the series
    images = [s for s in series.children if s.DirectoryRecordType == typ]

    # get the absolute file path to each instance
    elems = [s["ReferencedFileID"] for s in images]

    # make sure the relative file path is always a list of str
    paths = [[ee.value] if ee.VM == 1 else ee.value for ee in elems]
    paths = [Path(*p) for p in paths]

    # init
    spec_data_main, spec_data_ref, header = None, None, None

    # list the instance file paths
    for p in paths:
        if verbose == 1 or verbose == 2:
            print(f"{'  ' * 3}- Path={os.fspath(p)}")

        # optionally read the corresponding SOP instance
        instance = dcmread(Path(root_dir) / p)

        if typ == 'IMAGE' and verbose == 2:
            plt.imshow(instance.pixel_array[0], cmap=plt.cm.gray)
            plt.show()

        if typ == 'PRIVATE':

            try:
                header = {
                    'protocolName': instance.ProtocolName,
                    'csShiftReference': instance.ChemicalShiftReference,
                    'centralFrequency': instance.TransmitterFrequency,
                    'bandwidth': instance.SpectralWidth,
                    'resonantNucleus': instance.ResonantNucleus
                }

                specData = np.frombuffer(instance.SpectroscopyData, dtype=np.single)
                print(specData.shape)
                specDataCmplx = specData[0::2] + 1j * specData[1::2]

                spec_points = instance.SpectroscopyAcquisitionDataColumns

                spec_data_main = specDataCmplx[:spec_points]
                spec_data_ref = specDataCmplx[spec_points:]

            except:
                if verbose == 1 or verbose == 2:
                    print(f"{'  ' * 3}  Instance {os.fspath(p)} does "
                          'not contain required information...')

    return spec_data_main, spec_data_ref, header


#******************#
#   loading data   #
#******************#
def loadDICOMDIR(path2data, verbose=0):
    """
    Load a corpus of data sets from the DICOMDIR format.

    @param path2data -- The path to the directory.
    @param verbose -- Default 0, 1 - give dir structure, 2 - visualize all

    @returns -- The spectra and metadata.
    """
    ds = dcmread(path2data)
    root_dir = Path(ds.filename).resolve().parent

    if verbose == 1 or verbose == 2:
        print(f'Root directory: {root_dir}\n')

    specs = []
    specRefs = []
    headers = []

    # get records of patients
    for patient in ds.patient_records:

        # see what patient(s)
        if verbose == 1 or verbose == 2:
            print(
                f"PATIENT: PatientID={patient.PatientID}, "
                f"PatientName={patient.PatientName}"
            )

        # get study records for the patient
        studies = [s for s in patient.children if s.DirectoryRecordType == "STUDY"]

        for study in studies:
            if verbose == 1 or verbose == 2:
                descr = study.StudyDescription or "(no value available)"
                print(
                    f"{'  ' * 1}STUDY: StudyID={study.StudyID}, "
                    f"StudyDate={study.StudyDate}, StudyDescription={descr}"
                )

            # get series records in the study
            allSeries = [s for s in study.children if s.DirectoryRecordType == "SERIES"]

            for series in allSeries:

                # print(series)

                if verbose == 1 or verbose == 2:
                    # get image records in the series
                    images = [s for s in series.children]
                    plural = ('', 's')[len(images) > 1]
                    descr = getattr(series, "SeriesDescription", "(no value available)")
                    print(
                        f"{'  ' * 2}SERIES: SeriesNumber={series.SeriesNumber}, "
                        f"Modality={series.Modality}, SeriesDescription={descr} - "
                        f"{len(images)} SOP Instance{plural}"
                    )

                # extract information
                getRecords(series, 'IMAGE', root_dir, verbose)

                spec, specRef, header = getRecords(series, 'PRIVATE', root_dir, verbose)

                if not isinstance(spec, type(None)):
                    specs.append(spec)
                    specRefs.append(specRef)
                    headers.append(header)

    return specs, specRefs, headers