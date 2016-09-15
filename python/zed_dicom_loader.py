import time
import sys
import os
import fnmatch
import dicom
import functools
import numpy as np
from PyQt4 import QtGui
from zed_debug import *
from volume_viewer import *
import zed_search_files

#####################################################################
## TODO: move the dicom tags analysis to a different python script
#####################################################################

def instance_number_order_comp(a, b):
    # if a.SeriesNumber > b.SeriesNumber:
    #    return 1
    # if a.SeriesNumber < b.SeriesNumber:
    #    return -1
    if a.InstanceNumber > b.InstanceNumber:
        return 1
    if a.InstanceNumber < b.InstanceNumber:
        return -1
    return 0


def load_dicom_files_and_sort_SERIES(filenames):
    dicoms = []
    for f in filenames:
        dcm = dicom.read_file(f)
        dicoms.append(dcm)
    sorted_series = sorted(dicoms, key=functools.cmp_to_key(instance_number_order_comp))
    return sorted_series







def my_comp(a, b):
    if a.SeriesNumber > b.SeriesNumber:
        return 1
    if a.SeriesNumber < b.SeriesNumber:
        return -1
    if a.InstanceNumber > b.InstanceNumber:
        return 1
    if a.InstanceNumber < b.InstanceNumber:
        return -1
    return 0


def check_intra_series_description_change(series):
    desc = series[0].SeriesDescription
    for d in series:
        if d.SeriesDescription != desc:
            return True
    return False

def check_intra_series_orientation_change(series):
    orientation = series[0].ImageOrientationPatient
    for d in series:
        for i in range(len(d.ImageOrientationPatient)):
            if orientation[i] != d.ImageOrientationPatient[i]:
                return True
    return False

def check_series_split(series):
    seen_positions = {}
    for d in series:
        if not d.ImagePositionPatient in seen_positions:
            seen_positions[d.ImagePositionPatient] = 1
        else:
            seen_positions[d.ImagePositionPatient] += 1
    print('unique positions seen:',seen_positions.keys())

    sub_ser_size = -1

    if len(seen_positions.keys()) > 1:
        for k,v in seen_positions.items():
            #if -1 ==  sub_ser_size:
            pass


#[name, must_match_fully, must_substr, forbidden_substr]
T2_fat_supressed = ['T2_fat_supressed',[], ['t2_tse_sag_spair', 't2_tse_sag spair', 'tirm_tra', 'STIR SENSE', 'SAG T2 (FAT-SAT)', 'AX STIR', 'T2 STIR'],
                                []]
T2_non_fat_supressed = ['T2_non_fat_supressed',[], ['t2_tse_tra', 'T2W_TSE SENSE', 'T2 NON FS'],
                        []]
T1_fat_suppressed = ['T1_fat_suppressed',['ad'] ,['t1_fl3d_tra', 't1_fl3d_axi_spair_pre', 't1_fl3d_AXIAL BEFORE DYNA', 'SAG 3D (PRE-CONTRAST)', 'Vibrant Pre', 'AX BLISS_PRE',
                                           'IR_3DFGRE', 'Dynamic-3dfgre', 'BREASTPA', 'Sagittal-3d', 'Sagittal-3DFGRE', 'BREAST,Sag,3D,Gradient Echo'],
                     ['4DYN', 'test', 'SUB', 'MIP', 'SER', 'PE1']]
T1_non_fat_suppresed = ['T1_non_fat_suppresed',[], ['SAG T1 (PRE)', 't1_fl2d_tra_dyn_nonFS', 't1_fl3d_tra NO fs', 'T1W/TSE', 'AX 3D NON FAT SAT'],
                        []]

def check_sequence_type_text_based(series):
    if check_intra_series_description_change(series):
        print('Error! intra series SeriesDescription change!')
        return []

    series_types = [T2_fat_supressed, T2_non_fat_supressed, T1_fat_suppressed, T1_non_fat_suppresed]

    def check_for_type(type, txt):
        # forbidden text
        for s in type[3]:
            if s in txt:
                return None

        # must match fully
        for s in type[1]:
            if s == txt:
                return type[0]

        # must have substr
        for s in type[2]:
            if s in txt:
                return type[0]
        return None

    def check_text_based_types():
        txt = series[0].SeriesDescription
        detected_types = []
        for t in series_types:
            type = check_for_type(t, txt)
            if None != type:
                print(type)
                detected_types.append(type)
        if len(detected_types) > 1:
            print('Error: detected two types!')
            #raise ValueError('detected two series types for the same series!')
            return None
        return detected_types

    detected_sequence_types = check_text_based_types()

    return detected_sequence_types


def validate_series(series):
    if 'SeriesDescription' not in series[0]:
        print('no series description in one of the slices, not supported yet.')
        return False

    print('Validating ',series[0].SeriesDescription,' (',len(series),') slices ...')

    if check_intra_series_description_change(series):
        print('intra series description change, not supported yet.')
        return False
    print(series[0].SeriesDescription)
    if len(series) < 2:
        print("Not enough dicom slices in series, not supported yet.")
        return False
    if check_intra_series_orientation_change(series):
        print("Intra series image orientation (patient) change. Not valid.")
        return False

    series_seq_types = check_sequence_type_text_based(series)

    if len(series_seq_types)>1:
        print("Must match exactly one sequence type. Matching ",len(series_seq_types), " sequence types is not allowed.")
        return False

    if 0==len(series_seq_types):
        print("Not a known sequence type.")
        return False

    seq_type = series_seq_types[0]

    if seq_type != 'T1_fat_suppressed':
        print("supporting only T1_fat_suppressed sequence types at the moment.")
        return False

    return True

def fix_unicode(d):
    #if ((0x0008, 0x0005) in d) and ('ISO_IR' in d[0x0008, 0x0005].value):
    if (0x0008, 0x0005) in d:
        for i in range(len(d[0x0008, 0x0005].value)):
            if 'IR' in d[0x0008, 0x0005].value[i]:
                #d[0x0008, 0x0005].value[i] = d[0x0008, 0x0005][i][:6] + ' ' + d[0x0008, 0x0005][i][6:]
                #fixme: hack
                d[0x0008, 0x0005].value[i] = 'utf-8'
                #d[0x0008, 0x0005].value = d[0x0008, 0x0005][:6] + ' ' + d[0x0008, 0x0005][6:]


def load_dicom(path):
    dc = dicom.read_file(path)
    fix_unicode(dc)
    return dc

#notice that the default is recursive here
#the approach is to load all dicom files into one big pile, and then build the heirarchy
# assumes that root is one patient
def load_dicoms(path, recursive=True, allowed_series_UIDs = None, validate_sequences=True, match_filename=[], only_first=False):
    tic = time.clock()
    filenames = []

    print("loading...")
    filenames = None
    if recursive:
        filenames = zed_search_files.list_files_recurse(path,match_substring = '.dcm')
    else:
        print('non recursive')
        filenames = os.listdir(path)
        filenames = [path + '\\' + f for f in filenames if '.dcm' in f.lower() and os.path.isfile(path + '\\' + f)]

    all_dicoms = []

    for s in filenames:
        match_missed = False
        for mtch in match_filename:
            if mtch not in s:
                match_missed = True
                break
        if match_missed:
            #print('skipping ')
            continue
        #print(s)
        dcm = dicom.read_file(s)
        fix_unicode(dcm)

        ok_to_append = False
        if None == allowed_series_UIDs:
            ok_to_append = True
        else:
            if dcm.SeriesInstanceUID in allowed_series_UIDs:
                ok_to_append = True

        if ok_to_append:
            if only_first:
                return dcm
            all_dicoms.append(dcm)


    sorted_by_series_uid = {}

    per_study = {}

    for dcm in all_dicoms:
        if not dcm.StudyInstanceUID in per_study:
            per_study[dcm.StudyInstanceUID] = {}
        if not dcm.SeriesInstanceUID in per_study[dcm.StudyInstanceUID]:
            per_study[dcm.StudyInstanceUID][dcm.SeriesInstanceUID] = []
        per_study[dcm.StudyInstanceUID][dcm.SeriesInstanceUID].append(dcm)


    #fixme: validation of serieses is not implemented yet
    if False:
        #eliminate serieses that we do not support
        valid_series = {}


        for study_key, study in per_study.items():
            for series_key, series in study.items():
                # sort the slices
                sorted_series = sorted(series, key=functools.cmp_to_key(instance_number_order_comp))
                study[series_key] = sorted_series

                if not validate_sequences or validate_series(series):
                    valid_series[series_key] = sorted_series

        print('--- valid sequences --- ')

        for k,s in valid_series.items():
            print(s[0].SeriesDescription)
            #show_volume(s)
    #sort series sequences internally

    toc = time.clock()
    print('time=', toc - tic)

    return per_study

def test_interactive():
    app = QtGui.QApplication(sys.argv)
    selected_dir = QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder:', r'C:\dev\dbs', QtGui.QFileDialog.ShowDirsOnly)
    if '' != selected_dir:
        load_dicoms(selected_dir)

#test_interactive()
#test_interactive()