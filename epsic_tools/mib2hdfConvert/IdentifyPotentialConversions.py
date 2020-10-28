import os
import argparse
import sys

def check_differences(beamline, year, visit, folder = None):
    """Checks for .mib files associated with a specified session that have
    not yet been converted to .hdf5.

    Parameters
    ----------
    beamline : str
        Assumed e02 but could be any
    year : str
        Year of user session
    visit : str
        User session number e.g. mg20198-1

    Returns
    -------
    a dictionary with the following keys:
    to_convert_folder : list
        List of directories that is the difference between those converted
        and those to be converted. NB: These are just the folder names and
        not complete paths, e.g. '20190830 112739'.
    mib_paths : list
        List of ALL the mib files including the entire paths that are found
        in the experimental session ,e.g. '/dls/e02/data/2019/em20198-8/Merlin
        /Merlin/Calibrations/AuXgrating/20190830 111907/
        AuX_100kx_10umAp_20cmCL_3p55A2.mib'
    mib_to_convert : list
        List of unconverted mib files including the complete path
    """

    mib_paths = []
    raw_dirs = []
    
    if folder == ' ':
        folder=None

    if folder:
        # check that the path folder exists
        raw_location = os.path.join('/dls',beamline,'data', year, visit, os.path.relpath(folder))
        # print(raw_location)
        if not os.path.exists(raw_location):
            print('This folder ', raw_location,'does not exist!')
            print('The expected format for folder is sample1/dataset1/')
            sys.exit()
    else:
        raw_location = os.path.join('/dls', beamline,'data', year, visit, 'Merlin')

    if folder:
        proc_location = os.path.join('/dls', beamline,'data', year, visit, 'processing', os.path.relpath(folder))
    else:
        proc_location = os.path.join('/dls', beamline,'data', year, visit, 'processing', 'Merlin')
    # print(proc_location)
    if not os.path.exists(proc_location):
        os.makedirs(proc_location)
    # look through all the files in that location and find any mib files

    for p, d, files in os.walk(raw_location):
        # look at the files and see if there are any mib files there
        for f in files:
            if f.endswith('mib'):
                mib_paths.append(os.path.join(str(p), str(f)))
                raw_dirs.append(p)
    # look in the processing folder and list all the directories
    converted_dirs = []

    hdf_files = []
    for p, d, files in os.walk(proc_location):
        # look at the files and see if there are any mib files there
        for f in files:
            if f.endswith('hdf5'):
                if folder:
                    p = './'+ folder + p[1:]
                hdf_files.append((p, f))
                converted_dirs.append(p)

    # only using the time-stamp section of the paths to compare:
    raw_dirs_check = []
    converted_dirs_check = []
    for folder in raw_dirs:
        raw_dirs_check.append(folder.split('/')[-1])
    for folder in converted_dirs:
        converted_dirs_check.append(folder.split('/')[-1])
    # compare the directory lists, and see which have not been converted.
    #converted = set(converted_dirs_check)
    # print(converted)
    to_convert_folder = set(raw_dirs_check) - set(converted_dirs_check)

    mib_to_convert = []
    for mib_path in mib_paths:
        if mib_path.split('/')[-2] in to_convert_folder:
            mib_to_convert.append(mib_path)


    # print('Converted Datasets: ', converted)
    # print('To CONVERT:  ', to_convert)

    # build a dict of to_convert, mib_paths, mib_to_convert
    mib_dict = {}
    mib_dict['processing_path'] = proc_location
    mib_dict['MIB_to_convert'] = mib_to_convert
    mib_dict['all_MIB_paths'] = mib_paths

    return mib_dict


def main(beamline, year, visit, folder):
    mib_dict = check_differences(beamline, year, visit, folder)
    print(mib_dict['MIB_to_convert'])
    print(mib_dict['processing_path'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('beamline', help='Beamline name')
    parser.add_argument('year', help='Year')
    parser.add_argument('visit', help='Session visit code')
    parser.add_argument('folder', nargs= '?', help='Option to add folder')
    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                        default=False)

    args = parser.parse_args()

    main(args.beamline, args.year, args.visit, args.folder)
