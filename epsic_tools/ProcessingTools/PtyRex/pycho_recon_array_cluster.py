import argparse

def main(beamline, year, visit, folder, folder_num):
    print(beamline, year, visit, folder)
    if folder=='False':
        folder = ''
        HDF5_dict= check_differences(beamline, year, visit)
    else:
        HDF5_dict= check_differences(beamline, year, visit, folder)

    # proc_path = HDF5_dict['processing_path']
    
    to_convert = HDF5_dict['MIB_to_convert']
    folder = to_convert[int(folder_num)-1].rpartition('/')[0].rpartition(visit)[2][1:]
    try:
        save_location = os.path.join('/dls',beamline,'data', year, visit, 'processing', folder)
        if os.path.exists(save_location) == False:
            os.makedirs(save_location)
        watch_convert(beamline, year, visit, folder)
        
    except Exception as e:
        print('** ERROR processing** \n ' , e)


if __name__ == "__main__":
    from distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers =20, memory_limit = 100e9)
    client = Client(cluster)
    parser = argparse.ArgumentParser()
    parser.add_argument('beamline', help='Beamline name')
    parser.add_argument('year', help='Year')
    parser.add_argument('visit', help='Session visit code')
    parser.add_argument('folder', nargs= '?',default=None, help='OPTION to add a specific folder within a visit \mib2hdf_watch_convert
                        to look for data, e.g. sample1/dataset1/. If None the assumption would be to look in Merlin folder')
    parser.add_argument('folder_num', nargs= '?', help='passed by scheduler')
    v_help = "Display all debug log messages"
    parser.add_argument("-v", "--verbose", help=v_help, action="store_true",
                        default=False)

    args = parser.parse_args()
    print(args)

    main(args.beamline, args.year, args.visit, args.folder, args.folder_num)
