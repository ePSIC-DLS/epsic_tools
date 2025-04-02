from mib_dask_import import mib_dask_reader
import numpy.testing as nptest
import os

def test_mib_load():
    my_path = os.path.join(os.path.dirname(__file__), 'tests')
    print(my_path)
    test_mib_files = []
    path_walker = os.walk(os.path.join(my_path, 'test_data'))
    for p, d, files in path_walker:
        for f in files:
            if f.endswith('mib'):
                test_mib_files.append(os.path.join(str(p), str(f)))

    for file in test_mib_files:
        d = mib_dask_reader(file)
        shape = d.data.shape
        nptest.assert_equal(shape, (1,515,515))

test_mib_load()
