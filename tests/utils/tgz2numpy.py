import tarfile
import os, io, gzip
import numpy as np
import h5py
import matplotlib.pyplot as plt

def np_show(arr, title=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(arr)
    if title:
        plt.title(title)
    plt.show()

def read_tgz_to_numpy(tgz_file_path):
    with tarfile.open(tgz_file_path, 'r:gz') as tar:
        list_data = {}
        for member in tar:
            # print(f'Tarfile member : {member}')
            if member.isfile():
                fobj = tar.extractfile(member)
                print(member.name)
                list_data[member.name] = fobj

        print(f'list data : {len(list_data.keys())}')
        basedir = list(list_data.keys())[0].split('/')[0]
        basename = 'monte-carlo-012502-000001_302040_1_1_20260128T233641Z_pixeldata-anode_.h5'
        for i in range(0,12):
            filename = os.path.join(basedir, basename.replace('anode_', f'anode{i}'))
            print(f'Processing file: {filename}')
            with h5py.File(list_data[filename], 'r') as h5file:
                subkey = 'frame_rebinned_reco'
                print(h5file.keys())
                data = np.array(h5file[f'1'][subkey])
                np_show(data, title=f'anode{i}/{subkey}')
        # with h5py.File(fobj, 'r') as h5file:
        #     print(h5file.keys())
        #     keys = list(h5file.keys())
        #     print(h5file[keys[0]].keys())
        #     for key in keys:
        #         subkey = "frame_rebinned_reco"
        #         data = np.array(h5file[key][subkey])
        #         print(f'{key}/{subkey} : {data.shape}')
        #         np_show(data, title=f'{key}/{subkey}')
        #         break
                

if __name__ == '__main__':
    '''
        From the visualization of each 2d array in anode0, we likely want the frame_rebinned_reco for the training data.
    '''
    tgz_file_path = "001/out_monte-carlo-012502-000001_302040_1_1_20260128T233641Z.tgz"
    read_tgz_to_numpy(tgz_file_path=tgz_file_path)