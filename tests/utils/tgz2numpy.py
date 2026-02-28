import sys
import tarfile
import os, io, gzip
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix

def np_show(arr, title=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(arr)
    if title:
        plt.title(title)
    plt.show()

def coords_features_to_sparse(coords, features, shape=None, format='csr'):
    """
    Convert coordinate-based data to a sparse matrix.
    
    Parameters:
    -----------
    coords : numpy.ndarray
        Array of shape (N, 2) containing (row, col) coordinates
    features : numpy.ndarray
        Array of shape (N,) containing feature values at each coordinate
    shape : tuple, optional
        Shape of the output sparse matrix (rows, cols). 
        If None, inferred from max coordinates + 1
    format : str, default='csr'
        Sparse matrix format: 'csr' (Compressed Sparse Row) or 'coo' (Coordinate)
    
    Returns:
    --------
    sparse_matrix : scipy.sparse matrix
        Sparse matrix in the specified format (CSR or COO)
    
    Examples:
    ---------
    >>> coords = np.array([[0, 1], [2, 3], [1, 1]])
    >>> features = np.array([0.5, 1.2, 0.8])
    >>> sparse_mat = coords_features_to_sparse(coords, features)
    """
    # Validate inputs
    if coords.shape[0] != features.shape[0]:
        raise ValueError(f"Coords and features must have same length: "
                        f"{coords.shape[0]} != {features.shape[0]}")
    
    if coords.shape[1] != 2:
        raise ValueError(f"Coords must have shape (N, 2), got {coords.shape}")
    
    # Extract row and column indices
    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)
    
    # Determine shape if not provided
    if shape is None:
        shape = (rows.max() + 1, cols.max() + 1)
    
    # Create COO matrix (most natural for coordinate data)
    sparse_mat = coo_matrix((features, (rows, cols)), shape=shape)
    
    # Convert to requested format
    if format.lower() == 'csr':
        return sparse_mat.tocsr()
    elif format.lower() == 'coo':
        return sparse_mat
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csr' or 'coo'")

def read_tgz_to_numpy(tgz_file_path):
    with tarfile.open(tgz_file_path, 'r:gz') as tar:
        list_data = {}
        for member in tar:
            # print(f'Tarfile member : {member}')
            if member.isfile():
                fobj = tar.extractfile(member)
                print(member.name)
                list_data[member.name] = fobj

        # print(f'list data : {len(list_data.keys())}')
        # print(f'list data : {list_data}')
        # basedir = list(list_data.keys())[0].split('/')[0]
        # print(f'Basedir : {basedir}')
        # basename = 'monte-carlo-012502-000001_302040_1_1_20260128T233641Z_pixeldata-anode_.h5'
        # for i in range(0,12):
        #     filename = os.path.join(basedir, basename.replace('anode_', f'anode{i}'))
        for filename in list_data.keys():
            if 'metadata' in filename:
                continue
            print(f'Processing file: {filename}')
            with h5py.File(list_data[filename], 'r') as h5file:
                subkey = 'frame_rebinned_reco'
                print(h5file.keys())
                print(f'h5file["1"][f"{subkey}"].keys() : {h5file["1"][subkey].keys()}')
                data = h5file[f'1'][subkey]
                print(data.keys())
                print(f"coords : {data['coords']}")
                print(f"features : {data['features']}")
                coords = np.array(data['coords'])
                features = np.array(data['features'])
                print(f"coords shape : {coords.shape}")
                print(f"features shape : {features.shape}")
                
                plt.figure(figsize=(10, 10))
                plt.scatter(coords[:, 0], coords[:, 1], c=features[:], cmap='viridis')
                plt.colorbar(label='Feature Value')
                plt.title(f'{filename} - {subkey}')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')                
                plt.show()
                
                # Convert to sparse matrix
                sparse_mat = coords_features_to_sparse(coords, features, format='csr')
                print(f"\nSparse matrix created:")
                print(f"  Shape: {sparse_mat.shape}")
                print(f"  Format: {type(sparse_mat).__name__}")
                print(f"  Non-zero elements: {sparse_mat.nnz}")
                print(f"  Sparsity: {(1 - sparse_mat.nnz / (sparse_mat.shape[0] * sparse_mat.shape[1])) * 100:.2f}%")
                
                # Plot sparse matrix as image
                plt.figure(figsize=(12, 10))
                plt.imshow(sparse_mat.toarray().T, cmap='viridis', aspect='auto')
                plt.colorbar(label='Feature Value')
                plt.title(f'{filename} - Sparse Matrix Visualization')
                plt.xlabel('Column Index')
                plt.ylabel('Row Index')
                plt.show()
                
                sys.exit()
                # data = np.array(h5file[f'1'][subkey])
                # np_show(data, title=f'anode/{subkey}')

if __name__ == '__main__':
    '''
        From the visualization of each 2d array in anode0, we likely want the frame_rebinned_reco for the training data.
    '''
    # tgz_file_path = "001/out_monte-carlo-012502-000001_302040_1_1_20260128T233641Z.tgz"
    tgz_file_path = "out_monte-carlo-013717-000499_304871_97_1_20260224T034018Z.tgz"
    read_tgz_to_numpy(tgz_file_path=tgz_file_path)