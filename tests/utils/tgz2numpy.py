import sys
import tarfile
import os, io, gzip
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
import click

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

def read_tgz_file(tgz_file_path, plot_sparsemat=False):
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
        list_h5 = [fname for fname in list_data.keys() if fname.endswith('.h5')]
        metadata_file = [fname for fname in list_data.keys() if 'metadata' in fname][0]
        # for filename in list_data.keys():
        #     if 'metadata' in filename:
        # continue
        print('==================================')
        print(f'Processing metadata file: {metadata_file}')
        with h5py.File(list_data[metadata_file], 'r') as h5file:
            print(f'Metadata keys: {list(h5file.keys())}')
            for key in h5file.keys():
                print(f'Frame {key}: {h5file[key]}')
                key_str = f"/{key}"
                print(f"data for {key_str} : {h5file[key][key_str]['metadata'][0]}")
            # else:
            #     continue
        for filename in list_h5:
            print('==================================')
            print(f'Processing file: {filename}')
            with h5py.File(list_data[filename], 'r') as h5file:
                subkey = 'frame_rebinned_reco'
                
                print(h5file.keys())
                
                # Create figure with 10 subplots (2 rows x 5 columns)
                fig, axes = plt.subplots(2, 5, figsize=(25, 10))
                axes = axes.flatten()
                
                # Store original limits for reset functionality
                original_limits = {}
                
                # Flag to prevent infinite recursion in event handlers
                updating = [False]
                
                def on_xlims_change(event_ax):
                    """Synchronize x-axis limits across all subplots"""
                    if updating[0]:
                        return
                    updating[0] = True
                    xlim = event_ax.get_xlim()
                    for ax in axes:
                        if ax != event_ax:
                            ax.set_xlim(xlim)
                    fig.canvas.draw_idle()
                    updating[0] = False
                
                def on_ylims_change(event_ax):
                    """Synchronize y-axis limits across all subplots"""
                    if updating[0]:
                        return
                    updating[0] = True
                    ylim = event_ax.get_ylim()
                    for ax in axes:
                        if ax != event_ax:
                            ax.set_ylim(ylim)
                    fig.canvas.draw_idle()
                    updating[0] = False
                
                # Process frames 1 through 10
                for frame_idx in range(1, 11):
                    frame_key = str(frame_idx)
                    ax = axes[frame_idx - 1]
                    
                    try:
                        print(f'Processing frame {frame_key}')
                        data = h5file[frame_key][subkey]
                        coords = np.array(data['coords'])
                        features = np.array(data['features'])
                        
                        # Convert to sparse matrix
                        sparse_mat = coords_features_to_sparse(coords, features, format='csr')
                        
                        # Plot sparse matrix as image
                        im = None
                        if plot_sparsemat:
                            im = ax.imshow(sparse_mat.toarray(), cmap='viridis', aspect='auto', origin='lower')
                        else:
                            im = ax.scatter(coords[:, 1], coords[:, 0], c=features, cmap='viridis', s=10)
                        ax.set_title(f'Frame {frame_key}')
                        ax.set_xlabel('Column Index')
                        ax.set_ylabel('Row Index')
                        
                        # Connect zoom/pan synchronization events
                        ax.callbacks.connect('xlim_changed', on_xlims_change)
                        ax.callbacks.connect('ylim_changed', on_ylims_change)
                        
                        # Store original limits for reset
                        original_limits[ax] = {
                            'xlim': ax.get_xlim(),
                            'ylim': ax.get_ylim()
                        }
                        
                        print(f"  Frame {frame_key} - Shape: {sparse_mat.shape}, Non-zero: {sparse_mat.nnz}")
                        
                    except Exception as e:
                        print(f"  Error processing frame {frame_key}: {e}")
                        ax.text(0.5, 0.5, f'Frame {frame_key}\nError: {str(e)}', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_xticks([])
                        ax.set_yticks([])
                
                # Override the toolbar's home button to reset all subplots simultaneously
                toolbar = fig.canvas.manager.toolbar
                original_home = toolbar.home
                
                def custom_home(*args, **kwargs):
                    """Custom home function that resets all axes simultaneously"""
                    updating[0] = True
                    for ax in axes:
                        if ax in original_limits:
                            ax.set_xlim(original_limits[ax]['xlim'])
                            ax.set_ylim(original_limits[ax]['ylim'])
                    toolbar._nav_stack.clear()
                    toolbar.push_current()
                    fig.canvas.draw_idle()
                    updating[0] = False
                
                toolbar.home = custom_home
                
                # Adjust layout to make room for colorbar
                fig.suptitle(f'{filename} - All Frames', fontsize=16, y=0.99)
                plt.tight_layout(rect=[0, 0.05, 1, 0.97])
                
                # Add a single colorbar for all subplots in a dedicated axis
                cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])  # [left, bottom, width, height]
                fig.colorbar(im, cax=cbar_ax, label='Feature Value', orientation='horizontal')
                
                plt.show()
                
                sys.exit()
                # data = np.array(h5file[f'1'][subkey])
                # np_show(data, title=f'anode/{subkey}')

@click.command()
@click.option('-i', '--input', 'tgz_file_path', type=click.Path(exists=True), 
              required=True, help='Path to the .tgz file to process')
@click.option('--plot-sparsemat', is_flag=True, default=False, 
              help='Plot as sparse matrix instead of scatter plot')
def main(tgz_file_path, plot_sparsemat):
    '''
    Visualize data from a TGZ file containing HDF5 data.
    
    TGZ_FILE_PATH: Path to the .tgz file to process
    
    From the visualization of each 2d array in anode0, we likely want 
    the frame_rebinned_reco for the training data.
    '''
    read_tgz_file(tgz_file_path=tgz_file_path, plot_sparsemat=plot_sparsemat)

if __name__ == '__main__':
    main()