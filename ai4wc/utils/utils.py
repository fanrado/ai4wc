import os
import shutil
import torch

def get_device():
    '''
        Returns the available device:
         - cuda if NVIDIA GPU is available
         - mps if Apple Silicon GPU is available
         - cpu otherwise
    '''
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    return device



def create_dir_structure(DATA_SOURCE, DATA_ROOT):
    def get_unique_filename(flavor='nutau'):
        # return unique list of prefixes in the flavor directory
        flavor_path = os.path.join(DATA_SOURCE, flavor)
        list_files = os.listdir(flavor_path)
        prefixes = [f.split('.')[0] for f in list_files if f.startswith('event')]
        unique_prefixes = list(set(prefixes))
        return unique_prefixes
    ## The particular dataset is balanced, so each flavor has the name number of events. This could be different for other datasets.
    N = len(get_unique_filename('nutau'))
    # N = 20
    print(f'Total number of events per flavor: {N}')

    # separate into train, val, test : 70%, 15%, 15%
    train_split = int(0.7 * N)
    val_split = int(0.15 * N)
    test_split = N - train_split - val_split
    # Each event is stored in a gz file with its corresponding .info file. We need to copy both files.
    ## Create train, val, and test directories
    train_path = os.path.join(DATA_ROOT, 'train')
    val_path = os.path.join(DATA_ROOT, 'val')
    test_path = os.path.join(DATA_ROOT, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    # For each flavor, create subdirectories
    # flavors = os.listdir(DATA_SOURCE)
    flavors = ['nue', 'numu', 'nc']
    for flavor in flavors:
        os.makedirs(os.path.join(train_path, flavor), exist_ok=True)
        os.makedirs(os.path.join(val_path, flavor), exist_ok=True)
        os.makedirs(os.path.join(test_path, flavor), exist_ok=True)
    # Now, copy files into respective directories
    for flavor in flavors:
        list_files = get_unique_filename(flavor=flavor)
        flavor_path = os.path.join(DATA_SOURCE, flavor)
        for i, file in enumerate(list_files):
            src_file_gz = os.path.join(flavor_path, file + '.gz')
            src_file_info = os.path.join(flavor_path, file + '.info')
            if i < train_split:
                dest_dir = os.path.join(train_path, flavor)
                # dest_dir = os.path.join(train_path)
            elif i < train_split + val_split:
                dest_dir = os.path.join(val_path, flavor)
                # dest_dir = os.path.join(val_path)
            else:
                dest_dir = os.path.join(test_path, flavor)
                # dest_dir = os.path.join(test_path)
            shutil.copy(src_file_gz, dest_dir)
            shutil.copy(src_file_info, dest_dir)

