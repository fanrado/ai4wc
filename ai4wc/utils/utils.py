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