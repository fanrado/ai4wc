import tarfile
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Union, Optional


class TGZDataset(Dataset):
    """
    PyTorch Dataset for loading .tgz files containing h5 anode data.
    
    Each .tgz file contains multiple h5 files (one per anode), and each h5 file
    contains frame_rebinned_reco data that will be used for training.
    
    Args:
        tgz_paths: List of paths to .tgz files or a directory containing .tgz files
        num_anodes: Number of anodes to read from each .tgz file (default: 12)
        transform: Optional transform to apply to the images
        cache_in_memory: If True, cache all data in memory (faster but uses more RAM)
    """
    
    def __init__(
        self, 
        tgz_paths: Union[str, List[str]], 
        num_anodes: int = 12,
        transform: Optional[callable] = None,
        cache_in_memory: bool = False
    ):
        super().__init__()
        
        # Handle directory path or list of file paths
        if isinstance(tgz_paths, str):
            tgz_dir = Path(tgz_paths)
            if tgz_dir.is_dir():
                self.tgz_files = sorted(list(tgz_dir.glob("*.tgz")))
            elif tgz_dir.is_file():
                self.tgz_files = [tgz_dir]
            else:
                raise ValueError(f"Path {tgz_paths} is neither a file nor a directory")
        else:
            self.tgz_files = [Path(p) for p in tgz_paths]
        
        self.num_anodes = num_anodes
        self.transform = transform
        self.cache_in_memory = cache_in_memory
        
        # Build index: each item is (tgz_file_idx, anode_idx)
        self.index = []
        for tgz_idx in range(len(self.tgz_files)):
            for anode_idx in range(self.num_anodes):
                self.index.append((tgz_idx, anode_idx))
        
        # Cache for storing loaded data
        self.cache = {} if cache_in_memory else None
        
        # Load all data into memory if caching is enabled
        if self.cache_in_memory:
            print(f"Caching {len(self)} images in memory...")
            for idx in range(len(self)):
                self.cache[idx] = self._load_image(idx)
            print("Caching complete!")
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        """
        Get a single image from the dataset.
        
        Returns:
            Tensor of shape (H, W) containing the frame_rebinned_reco data
        """
        if self.cache_in_memory:
            image = self.cache[idx]
        else:
            image = self._load_image(idx)
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def _load_image(self, idx):
        """Load a single image from disk."""
        tgz_idx, anode_idx = self.index[idx]
        tgz_path = self.tgz_files[tgz_idx]
        
        try:
            with tarfile.open(tgz_path, 'r:gz') as tar:
                # Get all members and extract the base directory structure
                members = tar.getmembers()
                file_members = {m.name: m for m in members if m.isfile()}
                
                # Find the h5 file for this anode
                # Pattern: basedir/basename_anode{i}.h5
                h5_files = [name for name in file_members.keys() if name.endswith('.h5')]
                
                if not h5_files:
                    raise ValueError(f"No h5 files found in {tgz_path}")
                
                # Extract base directory and basename pattern
                basedir = h5_files[0].split('/')[0]
                
                # Find the file matching this anode index
                anode_file = None
                for h5_file in h5_files:
                    if f'anode{anode_idx}' in h5_file:
                        anode_file = h5_file
                        break
                
                if anode_file is None:
                    raise ValueError(f"Could not find anode{anode_idx} in {tgz_path}")
                
                # Extract and read the h5 file
                fobj = tar.extractfile(file_members[anode_file])
                
                with h5py.File(fobj, 'r') as h5file:
                    # Read frame_rebinned_reco from group '1'
                    data = np.array(h5file['1']['frame_rebinned_reco'])
                    
                    # Convert to PyTorch tensor (float32)
                    image = torch.from_numpy(data).float()
                    
                    return image
                    
        except Exception as e:
            raise RuntimeError(f"Error loading {tgz_path}, anode {anode_idx}: {str(e)}")
    
    def get_image_shape(self):
        """Get the shape of a single image."""
        sample = self[0]
        return sample.shape
    
    def get_file_info(self, idx):
        """Get information about which file and anode an index corresponds to."""
        tgz_idx, anode_idx = self.index[idx]
        return {
            'tgz_file': str(self.tgz_files[tgz_idx]),
            'anode': anode_idx,
            'index': idx
        }


def create_dataloader(
    tgz_paths: Union[str, List[str]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    num_anodes: int = 12,
    transform: Optional[callable] = None,
    cache_in_memory: bool = False,
    **kwargs
) -> DataLoader:
    """
    Create a PyTorch DataLoader for .tgz files.
    
    Args:
        tgz_paths: Path to directory containing .tgz files or list of .tgz file paths
        batch_size: Number of images per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        num_anodes: Number of anodes per .tgz file
        transform: Optional transform to apply to images
        cache_in_memory: Whether to cache all data in memory
        **kwargs: Additional arguments to pass to DataLoader
    
    Returns:
        PyTorch DataLoader instance
    """
    dataset = TGZDataset(
        tgz_paths=tgz_paths,
        num_anodes=num_anodes,
        transform=transform,
        cache_in_memory=cache_in_memory
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )


if __name__ == '__main__':
    # Example usage
    import matplotlib.pyplot as plt
    
    # Create dataset from a single .tgz file or directory
    tgz_path = "001/out_monte-carlo-012502-000001_302040_1_1_20260128T233641Z.tgz"
    
    dataset = TGZDataset(tgz_path)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Image shape: {dataset.get_image_shape()}")
    
    # Get first image
    image = dataset[0]
    print(f"First image shape: {image.shape}")
    print(f"First image dtype: {image.dtype}")
    print(f"First image min/max: {image.min():.2f}/{image.max():.2f}")
    
    # Show file info
    print(f"First image info: {dataset.get_file_info(0)}")
    
    # Create a dataloader
    dataloader = create_dataloader(
        tgz_paths=tgz_path,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # Iterate through one batch
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        break
    
    # Visualize some images
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for i in range(12):
        ax = axes[i // 6, i % 6]
        img = dataset[i].numpy()
        ax.imshow(img)
        ax.set_title(f"Anode {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
