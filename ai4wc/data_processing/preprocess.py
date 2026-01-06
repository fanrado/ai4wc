from PIL import Image
import numpy as np

def read_image(image_path):
    """
        Use PIL to read an image from the given path and convert it to a NumPy array.
    """
    with Image.open(image_path) as img:
        return np.array(img)
    
# Resize the input image into a size multiple of the patch size
    
# Input image --> Image grid of patches

# Flatten the grid of patches into a list of patches

# Flatten each patch into a 1D vector

# Stack all patch vectors into a 2D array (num_patches x patch_vector_size)

# Create the embedding matrix for the patches

# Project the patch vectors into the embedding space : x^i * E

# Create the CLS token and prepend it to the sequence of patch embeddings

# Add positional embeddings to the patch embeddings + CLS token

