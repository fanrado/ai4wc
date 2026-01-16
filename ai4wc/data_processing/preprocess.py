from PIL import Image, ImageFilter
import numpy as np

def read_image(image_path):
    """
        Use PIL to read an image from the given path and convert it to a NumPy array.
    """
    with Image.open(image_path) as img:
        return np.array(img)
    
# Resize the input image into a size multiple of the patch size
def resize_image(image, target_size):
    """
        Resize the input image to the target size using PIL.
    """
    img = Image.fromarray(image)
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS) # Use LANCZOS for high-quality downsampling
    return np.array(img)

# Input image --> Image grid of patches
def image_to_patches(image, patch_size):
    """
        Divide the input image into non-overlapping patches of the given patch size.
    """
    img_height, img_width, _ = image.shape
    patches = []
    for i in range(0, img_height, patch_size):
        for j in range(0, img_width, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    return np.array(patches)
# Flatten the grid of patches into a list of patches
def flatten_patches(patches):
    """
        Flatten each patch into a 1D vector and stack them into a 2D array.
    """
    num_patches, patch_height, patch_width, channels = patches.shape
    patch_vector_size = patch_height * patch_width * channels
    flattened_patches = patches.reshape(num_patches, patch_vector_size)
    return flattened_patches
# Flatten each patch into a 1D vector
def flatten_patch(patch):
    """
        Flatten a single patch into a 1D vector.
    """
    patch_height, patch_width, channels = patch.shape
    patch_vector_size = patch_height * patch_width * channels
    return patch.reshape(patch_vector_size)
# Stack all patch vectors into a 2D array (num_patches x patch_vector_size)
def stack_patches(patch_vectors):
    """
        Stack a list of patch vectors into a 2D NumPy array.
    """
    return np.vstack(patch_vectors)
# Create the embedding matrix for the patches
def create_embedding_matrix(patch_vector_size, embedding_dim):
    """
        Create a random embedding matrix to project patch vectors into the embedding space.
    """
    return np.random.randn(patch_vector_size, embedding_dim) * 0.01
# Project the patch vectors into the embedding space : x^i * E
def project_patches(patch_vectors, embedding_matrix):
    """
        Project the patch vectors into the embedding space using the embedding matrix.
    """
    return np.dot(patch_vectors, embedding_matrix)  # Shape: (num_patches, embedding_dim)
# Create the CLS token and prepend it to the sequence of patch embeddings
def add_cls_token(patch_embeddings, cls_token):
    """
        Prepend the CLS token to the sequence of patch embeddings.
    """
    return np.vstack([cls_token, patch_embeddings])  # Shape: (num_patches + 1, embedding_dim)
# Add positional embeddings to the patch embeddings + CLS token
def add_positional_embeddings(embeddings, pos_embeddings):
    """
        Add positional embeddings to the patch embeddings.
    """
    return embeddings + pos_embeddings  # Shape: (num_patches + 1, embedding_dim)
