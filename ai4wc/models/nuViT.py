import torch
import os
import torch.nn as nn

def patchify(batch, patch_size):
    """"
        Patchify the input batch of images    
        Shape:
            batch: (B, H, W, C) where B is batch size, H is height, W is width, C is channels
            output: (B, NH, NW, PH, PW, C) where NH is number of patches along height,
                    NW is number of patches along width, PH is patch height, PW is patch width
    """
    b, c, h, w = batch.shape
    ph, pw = patch_size
    nh, nw = h//ph, w//pw

    batch_patches = torch.reshape(batch, (b, c, nh, ph, nw, pw)) ## if h==w and ph==pw, then nh==nw. In that case, each image in the batch is divided into nh patches of size ph x pw [[nh squares of size ph x pw, which can be seen as nh divisions along the height and nw divisions along the width]]
    batch_patches = torch.permute(batch_patches, (0, 1, 2, 4, 3, 5)) ## (B, C, NH, NW, PH, PW)

    return batch_patches

def test_patchify(path_to_image: str, patch_size=(16,16), image_size=(512, 512)):
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    ## read and resize image
    img = Image.open(path_to_image)
    img = img.resize(image_size)
    img = np.array(img)
    img = np.transpose(img, (2,0,1))  ## Change to (C, H, W)
    
    ## converting image to float32
    img = img / np.max(img)
    img = img.astype(np.float32)
    ##-- 

    batch = torch.tensor(img[None])
    print(batch.shape)
    batch_patches = patchify(batch, patch_size=patch_size)

    patches = batch_patches[0]
    print(patches.shape)
    c, nh, nw, ph, pw = patches.shape

    plt.figure(figsize=(5,5))
    plt.imshow(img[0], cmap='gray')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10,10))
    for i in range(nh):
        for j in range(nw):
            plt.subplot(nh, nw, i*nw+j+1)
            plt.imshow(patches[0, i, j], cmap='gray')
            plt.axis('off')

## MLP
def get_mlp(in_features, hidden_units, out_features):
    """
        Return a MLP head
    """
    dims = [in_features] + hidden_units + [out_features]
    layers = []
    for dim1, dim2 in zip(dims[:-2], dims[1:-1]):
        layers.append(nn.Linear(dim1, dim2))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)

## Taking a batch of images in the input, convert them into sequences to be fed to the transformer encoder
## We create the positional embeddings (and add them to the patch embeddings) and CLS token here (and prepend it to the sequence of patch_embeddings + positional embeddings)
class Img2Seq(nn.Module):
    """
        This class takes a batch of images as input and
        returns a batch of sequences.

        Shape:
            input: (b, h, w, c) where b is batch size, h is height, w is width, c is channels
            output: (b, s, d) where s is sequence length, d is embedding dimension
    """
    def __init__(self, img_size, patch_size, n_channels, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size

        nh, nw = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        n_tokens = nh * nw

        token_dim = patch_size[0] * patch_size[1] * n_channels
        self.linear = nn.Linear(token_dim, d_model) ## create a function taking in the input token_dim features and output d_model features.
        self.cls_token = nn.Parameter(torch.randn(1,1, d_model)) ## create the CLS token, tensor of the form [[[d_model elements]]]. These are learnable parameters of the model.
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, d_model)) ## create the positional embeddings. A 2-dimensional tensor with nh*nw rows and d_model columns. d_model is the number of elements in one token.
                                                                    ## These are learnable parameters of the model.
    
    def __call__(self, batch):
        batch = patchify(batch, self.patch_size)  ## (b, c, nh, nw, ph, pw)
        b, c, nh, nw, ph, pw = batch.shape # b: batch size, c: channels, nh: number of patches along height, nw: number of patches along width, ph: patch height, pw: patch width

        ## Flattening the patches
        batch = torch.permute(batch, [0, 2, 3, 4, 5, 1]) ## ==> [b, nh, nw, ph, pw, c]
        batch = torch.reshape(batch, (b, nh*nw, ph*pw*c)) ## ==> shape : [b, nh*nw, ph*pw*c] where ph*pw*c is the token_dim

        batch = self.linear(batch) ## Project the patch vectors into the embedding space ==> shape: [b, nh*nw, d_model]
        cls = self.cls_token.expand([b, -1, -1]) ## ==> shape [b, 1, d_model]
        emb = batch + self.pos_emb ## Add positional embeddings to the patch embeddings ==> shape: [b, nh*nw, d_model]

        return torch.cat([cls, emb], axis=1) ## prepend the CLS token to the sequence of patch embeddings + positional embeddings, shape: [b, nh*nw + 1, d_model]
    
## Vision Transformer model
## nn.TransformerEncoder and nn.TransformerEncoderLayer are use to implement the transformer encoder.
class nuViT(nn.Module):
    """
        Args:
            img_size: tuple of ints (H, W) - input image size
            patch_size: tuple of ints (PH, PW) - patch size
            n_channels: int - number of input channels
            d_model: int - number of features in the transformer encoder
            nhead: int - number of heads in the mutliheadattention models
            dim_feedforward: int - dimension of the feedforward network in the transformer encoder
            blocks: int - number of sub-encoder-layers in the transformer encoder
            mlp_head_units: hidden units of mlp_head
            n_classes: int - number of output classes
    """
    def __init__(self, img_size,
                 patch_size,
                 n_channels,
                 d_model,
                 nhead,
                 dim_feedforward,
                 blocks,
                 mlp_head_units,
                 n_classes):
        super().__init__()
        self.img2seq = Img2Seq(img_size, patch_size, n_channels, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    activation='gelu',
                                                    batch_first=True
                                                    )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, blocks)
        self.mlp = get_mlp(d_model, mlp_head_units, n_classes)

        self.output = nn.Sigmoid() if n_classes == 1 else nn.Softmax()

    def __call__(self, batch):
        batch = self.img2seq(batch)
        batch = self.transformer_encoder(batch)
        batch = batch[:, 0, :]  ## take the CLS token output
        batch = self.mlp(batch)
        output = self.output(batch)
        return output
    
        