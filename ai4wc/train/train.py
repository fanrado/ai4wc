## Data loader
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import numpy as np
import zlib
from PIL import Image
from ai4wc.models.nuViT import nuViT
import torchvision.transforms as transforms

class CVNDataset(Dataset):
    def __init__(self, root, img_size=(512, 512)):
        self.root = root
        self.img_size = img_size
        self.plane = None
        self.entries = []
        self.cand_flavs = ['numu', 'nue', 'nc']
        for flav in self.cand_flavs:
            d = os.path.join(root, flav)
            if not os.path.isdir(d):
                continue
            for gz in sorted(glob(os.path.join(d, "event*.gz"))):
                key = os.path.splitext(os.path.basename(gz))[0].replace("event", "")
                self.entries.append((flav, key, gz))
    
    def classes(self):
        return self.cand_flavs
    
    def __len__(self):
        return len(self.entries)
    
    def _read_array(self, gz_path):
        with open(gz_path, 'rb') as f:
            arr = np.frombuffer(bytearray(zlib.decompress(f.read())), dtype=np.uint8).reshape(3, 500, 500)
        return arr
    
    def _to_pil(self, arr3):
        idx_map = {"U": 0, "V": 1, "Z": 2, "0": 0, "1": 1, "2": 2}
        _plane = "Z"
        idx = idx_map[_plane]
        plane = arr3[idx]
        img = np.repeat(plane[..., None], 3, axis=2)
        return Image.fromarray(img, mode='RGB')

    def get_eventinfo(self, info_path):
        path = info_path
        ret = {}
        with open(path, 'rb') as info_file:
            info = info_file.readlines()
            ret['NuPDG'] = int(info[7].strip())
            ret['NuEnergy'] = float(info[1])
            ret['LepEnergy'] = float(info[2])
            ret['Interaction'] = int(info[0].strip()) % 4
            ret['NProton'] = int(info[8].strip())
            ret['NPion'] = int(info[9].strip())
            ret['NPiZero'] = int(info[10].strip())
            ret['NNeutron'] = int(info[11].strip())
            #ret['OscWeight'] = float(info[6])
        return ret
    
    def __getitem__(self, idx):
        _, _, gz = self.entries[idx]
        arr = self._read_array(gz)
        img = self._to_pil(arr)
        resize_to_torch = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = resize_to_torch(img)
        # target = 0  # dummy label for SSL
        target = self.get_eventinfo(gz.replace('.gz', '.info'))

        if target['NuPDG'] in [14, -14]:   # numuCC
            target = 0
        elif target['NuPDG'] in [12, -12]: # nueCC
            target = 1
        else:                              # NC
            target = 2
        return img, target
    

from ai4wc.models.nuViT import nuViT
import torch

def run_on_train_dataset(train_loader, model, device, loss_fn, optimizer):
    train_loss_epoch = 0.0
    train_accuracy_epoch = 0.0
    total_samples = 0
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        train_accuracy_epoch += (outputs.argmax(dim=1) == targets).sum().item()
        train_loss_epoch += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_samples += len(targets)
    train_loss_epoch /= total_samples
    train_accuracy_epoch /= total_samples
    return train_loss_epoch, train_accuracy_epoch

def run_validation(val_loader, model, device, loss_fn):
    model.eval()
    val_loss_epoch = 0.0
    val_accuracy_epoch = 0.0
    total_samples = 0
    for images, targets in val_loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, targets)
        val_accuracy_epoch += (outputs.argmax(dim=1) == targets).sum().item()
        val_loss_epoch += loss.item()
        total_samples += len(targets)
    val_loss_epoch /= total_samples
    val_accuracy_epoch /= total_samples
    return val_loss_epoch, val_accuracy_epoch

if __name__ == '__main__':
    path_to_train = '/home/rrazakami/work/WC/ai4wc/notebooks/data_cvn/train'
    path_to_val = '/home/rrazakami/work/WC/ai4wc/notebooks/data_cvn/val'
    path_to_test = '/home/rrazakami/work/WC/ai4wc/notebooks/data_cvn/test'

    img_size = (512, 512)
    patch_size = (16, 16)
    n_channels = 3
    d_model = 1024
    nhead = 4
    dim_feedforward = 2048
    blocks = 8
    mlp_head_units = [1024, 512]
    n_classes = 3
    BATCH_SIZE = 8
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset = CVNDataset(root=path_to_train, img_size=img_size)
    val_dataset = CVNDataset(root=path_to_val, img_size=img_size)
    test_dataset = CVNDataset(root=path_to_test, img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = nuViT(
        img_size = img_size,
        patch_size = patch_size,
        d_model = d_model,
        n_channels = n_channels,
        nhead = nhead,
        dim_feedforward = dim_feedforward,
        blocks = blocks,
        mlp_head_units = mlp_head_units,
        n_classes = n_classes,
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    training_log = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [], 'epoch': []}
    for epoch in range(10):
        print(f'Epoch {epoch+1}/{10}-----')
        train_loss_epoch, train_accuracy_epoch = run_on_train_dataset(train_loader, model, device, loss_fn, optimizer)
        training_log['train_loss'].append(train_loss_epoch)
        training_log['train_accuracy'].append(train_accuracy_epoch)

        val_loss_epoch, val_accuracy_epoch = run_validation(val_loader, model, device, loss_fn)

        training_log['val_loss'].append(val_loss_epoch)
        training_log['val_accuracy'].append(val_accuracy_epoch)
        training_log['epoch'].append(epoch)
        scheduler.step()

        print(f'Training Accuracy: {train_accuracy_epoch}; Training Loss: {train_loss_epoch}')
        print(f'Validation Accuracy: {val_accuracy_epoch}; Validation Loss: {val_loss_epoch}')
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    plt.plot(training_log['epoch'], training_log['train_accuracy'], label='Train Accuracy', color='blue', linestyle='-')
    plt.plot(training_log['epoch'], training_log['val_accuracy'], label='Validation Accuracy', color='orange', linestyle='-')
    plt.plot(training_log['epoch'], training_log['train_loss'], label='Train Loss', color='blue', linestyle='--')
    plt.plot(training_log['epoch'], training_log['val_loss'], label='Validation Loss', color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.savefig('training_validation_metrics.png')
    plt.close()
    