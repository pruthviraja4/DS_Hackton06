import argparse, os, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image
from src.utils.logger import get_logger
logger = get_logger('train_unet')

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for f in features:
            self.downs.append(nn.Sequential(nn.Conv2d(in_channels, f, kernel_size=3, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(f, f, kernel_size=3, padding=1),
                                            nn.ReLU()))
            in_channels = f
        for f in reversed(features):
            self.ups.append(nn.Sequential(nn.ConvTranspose2d(in_channels, f, kernel_size=2, stride=2),
                                          nn.ReLU()))
            in_channels = f
        self.final = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self,x):
        skips=[]
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = nn.functional.max_pool2d(x,2)
        for up in self.ups:
            x = up(x)
        return torch.sigmoid(self.final(x))

class ImageSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images = sorted([str(p) for p in Path(images_dir).glob('**/*.png')])
        self.masks = sorted([str(p) for p in Path(masks_dir).glob('**/*.png')])
        self.transform = transform
    def __len__(self):
        return min(len(self.images), len(self.masks))
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        m = Image.open(self.masks[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
            m = self.transform(m)
        img = T.ToTensor()(img)
        m = T.ToTensor()(m)
        return img, m

def train_loop(model, loader, optim, loss_fn, device):
    model.train()
    total_loss=0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss/len(loader)

def main(data_dir, epochs, batch_size, output):
    # Data dir expected: images/ and masks/ inside data_dir
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        raise FileNotFoundError('Expected images/ and masks/ in '+data_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device: '+str(device))
    transform = T.Compose([T.Resize((128,128))])
    ds = ImageSegDataset(images_dir, masks_dir, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = SimpleUNet(in_channels=3, out_channels=1)
    model = model.to(device)
    optim = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    for epoch in range(epochs):
        t0 = time.time()
        loss = train_loop(model, loader, optim, loss_fn, device)
        logger.info(f'Epoch {epoch+1}/{epochs} loss={loss:.4f} time={time.time()-t0:.1f}s')
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(model.state_dict(), output)
    logger.info('Saved UNet to '+output)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir', default='data/raw/sen12flood')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--output', default='models/unet_flood.pt')
    args=p.parse_args()
    main(args.data_dir, args.epochs, args.batch_size, args.output)
