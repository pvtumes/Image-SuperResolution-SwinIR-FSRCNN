import os
import glob
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from models.network_swinir import SwinIR  # Ensure this path matches your project structure


# ✅ Dataset class for GoPro
class GoProDataset(Dataset):
    def __init__(self, root_dir, patch_size=64, scale=4):
        super().__init__()
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.scale = scale

        # Get sorted LR and HR images
        self.lr_images = sorted(glob.glob(os.path.join(root_dir, '*/blur/*')))
        self.hr_images = sorted(glob.glob(os.path.join(root_dir, '*/sharp/*')))

        assert len(self.lr_images) == len(self.hr_images), "Mismatch between LR and HR image counts."

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        img_lq = Image.open(self.lr_images[idx]).convert('RGB')
        img_gt = Image.open(self.hr_images[idx]).convert('RGB')

        # Random crop within bounds
        i = torch.randint(0, img_lq.height - self.patch_size + 1, (1,)).item()
        j = torch.randint(0, img_lq.width - self.patch_size + 1, (1,)).item()
        h, w = self.patch_size, self.patch_size

        # Crop LR and corresponding HR patch
        img_lq = TF.crop(img_lq, i, j, h, w)
        img_gt = TF.crop(img_gt, i * self.scale, j * self.scale, h * self.scale, w * self.scale)

        # Convert to tensor
        img_lq = TF.to_tensor(img_lq)
        img_gt = TF.to_tensor(img_gt)

        return img_lq, img_gt


# ✅ Command line argument parser
def get_args():
    parser = argparse.ArgumentParser(description='Train SwinIR on GoPro dataset')
    parser.add_argument('--train_folder', type=str, required=True, help='Path to training folder')
    parser.add_argument('--patch_size', type=int, default=64, help='LR patch size')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--pretrained_weights', type=str, default=None)
    return parser.parse_args()


# ✅ Training loop
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = GoProDataset(args.train_folder, patch_size=args.patch_size, scale=args.scale)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = SwinIR(
    upscale=args.scale,            # ✅ explicitly tell model to do x4 upscaling
    in_chans=3,
    img_size=args.patch_size,      # LR patch size
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='pixelshuffle',
    resi_connection='1conv'
    ).to(device)


    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ✅ Load pretrained weights (optional)
    if args.pretrained_weights is not None:
        print(f'Loading pretrained weights from {args.pretrained_weights}')
        try:
            state_dict = torch.load(args.pretrained_weights, map_location=device)  # weights_only=True optional
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights. {e}")

    # ✅ Training loop
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for i, (img_lq, img_gt) in enumerate(dataloader):
            img_lq, img_gt = img_lq.to(device), img_gt.to(device)

            optimizer.zero_grad()
            output = model(img_lq)
            loss = criterion(output, img_gt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}/{args.epochs}] Average Loss: {epoch_loss / len(dataloader):.4f}')

        # Save checkpoint
        checkpoint_path = f'checkpoints/swinir_epoch_{epoch+1}.pth'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')


# ✅ Run training
if __name__ == '__main__':
    args = get_args()
    train(args)
