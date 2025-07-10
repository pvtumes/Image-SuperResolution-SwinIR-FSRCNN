import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from models.network_swinir import SwinIR  # Adjust import path if needed

def load_model(model_path, device, scale=4):
    model = SwinIR(
        img_size=64,  # typical patch size used in training
        window_size=8,
        img_channel=3,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        scale=scale
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    img_t = transform(img).unsqueeze(0)  # add batch dimension
    return img_t

def save_image(tensor, output_path):
    tensor = tensor.squeeze(0).clamp(0, 1).cpu()
    img = transforms.ToPILImage()(tensor)
    img.save(output_path)
    print(f"Saved SR image to {output_path}")

def run_sr(image_path, model, device, output_path):
    img_lq = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(img_lq)
    save_image(output, output_path)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="SwinIR inference pipeline")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint .pth')
    parser.add_argument('--input', type=str, required=True, help='Input low-quality image path')
    parser.add_argument('--output', type=str, default='output.png', help='Output image path')
    parser.add_argument('--scale', type=int, default=4, help='Upscaling factor')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device, scale=args.scale)
    run_sr(args.input, model, device, args.output)
