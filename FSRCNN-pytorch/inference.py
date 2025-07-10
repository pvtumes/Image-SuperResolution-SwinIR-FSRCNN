import torch
from PIL import Image
import torchvision.transforms as transforms
from models import FSRCNN  # Adjust if your FSRCNN model is defined elsewhere

weights_path = 'weights/fsrcnn_x4.pth'
input_image_path = 'data/0014.jpg'              # Correct path to your input image
output_image_path = 'thumbnails/fsrcnn_output.png'

def load_model(weights_path, scale_factor=4):
    model = FSRCNN(scale_factor=scale_factor)
    checkpoint = torch.load(weights_path, map_location='cpu')  # Use 'cuda' if GPU available
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    to_tensor = transforms.ToTensor()
    input_tensor = to_tensor(img).unsqueeze(0)  # add batch dimension
    return input_tensor

def postprocess_and_save(tensor, save_path):
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor.squeeze(0).clamp(0,1))  # remove batch, clamp between 0-1
    img.save(save_path)
    print(f'Saved super-resolved image to {save_path}')

def main():
    model = load_model(weights_path)
    input_tensor = preprocess_image(input_image_path)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    postprocess_and_save(output_tensor, output_image_path)

if __name__ == '__main__':
    main()
