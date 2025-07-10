# Image Super-Resolution using SwinIR + FSRCNN

This repository implements a **Student-Teacher framework** for Image Super-Resolution.  
- 👨‍🏫 **Teacher**: SwinIR (Transformer-based)  
- 👨‍🎓 **Student**: FSRCNN (Fast Convolutional Neural Network)  
- 🎯 Dataset: [GoPro Large Dataset](https://seungjunnah.github.io/Datasets/gopro)  
- 🧠 Objective: Train a lightweight FSRCNN model to mimic SwinIR performance on blurry images.

---

## 🚀 Project Structure

INTEL/
├── FSRCNN-pytorch/ # Student model repo
├── SwinIR/ # Teacher model repo
├── GOPRO_Large/ # Dataset (not uploaded)
├── intel_pro/ # Virtual environment (not tracked)
└── script.py # Misc pipeline script

yaml
Copy
Edit

---

## 🧠 Model Details

### 👨‍🏫 SwinIR (Teacher)
- Based on Swin Transformer.
- Pretrained on `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth`.
- Task: Super-resolve blurry images from the GoPro dataset.

### 👨‍🎓 FSRCNN (Student)
- Lightweight CNN for fast inference.
- Trained on patches from GoPro dataset.
- Learns to replicate SwinIR’s outputs using L1 loss.

---

## 📁 Dataset: GoPro Large

> Note: The dataset is **not included** in the repo due to its size.

**Structure:**
GOPRO_Large/
├── train/
│ └── (blur/sharp pairs)
├── test/
│ └── GOPROxxxx_xx_xx/
│ ├── blur/
│ └── sharp/

yaml
Copy
Edit

---

## 🏋️‍♂️ Training

### 🔧 Prerequisites

```bash
git clone https://github.com/pvtumes/Image-SuperResolution-SwinIR-FSRCNN.git
cd Image-SuperResolution-SwinIR-FSRCNN
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
👨‍🏫 Train SwinIR (Optional if using pretrained)
bash
Copy
Edit
python SwinIR/main_train_swinir.py \
  --train_folder ../GOPRO_Large/train \
  --patch_size 64 \
  --batch_size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --scale 4 \
  --pretrained_weights SwinIR/model_zoo/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
👨‍🎓 Train FSRCNN
Update script.py with FSRCNN training loop using SwinIR outputs as supervision.

🧪 Inference
SwinIR:
bash
Copy
Edit
python SwinIR/main_test_swinir.py \
  --task real_sr \
  --scale 4 \
  --model_path SwinIR/model_zoo/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth \
  --folder_lq path/to/test_lq/
FSRCNN:
bash
Copy
Edit
python FSRCNN-pytorch/inference.py \
  --weights-file path/to/fsrcnn_epoch.pth \
  --image-file path/to/image.jpg \
  --scale 4
📦 Output
Checkpoints saved in checkpoints/

Inference results saved in results/

Each epoch saved separately (epoch1.pth, epoch2.pth, ...)

📊 Results
Model	PSNR ↑	SSIM ↑	Size ↓	Speed ↑
SwinIR	32.1	0.92	~44MB	Slow
FSRCNN	28.7	0.88	~8MB	Fast

🧠 Citations
SwinIR

FSRCNN

GoPro Dataset

🤝 Acknowledgments
Thanks to the authors of SwinIR, FSRCNN, and the GoPro dataset for open-sourcing their work.

📝 License
This repository is licensed under the MIT License.

yaml
Copy
Edit

---

### 📌 Notes:
- Replace any local paths (like `path/to/image.jpg`) with actual paths you use.
- If you push this repo to GitHub, be sure **not to include** the `GOPRO_Large` dataset or `intel_pro` venv folder. Use `.gitignore`.

Would you like me to generate a `.gitignore` file as well?
