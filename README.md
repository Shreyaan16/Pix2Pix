Generator
---------

Architecture (U-Net generator with skip connections):

- Input: 3-channel image (default 256x256).
- Initial downsample: Conv2d(in_channels=3, features=64, kernel=4, stride=2, padding=1, padding_mode="reflect") + LeakyReLU(0.2)
- Encoder blocks (each block = Conv2d -> BatchNorm -> LeakyReLU):
  - down1: 64 -> 128
  - down2: 128 -> 256
  - down3: 256 -> 512
  - down4: 512 -> 512
  - down5: 512 -> 512
  - down6: 512 -> 512
- Bottleneck: Conv2d(512 -> 512, kernel=4, stride=2, padding=1) + ReLU
- Decoder blocks (each block = ConvTranspose2d -> BatchNorm -> ReLU, optional Dropout on first 3 ups):
  - up1: 512 -> 512 (use_dropout=True)
  - up2: 1024 -> 512 (concat skip) (use_dropout=True)
  - up3: 1024 -> 512 (use_dropout=True)
  - up4: 1024 -> 512
  - up5: 1024 -> 256
  - up6: 512 -> 128
  - up7: 256 -> 64
- Final upsampling: ConvTranspose2d(128 -> 3, kernel=4, stride=2, padding=1) + Tanh (output range [-1, 1])

Notes: skip connections concatenate encoder feature maps to decoder inputs, matching the standard pix2pix U-Net.

Discriminator
-------------

Architecture (PatchGAN discriminator):

- Input: concatenation of input and target/generated image along channel dimension (in_channels * 2, default 6 channels for RGB pairs).
- Initial layer: Conv2d(in_channels=6, features=64, kernel=4, stride=2, padding=1, padding_mode="reflect") + LeakyReLU(0.2)
- Sequence of CNN blocks (Conv2d -> BatchNorm -> LeakyReLU):
  - 64 -> 128 (stride=2)
  - 128 -> 256 (stride=2)
  - 256 -> 512 (stride=1 for the last listed feature)
- Final conv: Conv2d(512 -> 1, kernel=4, stride=1, padding=1, padding_mode="reflect")

Output: a feature map where each element corresponds to a patch's real/fake score (PatchGAN).

Dataset
-------

MapDataset (paired dataset loader):

- Root: directory containing paired images where each image is a concatenation of input and target side-by-side (left = input, right = target).
- For each file:
  - Loads the full image with PIL.
  - Splits width in half: left half -> input_image, right half -> target_image.
  - Applies `config.both_transform` (resize/transform) to both images.
  - Applies synchronized horizontal flip (random seed saved and flip applied to both to keep augmentations aligned).
  - Applies `config.transform_only_input` to the input (color jitter / input-specific augmentations).
  - Applies `config.transform_only_mask` to the target (normalization, no color augmentation).
  - Returns (input_image, target_image) as tensors.

How to run locally
------------------

Prerequisites:

- Python 3.8+ and pip
- The project's dependencies listed in `requirements.txt` installed into your environment.

Basic steps (from project root):

1. Create and activate a virtual environment (optional but recommended).

2. Install dependencies:

   pip install -r requirements.txt

3. Prepare data:

   - Place paired images in `data/maps/train/` and `data/maps/val/`.
   - Each image must contain input and target side-by-side (left = input, right = target). Images should be 512x256 for a final 256x256 split (or any even width so split works).

4. Train the model:

   python train.py

Notes:

- The code automatically uses GPU if available (PyTorch CUDA). Ensure CUDA drivers are installed to use GPU.
- The transforms and augmentation behavior are defined in `config.py` (see `both_transform`, `transform_only_input`, `transform_only_mask`).
