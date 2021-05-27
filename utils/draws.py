from PIL import Image, ImageDraw
import numpy as np
import torch
import utils.transforms


def text_to_image(labels, width, height=50):
    images_tensor = torch.zeros((len(labels), 3, height, width))
    for i, label in enumerate(labels):
        img = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        w, h = draw.textsize(label)
        draw.text(((width - w) // 2, (height - h) // 2), label, fill=(0, 0, 0))
        images_tensor[i] = torch.from_numpy(np.array(img)).permute(2, 0, 1) / 255
    return images_tensor


def add_border(x, mask):
    """Adds a border given by mask to x.

    Args:
        x (np.Array): array of size (C,F,H,W) containing the image.
        mask (np.Array): array of size (1,H,W) containing the mask of the border.

    Returns:
        torch.FloatTensor: images with the border applied of size (C,F,H,W).
    """
    mask_dilated = utils.transforms.ImageTransforms.dilatate(
        torch.from_numpy(mask).unsqueeze(1).repeat(1, x.shape[1], 1, 1), (3, 3), 1
    )
    border = ((mask_dilated.numpy() - np.expand_dims(mask, axis=1).repeat(x.shape[1], axis=1)) > 0)
    return x * (1 - border) + border.astype(np.float)
