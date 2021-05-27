import cv2
import random
import torch
import torch.nn.functional as F


def resize_set(x, v, y, size):
    """Resizes the entire set of data (x, m, y).

    Args:
        x (torch.FloatTensor): tensor of size (B,C,F,H,W) containing masked images quantized from [0, 1].
        v (torch.FloatTensor): tensor of size (B,1,F,H,W) containing visibility maps quantized from [0, 1].
        y (torch.FloatTensor): tensor of size (B,C,F,H,W) containing images from [0, 1].
        size (int): new size of the set of data.
    """
    b, c, f, h, w = x.size()
    x_new = F.interpolate(x.transpose(1, 2).reshape(-1, c, h, w), (size, size), mode='bilinear'). \
        reshape(b, f, c, size, size).transpose(1, 2)
    m_new = F.interpolate(v.transpose(1, 2).reshape(-1, 1, h, w), (size, size)). \
        reshape(b, f, 1, size, size).transpose(1, 2)
    y_new = F.interpolate(y.transpose(1, 2).reshape(-1, c, h, w), (size, size), mode='bilinear'). \
        reshape(b, f, c, size, size).transpose(1, 2)
    return x_new, m_new, y_new


class ImageTransforms:

    @staticmethod
    def resize(image, size, mode='bilinear', keep_ratio=True):
        """Resize an image using the the algorithm given in ``mode``.

        Args:
            image (torch.FloatTensor): tensor of size (C,F,H,W) containing the image quantized from [0, 1].
            size (tuple): tuple containing the desired size in the form (H, W).
            mode (str): mode used to resize the image. Same format as in ``torch.nn.functional.interpolate()``.

        Returns:
            torch.FloatTensor: resized image.
        """
        if keep_ratio and size[1] == -1:
            new_height = size[0]
            new_width = round(image.size(3) * size[0] / image.size(2))
            new_size = (new_height, new_width)
            return F.interpolate(image.transpose(0, 1), new_size, mode=mode).transpose(0, 1)[:, :, :size[0], :size[1]]
        elif keep_ratio:
            new_height = size[0] if image.size(2) < image.size(3) else round(image.size(2) * size[1] / image.size(3))
            new_width = size[1] if image.size(3) <= image.size(2) else round(image.size(3) * size[0] / image.size(2))
            new_size = (new_height, new_width)
            return F.interpolate(image.transpose(0, 1), new_size, mode=mode).transpose(0, 1)[:, :, :size[0], :size[1]]
        else:
            return F.interpolate(image.transpose(0, 1), size, mode=mode).transpose(0, 1)

    @staticmethod
    def crop(image, size, crop_center=True, crop_position=None):
        """Crop a patch from the image.

        Args:
            image (torch.FloatTensor): tensor of size (C, F, H, W) containing the image.
            size (tuple): tuple containing the desired size in the form (H, W).
            crop_position (tuple): coordinates of the top-left pixel from where to cut the patch. If not set, it is
            generated randomly.

        Returns:
            torch.FloatTensor: patch of the image.
        """
        if crop_position is None and crop_center:
            crop_position = ((image.size(2) - size[0]) // 2, (image.size(3) - size[1]) // 2)
        elif crop_position is None:
            crop_position = (random.randint(0, image.size(2) - size[0]), random.randint(0, image.size(3) - size[1]))
        return image[:, :, crop_position[0]:crop_position[0] + size[0], crop_position[1]:crop_position[1] + size[1]], \
               crop_position

    @staticmethod
    def binarize(image, threshold=0.5):
        """Binarizes an image using ``threshold``.

        Args:
            image (torch.FloatTensor): tensor of size (C, H, W) containing the image.
            threshold (float): value used to binarize the image.

        Returns:
            torch.FloatTensor: binary image containing only 0's and 1s.
        """
        return (torch.sum(image, dim=0) > threshold).type(torch.float32)

    @staticmethod
    def dilatate(images, filter_size, iterations):
        """Dilatates an image with a filter of size ``filter_size``.

        Args:
            image (torch.FloatTensor): tensor of size (1,F,H,W) containing the image.
            filter_size (tuple): size of the filter in the form (H,W).
            iterations (integer): number of times to apply the filter.

        Returns:
            torch.FloatTensor: dilatated image.
        """
        images_dilated = torch.zeros(images.size())
        for f in range(images.size(1)):
            images_dilated[:, f] = torch.from_numpy(cv2.dilate(
                images[:, f].permute(1, 2, 0).numpy(),
                cv2.getStructuringElement(cv2.MORPH_CROSS, filter_size),
                iterations=iterations
            )).unsqueeze(0).float()
        return images_dilated
