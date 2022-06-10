import numpy as np
import torch


def blend(img, labels, num_classes):
    colors = torch.tensor([[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [
                          255, 0, 255], [0, 255, 255], [255, 255, 0]]).cuda(img.device).float()

    img = img[..., None].repeat(1, 1, 1, 3)
    masks = torch.zeros_like(img)
    for cls in range(1, num_classes):
        masks += torch.ones_like(img) * \
            colors[cls] * (labels == cls).float()[:, :, :, None]

    overlay = np.uint8((255 * img * 0.8 + masks * 0.2).data.cpu().numpy())
    return overlay


def blend_cpu(img, labels, num_classes):
    colors = torch.tensor([[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [
                          255, 0, 255], [0, 255, 255], [255, 255, 0]]).float()

    img = img[..., None].repeat(1, 1, 1, 3)
    masks = torch.zeros_like(img)
    for cls in range(1, num_classes):
        masks += torch.ones_like(img) * \
            colors[cls] * (labels == cls).float()[:, :, :, None]

    overlay = np.uint8((255 * img * 0.8 + masks * 0.2).data.numpy())
    return overlay
