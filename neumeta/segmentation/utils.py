from PIL import Image
import torch
import numpy as np
import cv2
import wandb
import matplotlib.pyplot as plt
import os

def mask_to_color(mask, colormap_array, ignore_index=255):
    """
    Convert a segmentation mask (numpy array) to a color mask using the provided colormap.

    Parameters:
    - mask (numpy array): The segmentation mask to convert. Must be a 2D array with integer types.
    - colormap: A LinearSegmentedColormap instance from matplotlib.

    Returns:
    - A color mask as a PIL Image.
    """
    # Ensure the mask is of type integer, if not convert it
    mask = mask.astype('int')
    mask[mask == ignore_index] = 0

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Map each label to its corresponding color
    for label in range(colormap_array.shape[0]):
        color_mask[mask == label] = colormap_array[label]

    return color_mask


def create_pascal_voc_colormap():
    from matplotlib.colors import LinearSegmentedColormap
    """
    Create a colormap for PASCAL VOC segmentation maps.
    Returns a LinearSegmentedColormap instance.
    """
    colormap = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        [0, 64, 128],
    ], dtype=np.uint8)
    return LinearSegmentedColormap.from_list('pascal_voc', colormap / 255.0, N=colormap.shape[0]), colormap


def concat_pil_images(img1, img2, img3):
    """
    Concatenate three PIL images side by side (horizontally).

    Parameters:
    - img1, img2, img3: PIL.Image instances.

    Returns:
    - A new PIL.Image instance containing the three images side by side.
    """
    # Ensure all images are the same height
    total_width = img1.width + img2.width + img3.width
    max_height = max(img1.height, img2.height, img3.height)

    # Create a new image with a width to hold all three images side by side
    new_img = Image.new('RGB', (total_width, max_height))

    # Paste each image into the new image
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    new_img.paste(img3, (img1.width + img2.width, 0))

    return new_img


def visualize_model_wandb(model, valid_loader, DEVICE):
    model.eval()

    _, colormap_array = create_pascal_voc_colormap()
    data_vis = valid_loader.dataset
    model.eval()
    with torch.no_grad():
        for i in range(5):
            n = np.random.choice(len(data_vis))
            image_vis = cv2.imread(data_vis.images[n])
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)
            image, gt_mask = data_vis[n]

            gt_mask = gt_mask.cpu().numpy()
            x_tensor = image.to(DEVICE).unsqueeze(0)
            pr_mask = model(x_tensor)
            pr_mask = torch.argmax(
                pr_mask.squeeze(), dim=0).detach().cpu().numpy()

            # Convert the tensors to PIL images
            image_pil = Image.fromarray(image_vis).convert("RGB")
            gt_mask_colored = Image.fromarray(
                mask_to_color(gt_mask, colormap_array))
            pr_mask_colored = Image.fromarray(
                mask_to_color(pr_mask, colormap_array))
            combined_image = concat_pil_images(
                image_pil, gt_mask_colored, pr_mask_colored)
            # Log the images to wandb
            wandb.log({
                "Image": wandb.Image(combined_image, caption="Combined Image")
            })

def save_checkpoint(model, optimizer, epoch, scores, checkpoint_dir, filename='checkpoint.pth.tar'):
    """
    Save a training checkpoint.

    Parameters:
    - model (torch.nn.Module): The model to save.
    - optimizer (torch.optim.Optimizer): The optimizer to save.
    - epoch (int): The current epoch number.
    - scores (dict): A dictionary of scores/metrics.
    - checkpoint_dir (str): Directory where to save the checkpoint.
    - filename (str): Name of the checkpoint file.
    """
    # Ensure the checkpoint directory exists; if not, create it
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Construct the checkpoint object
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scores': scores
    }

    # Save the checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to '{checkpoint_path}' at epoch {epoch}.")


def visualize(color_map, name='image', **images):
    
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (im_name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(im_name.split('_')).title())
        if im_name == 'image':
            plt.imshow(image)
        else:
            image[image == 255] = 0
            # print(np.unique(image))
            plt.imshow(image, cmap=color_map, interpolation='nearest')
        plt.axis('off')
    # plt.show()
    plt.savefig(f'seg_results{name}.png')