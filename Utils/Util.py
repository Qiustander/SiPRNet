from .colormap import cm_data
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


# Define colormap. Use defaualt colormap in MATLAB imagesc function
parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
def save_fig(fig_name, img):
    # plot figure
    plt.figure(dpi=150)
    plt.figure(figsize=(3, 3))
    im = plt.imshow(img, cmap=parula_map, vmin=0, vmax=2*np.pi)
    ax = plt.subplot()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(fig_name)
    plt.close()


def save_tensor_img(img_tensor, save_file_prefix, isGT=False):
    angle_target = np.angle(img_tensor[0].detach().cpu().numpy()[0, :, :]+
                        1j*img_tensor[0].detach().cpu().numpy()[1, :,
                        :]) + np.pi
    if not isGT:
        save_fig(save_file_prefix + '_angle.png', angle_target)
    else:
        save_fig(save_file_prefix + '_angle_gt.png', angle_target)
    return