import matplotlib.pyplot as plt
import numpy as np
import torch

    
def get_three_channels_visulized_image(sample):
    if 'rgb' in sample.keys():
        return sample['rgb']
    else:
        concatnated_image = torch.cat(list(sample.valus()), dim=1)
        rgb = [None, None, None]
        for i in range(int(concatnated_image.shape[1]/3)):
            if rgb[0] is None:
                rgb[0] = concatnated_image[:,i,:,:]
                rgb[1] = concatnated_image[:,i+1,:,:]
                rgb[2] = concatnated_image[:,i+2,:,:]
            else:
                rgb[0] = torch.cat((rgb[0], concatnated_image[:,i*3,:,:]), dim=1)
                rgb[1] = torch.cat((rgb[1], concatnated_image[:,i*3+1,:,:]), dim=1)
                rgb[2] = torch.cat((rgb[2], concatnated_image[:,i*3+2,:,:]), dim=1)
        
        for j in range(concatnated_image.shape[1]%3):
            rgb[j] = torch.cat((rgb[j], concatnated_image[:,-(concatnated_image.shape[1]%3-j),:,:]), dim=1)

        return torch.cat([torch.unsqueeze(torch.mean(x, dim=1), dim=1) for x in rgb], dim=1)


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'kitti' or dataset == 'kitti_advanced' or dataset == 'kitti_advanced_manta' \
            or dataset == 'handmade_dataset' or dataset == 'handmade_dataset_stereo' or dataset == 'multimodal_dataset' or dataset == 'multimodal_dataset_4polarization':
        n_classes = 20
        label_colours = get_my_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    elif dataset == 'rgb_thermal_dataset':
        n_classes = 9
        label_colours = get_rgb_thermal_dataset_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    r[label_mask == 255] = 0
    g[label_mask == 255] = 0
    b[label_mask == 255] = 0
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])

def get_my_labels():
    " r,g,b"
    return np.array([
        [ 44, 160,  44], # asphalt
        [ 31, 119, 180], # concrete
        [255, 127,  14], # metal
        [214,  39,  40], # road marking
        [140,  86,  75], # fabric, leather
        [127, 127, 127], # glass
        [188, 189,  34], # plaster
        [255, 152, 150], # plastic
        [ 23, 190, 207], # rubber
        [174, 199, 232], # sand
        [196, 156, 148], # gravel
        [197, 176, 213], # ceramic
        [247, 182, 210], # cobblestone
        [199, 199, 199], # brick
        [219, 219, 141], # grass
        [158, 218, 229], # wood
        [ 57,  59, 121], # leaf
        [107, 110, 207], # water
        [156, 158, 222], # human body
        [ 99, 121,  57]]) # sky

def get_rgb_thermal_dataset_labels():
    unlabelled = [0,0,0]
    car        = [64,0,128]
    person     = [64,64,0]
    bike       = [0,128,192]
    curve      = [0,0,192]
    car_stop   = [128,128,0]
    guardrail  = [64,64,128]
    color_cone = [192,128,128]
    bump       = [192,64,0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette
