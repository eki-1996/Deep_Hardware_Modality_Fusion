import os
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np

image_path = "./datasets/multimodal_dataset/polL_color"
aolp_cos_path = "./datasets/multimodal_dataset/polL_aolp_cos"
aolp_sin_path = "./datasets/multimodal_dataset/polL_aolp_sin"
dolp_path = "./datasets/multimodal_dataset/polL_dolp"
nir_path = "./datasets/multimodal_dataset/NIR_warped"
label_path = "./datasets/multimodal_dataset/GT"

save_path = "./datasets/multimodal_dataset/"

data_file_path = "./datasets/multimodal_dataset/list_folder/all.txt"

data_list = []

# # following code is for visualize L1 error and L2 error.
# with open(save_path + 'polarization_error_L1.txt', "r") as f:
#     lines = f.read().splitlines()
#     aolp = []
#     dolp = []
#     for line in lines:
#         aolp_str, dolp_str = line.split(',')
#         aolp.append(float(aolp_str))
#         dolp.append(float(dolp_str))
#     plt.figure()
#     plt.plot([x for x in range(len(aolp))], aolp)
#     plt.plot([x for x in range(len(aolp))], [np.mean(np.asarray(aolp)) for x in range(len(aolp))])
#     plt.title("aolp L1 err")

#     plt.figure()
#     plt.plot([x for x in range(len(dolp))], dolp)
#     plt.plot([x for x in range(len(aolp))], [np.mean(np.asarray(dolp)) for x in range(len(aolp))])
#     plt.title("dolp L1 err")

#     plt.show()

#     print(np.mean(np.asarray(aolp))/np.pi)
#     print(np.mean(np.asarray(dolp)))

# exit()

if not os.path.exists(save_path + "pol_I000"):
    os.mkdir(save_path + "pol_I000")
if not os.path.exists(save_path + "pol_I045"):
    os.mkdir(save_path + "pol_I045")
if not os.path.exists(save_path + "pol_I090"):
    os.mkdir(save_path + "pol_I090")
if not os.path.exists(save_path + "pol_I135"):
    os.mkdir(save_path + "pol_I135")

with open(data_file_path, "r") as f:
    lines = f.read().splitlines()
    for line in lines:
        data_list.append(line)

for label in data_list:
    img = cv2.imread(os.path.join(image_path , label + ".png"),-1)[:,:,::-1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    aolp_cos = np.load(os.path.join(aolp_cos_path, label + ".npy"))
    aolp_sin = np.load(os.path.join(aolp_sin_path, label + ".npy"))

    dolp = np.load(os.path.join(dolp_path, label + ".npy"))
    dolp = np.clip(dolp, 0, 1).astype(np.float64)

    aolp = np.mod(np.arctan2(aolp_sin, aolp_cos), np.pi)
    aolp = np.clip(aolp, 0, np.pi).astype(np.float64)
    
    reconstract_I000 = img_gray * (1 + dolp * np.cos(2 * (0 - aolp)))
    reconstract_I045 = img_gray * (1 + dolp * np.cos(2 * (np.pi/4 - aolp)))
    reconstract_I090 = img_gray * (1 + dolp * np.cos(2 * (np.pi/2 - aolp)))
    reconstract_I135 = img_gray * (1 + dolp * np.cos(2 * (np.pi*3/4 - aolp)))

    reconstract_I000 = np.clip(reconstract_I000, 0, 65535).astype(np.uint16)
    reconstract_I045 = np.clip(reconstract_I045, 0, 65535).astype(np.uint16)
    reconstract_I090 = np.clip(reconstract_I090, 0, 65535).astype(np.uint16)
    reconstract_I135 = np.clip(reconstract_I135, 0, 65535).astype(np.uint16)

    cv2.imwrite(save_path + "pol_I000/" + label + ".png", reconstract_I000)
    cv2.imwrite(save_path + "pol_I045/" + label + ".png", reconstract_I045)
    cv2.imwrite(save_path + "pol_I090/" + label + ".png", reconstract_I090)
    cv2.imwrite(save_path + "pol_I135/" + label + ".png", reconstract_I135)

    reconstract_I000 = reconstract_I000.astype(np.float64)
    reconstract_I045 = reconstract_I045.astype(np.float64)
    reconstract_I090 = reconstract_I090.astype(np.float64)
    reconstract_I135 = reconstract_I135.astype(np.float64)

    S0 = 0.5 * (reconstract_I000 + reconstract_I090 + reconstract_I045 + reconstract_I135).astype(np.float64)
    S1 = (reconstract_I000 - reconstract_I090).astype(np.float64)
    S2 = (reconstract_I045 - reconstract_I135).astype(np.float64)

    S0 = S0 + sys.float_info.min * (S0==0)
    S1 = S1 + sys.float_info.min * (S1==0)

    reconstract_dolp = np.sqrt(S1**2 + S2**2) / S0
    reconstract_aolp = np.mod(0.5 * np.arctan2(S2, S1), np.pi)

    with open(save_path + 'polarization_error_L1.txt', 'a') as f:
        f.writelines(f"{np.sum(np.abs(aolp-reconstract_aolp))/(aolp.shape[0]*aolp.shape[0])},{np.sum(np.abs(dolp-reconstract_dolp))/(dolp.shape[0]*dolp.shape[0])}\n")
    
    with open(save_path + 'polarization_error_L2.txt', 'a') as f:
        f.writelines(f"{np.sum((aolp-reconstract_aolp)**2)/(aolp.shape[0]*aolp.shape[0])},{np.sum((dolp-reconstract_dolp)**2)/(dolp.shape[0]*dolp.shape[0])}\n")